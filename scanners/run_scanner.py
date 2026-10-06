import argparse
import html
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = REPO_ROOT / "min10B_listed.csv"
DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "stock_data.pkl"

# ------------------------------------------------------------
# Monitoring / diagnostics
# ------------------------------------------------------------

RECENT_TRIGGER_SESSIONS = 5

# Number of daily bars displayed on each chart.
CHART_LOOKBACK_BARS = 80

# Diagnostic thresholds only.
# These are NOT currently part of the trigger rules.
RVOL_LEVELS = [1.00, 1.25, 1.50, 2.00]
ATR_LEVELS = [0.75, 1.00, 1.25, 1.50]
SPREAD_LEVELS = [0.75, 1.00, 1.25, 1.50]

# Diagnostic only:
# abs(gap) <= this value is treated as "flat".
DIAGNOSTIC_FLAT_GAP_PCT = 0.50


# ============================================================
# DATA
# ============================================================

def load_tickers():
    df = pd.read_csv(INPUT_FILE)
    tickers = df["Ticker"].dropna().astype(str).str.strip().tolist()
    print(f"Loaded {len(tickers)} tickers from {INPUT_FILE}")
    return tickers


def download_data(tickers):
    print(f"Downloading daily data for {len(tickers)} stocks...")

    return yf.download(
        tickers,
        period="1y",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=True
    )


def save_data_file(data, data_file):
    data_file = Path(data_file)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(data_file)
    print(f"Saved downloaded stock data: {data_file}")


def load_data_file(data_file):
    data_file = Path(data_file)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file does not exist: {data_file}. "
            "Run the scanner without --data-file first to download and cache the data."
        )

    print(f"Loading stock data from: {data_file}")
    data = pd.read_pickle(data_file)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Data file does not contain a pandas DataFrame: {data_file}"
        )

    return data


def get_ticker_data(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return None

        df = data[ticker].copy()
    else:
        df = data.copy()

    if df.empty:
        return None

    df = df.dropna(subset=["Close"]).copy()

    if len(df) < 200:
        return None

    # --------------------------------------------------------
    # Moving averages / performance
    # --------------------------------------------------------

    df["SMA12"] = df["Close"].rolling(12).mean()
    df["SMA22"] = df["Close"].rolling(22).mean()
    df["SMA55"] = df["Close"].rolling(55).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    # 63 trading days ~= 3 months.
    df["Perf3M"] = df["Close"].pct_change(63) * 100

    # --------------------------------------------------------
    # Volume / RVOL
    # --------------------------------------------------------

    df["AvgVolume20"] = df["Volume"].rolling(20).mean()
    df["RVOL"] = df["Volume"] / df["AvgVolume20"]

    # --------------------------------------------------------
    # True Range / ATR14
    #
    # Wilder-style ATR14.
    # --------------------------------------------------------

    previous_close = df["Close"].shift(1)

    tr_components = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs()
        ],
        axis=1
    )

    df["TR"] = tr_components.max(axis=1)

    df["ATR14"] = df["TR"].ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    df["TR_ATR"] = df["TR"] / df["ATR14"]

    # --------------------------------------------------------
    # Gap / price-action measurements
    # --------------------------------------------------------

    df["Gap"] = df["Open"] - previous_close
    df["GapPct"] = (df["Gap"] / previous_close) * 100
    df["GapATR"] = df["Gap"] / df["ATR14"]

    df["IntradayPct"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    ) * 100

    df["NetDayPct"] = (
        (df["Close"] - previous_close) / previous_close
    ) * 100

    # --------------------------------------------------------
    # Candle / spread measurements
    # --------------------------------------------------------

    df["DayRange"] = df["High"] - df["Low"]
    df["Spread"] = df["DayRange"]

    # Previous 20 sessions only.
    # Today's spread does not affect its own baseline.
    df["AvgSpread20"] = (
        df["Spread"]
        .rolling(20)
        .mean()
        .shift(1)
    )

    # 1.00 = normal
    # 1.25 = 25% wider than normal
    # 1.50 = 50% wider than normal
    # 2.00 = twice normal
    df["SpreadRatio20"] = (
        df["Spread"] / df["AvgSpread20"]
    )

    df["Spread_ATR"] = df["Spread"] / df["ATR14"]

    df["Body"] = (
        df["Close"] - df["Open"]
    ).abs()

    df["UpperWick"] = (
        df["High"] - df[["Open", "Close"]].max(axis=1)
    )

    df["LowerWick"] = (
        df[["Open", "Close"]].min(axis=1) - df["Low"]
    )

    # 0.00 = close at low
    # 0.50 = close at midpoint
    # 1.00 = close at high
    df["ClosePos"] = (
        (df["Close"] - df["Low"]) / df["DayRange"]
    ).where(df["DayRange"] > 0)

    # User's candle interpretation.
    df["CandleBullish"] = (
        df["Close"] > ((df["High"] + df["Low"]) / 2)
    )

    # Conventional open/close direction.
    df["OCBullish"] = df["Close"] > df["Open"]

    df["BodyPct"] = (
        df["Body"] / df["DayRange"]
    ).where(df["DayRange"] > 0)

    df["UpperWickBody"] = (
        df["UpperWick"] / df["Body"]
    ).where(df["Body"] > 0)

    return df


def get_common_latest_date(ticker_data):
    dates = []

    for df in ticker_data.values():
        if df is not None and not df.empty:
            dates.append(df.index[-1])

    if not dates:
        return None

    return Counter(dates).most_common(1)[0][0]


# ============================================================
# SETUP CONDITIONS
# ============================================================

def pullback_setup(row, previous=None):
    return (
        row["Perf3M"] > 15
        and row["Close"] < row["SMA12"]
        and row["Close"] >= row["SMA22"]
        and row["SMA55"] > row["SMA200"]
    )


def basing_setup(row, previous=None):
    return (
        row["Perf3M"] < -15
        and row["SMA22"] < row["SMA55"]
        and row["SMA12"] > row["SMA22"]
    )


def crossover_setup(row, previous):
    if previous is None:
        return False

    return (
        row["SMA22"] < row["SMA55"]
        and previous["SMA12"] <= previous["SMA22"]
        and row["SMA12"] > row["SMA22"]
    )


# ============================================================
# TRIGGER CONDITIONS
#
# IMPORTANT REFINEMENT:
#
# A gap-down trigger is rejected when:
#
#     Open < previous Close
# AND
#     Close < previous Close
#
# A gap-down that recovers to the prior close or higher is
# allowed.
# ============================================================

def gap_down_unrecovered(row, previous):
    if previous is None:
        return False

    return (
        row["Open"] < previous["Close"]
        and row["Close"] < previous["Close"]
    )


def trigger_gap_rule(row, previous):
    return not gap_down_unrecovered(row, previous)


def pullback_trigger(row, previous=None):
    return (
        row["Close"] > row["SMA12"]
        and row["RVOL"] > 1
        and row["Close"] > ((row["High"] + row["Low"]) / 2)
        and trigger_gap_rule(row, previous)
    )


def basing_trigger(row, previous=None):
    return (
        row["Close"] > row["SMA12"]
        and row["RVOL"] > 1
        and row["Close"] > ((row["High"] + row["Low"]) / 2)
        and trigger_gap_rule(row, previous)
    )


def crossover_trigger(row, previous=None):
    return (
        row["RVOL"] > 1
        and row["Close"] > ((row["High"] + row["Low"]) / 2)
        and trigger_gap_rule(row, previous)
    )


# ============================================================
# INVALIDATION CONDITIONS
# ============================================================

def pullback_invalidated(row, previous=None):
    return (
        row["Close"] < row["SMA22"]
        or row["SMA55"] <= row["SMA200"]
    )


def basing_invalidated(row, previous=None):
    return (
        row["SMA12"] <= row["SMA22"]
        or row["SMA22"] >= row["SMA55"]
    )


def crossover_invalidated(row, previous=None):
    return (
        row["SMA12"] <= row["SMA22"]
        or row["SMA22"] >= row["SMA55"]
    )


# ============================================================
# HISTORICAL LIFECYCLE REPLAY
#
# No state file is required.
# ============================================================

def replay_lifecycle(df, setup_fn, trigger_fn, invalid_fn):
    state = "NONE"
    setup_date = None
    event_date = None

    previous = None

    for date, row in df.iterrows():

        required = [
            row["SMA12"],
            row["SMA22"],
            row["SMA55"],
            row["SMA200"],
            row["Perf3M"],
            row["RVOL"]
        ]

        if any(pd.isna(value) for value in required):
            previous = row
            continue

        # ----------------------------------------------------
        # No current candidate
        # ----------------------------------------------------

        if state == "NONE":

            if setup_fn(row, previous):

                setup_date = date
                event_date = date

                if trigger_fn(row, previous):
                    state = "TRIGGERED"
                else:
                    state = "NEW"

        # ----------------------------------------------------
        # Candidate active
        # ----------------------------------------------------

        elif state in ("NEW", "ACTIVE"):

            # Invalidation takes precedence over trigger.
            if invalid_fn(row, previous):
                state = "INVALIDATED"
                event_date = date

            elif trigger_fn(row, previous):
                state = "TRIGGERED"
                event_date = date

            else:
                state = "ACTIVE"

        # ----------------------------------------------------
        # Triggered / invalidated are event states.
        # On next bar, reset unless a new setup occurs.
        # ----------------------------------------------------

        elif state in ("TRIGGERED", "INVALIDATED"):

            state = "NONE"
            setup_date = None
            event_date = None

            if setup_fn(row, previous):

                setup_date = date
                event_date = date

                if trigger_fn(row, previous):
                    state = "TRIGGERED"
                else:
                    state = "NEW"

        previous = row

    if state == "NONE":
        return {
            "status": "NONE",
            "setup_date": None,
            "event_date": None,
            "age": None
        }

    latest_date = df.index[-1]

    if setup_date is not None:
        age = len(df.loc[setup_date:]) - 1
    else:
        age = None

    if state == "NEW" and event_date != latest_date:
        state = "ACTIVE"

    if (
        state in ("TRIGGERED", "INVALIDATED")
        and event_date != latest_date
    ):
        state = "NONE"
        setup_date = None
        event_date = None
        age = None

    return {
        "status": state,
        "setup_date": setup_date,
        "event_date": event_date,
        "age": age
    }


# ============================================================
# BAR DETAIL HELPER
# ============================================================

def make_bar_details(df, position):
    row = df.iloc[position]

    if position == 0:
        return None

    previous = df.iloc[position - 1]

    return {
        "date": df.index[position],
        "open": row["Open"],
        "high": row["High"],
        "low": row["Low"],
        "close": row["Close"],
        "previous_close": previous["Close"],
        "gap": row["Gap"],
        "gap_pct": row["GapPct"],
        "gap_atr": row["GapATR"],
        "intraday_pct": row["IntradayPct"],
        "net_day_pct": row["NetDayPct"],
        "spread": row["Spread"],
        "avg_spread20": row["AvgSpread20"],
        "spread_ratio20": row["SpreadRatio20"],
        "spread_atr": row["Spread_ATR"],
        "midpoint": (row["High"] + row["Low"]) / 2,
        "close_pos": row["ClosePos"],
        "rvol": row["RVOL"],
        "tr_atr": row["TR_ATR"],
        "candle_bullish": row["CandleBullish"]
    }


# ============================================================
# CURRENT SCAN RESULTS
# ============================================================

def build_scan_results(data, tickers):
    results = {
        "PULLBACK": {
            "NEW": [],
            "ACTIVE": [],
            "TRIGGERED": [],
            "INVALIDATED": []
        },
        "BASING": {
            "NEW": [],
            "ACTIVE": [],
            "TRIGGERED": [],
            "INVALIDATED": []
        },
        "12/22": {
            "NEW": [],
            "ACTIVE": [],
            "TRIGGERED": [],
            "INVALIDATED": []
        }
    }

    data_issues = []
    ticker_data = {}

    lifecycle_definitions = {
        "PULLBACK": (
            pullback_setup,
            pullback_trigger,
            pullback_invalidated
        ),
        "BASING": (
            basing_setup,
            basing_trigger,
            basing_invalidated
        ),
        "12/22": (
            crossover_setup,
            crossover_trigger,
            crossover_invalidated
        )
    }

    for ticker in tickers:

        df = get_ticker_data(data, ticker)

        if df is None:
            data_issues.append(ticker)
            continue

        ticker_data[ticker] = df

        for setup_name, definitions in lifecycle_definitions.items():

            state = replay_lifecycle(
                df,
                definitions[0],
                definitions[1],
                definitions[2]
            )

            if state["status"] == "NONE":
                continue

            latest = df.iloc[-1]
            previous = df.iloc[-2]

            result = {
                "ticker": ticker,
                "status": state["status"],
                "age": state["age"],
                "setup_date": state["setup_date"],
                "event_date": state["event_date"],

                "previous_close": previous["Close"],
                "open": latest["Open"],
                "high": latest["High"],
                "low": latest["Low"],
                "close": latest["Close"],

                "gap": latest["Gap"],
                "gap_pct": latest["GapPct"],
                "gap_atr": latest["GapATR"],

                "intraday_pct": latest["IntradayPct"],
                "net_day_pct": latest["NetDayPct"],

                "spread": latest["Spread"],
                "avg_spread20": latest["AvgSpread20"],
                "spread_ratio20": latest["SpreadRatio20"],
                "spread_atr": latest["Spread_ATR"],

                "midpoint": (
                    latest["High"] + latest["Low"]
                ) / 2,

                "close_pos": latest["ClosePos"],
                "rvol": latest["RVOL"],
                "tr_atr": latest["TR_ATR"],

                "candle_bullish": latest["CandleBullish"]
            }

            results[setup_name][state["status"]].append(result)

    return results, ticker_data, data_issues


# ============================================================
# RECENT TRIGGER EVENTS
#
# Trigger day = session 0.
# Four following sessions = days 1-4.
# Total memory = 5 trading sessions.
# ============================================================

def collect_trigger_events(
    df,
    setup_fn,
    trigger_fn,
    invalid_fn,
    sessions=RECENT_TRIGGER_SESSIONS
):
    events = []

    state = "NONE"
    previous = None

    for position, (date, row) in enumerate(df.iterrows()):

        required = [
            row["SMA12"],
            row["SMA22"],
            row["SMA55"],
            row["SMA200"],
            row["Perf3M"],
            row["RVOL"]
        ]

        if any(pd.isna(value) for value in required):
            previous = row
            continue

        if state == "NONE":

            if setup_fn(row, previous):

                if trigger_fn(row, previous):
                    details = make_bar_details(
                        df,
                        position
                    )

                    if details is not None:
                        events.append(details)

                    state = "NONE"

                else:
                    state = "ACTIVE"

        elif state == "ACTIVE":

            if invalid_fn(row, previous):
                state = "NONE"

            elif trigger_fn(row, previous):

                details = make_bar_details(
                    df,
                    position
                )

                if details is not None:
                    events.append(details)

                state = "NONE"

        previous = row

    if not events:
        return []

    latest_position = len(df) - 1

    recent_events = []

    for event in events:

        event_position = df.index.get_loc(
            event["date"]
        )

        days_since = latest_position - event_position

        if days_since >= sessions:
            continue

        event["days_since"] = days_since
        recent_events.append(event)

    recent_events.sort(
        key=lambda item: item["date"],
        reverse=True
    )

    return recent_events


def collect_recent_triggers(ticker_data, tickers):
    lifecycle_definitions = {
        "PULLBACK": (
            pullback_setup,
            pullback_trigger,
            pullback_invalidated
        ),
        "BASING": (
            basing_setup,
            basing_trigger,
            basing_invalidated
        ),
        "12/22": (
            crossover_setup,
            crossover_trigger,
            crossover_invalidated
        )
    }

    recent = []

    for ticker in tickers:

        df = ticker_data.get(ticker)

        if df is None:
            continue

        for strategy, definitions in lifecycle_definitions.items():

            events = collect_trigger_events(
                df,
                definitions[0],
                definitions[1],
                definitions[2]
            )

            for event in events:

                record = event.copy()
                record["ticker"] = ticker
                record["strategy"] = strategy

                recent.append(record)

    recent.sort(
        key=lambda item: item["date"],
        reverse=True
    )

    return recent


# ============================================================
# NORMAL OUTPUT
# ============================================================

def format_ticker(item):
    ticker = item["ticker"]
    age = item.get("age")

    if item["status"] == "ACTIVE" and age is not None:
        return f"{ticker}[{age}d]"

    return ticker


def print_lifecycle(title, states):
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)

    for status in ["NEW", "ACTIVE", "TRIGGERED", "INVALIDATED"]:

        items = states[status]

        print()
        print(f"{status}: {len(items)}")

        if items:
            print(
                ", ".join(
                    format_ticker(item)
                    for item in items
                )
            )


def print_today_triggers(results):
    print()
    print("=" * 65)
    print("TODAY'S TRIGGERS")
    print("=" * 65)

    any_trigger = False

    for strategy in ["PULLBACK", "BASING", "12/22"]:

        items = results[strategy]["TRIGGERED"]

        print()
        print(f"{strategy}: {len(items)}")

        if items:
            any_trigger = True
            print(
                ", ".join(
                    item["ticker"]
                    for item in items
                )
            )

    if not any_trigger:
        print()
        print("None")


def classify_gap(gap_pct):
    if pd.isna(gap_pct):
        return "UNKNOWN"

    if gap_pct > DIAGNOSTIC_FLAT_GAP_PCT:
        return "GAP UP"

    if gap_pct < -DIAGNOSTIC_FLAT_GAP_PCT:
        return "GAP DOWN"

    return "FLAT"


def print_recent_triggers(recent):
    print()
    print("=" * 65)
    print(
        f"RECENT TRIGGERS — "
        f"LAST {RECENT_TRIGGER_SESSIONS} TRADING DAYS"
    )
    print("=" * 65)

    if not recent:
        print()
        print("None")
        return

    for event in recent:

        age = event["days_since"]
        age_text = "today" if age == 0 else f"{age}d ago"

        print(
            f"{event['ticker']} "
            f"({event['strategy']}) "
            f"{event['date'].strftime('%Y-%m-%d')} "
            f"[{age_text}]"
        )


# ============================================================
# DEBUG OUTPUT
# ============================================================

def print_trigger_details(title, items):
    print()
    print("=" * 65)
    print(f"{title} — TODAY'S TRIGGER DETAILS")
    print("=" * 65)

    if not items:
        print("None")
        return

    for item in items:

        print()

        print(
            f"{item['ticker']}: "
            f"O={item['open']:.2f} "
            f"H={item['high']:.2f} "
            f"L={item['low']:.2f} "
            f"C={item['close']:.2f}"
        )

        print(
            f"  Previous close:   "
            f"{item['previous_close']:.2f}"
        )

        print(
            f"  Gap:              "
            f"{item['gap']:.2f}"
        )

        print(
            f"  Gap %:            "
            f"{item['gap_pct']:.2f}%"
        )

        print(
            f"  Gap / ATR14:      "
            f"{item['gap_atr']:.2f}"
        )

        print(
            f"  Close vs Open:    "
            f"{item['intraday_pct']:.2f}%"
        )

        print(
            f"  Close vs Prev:    "
            f"{item['net_day_pct']:.2f}%"
        )

        print(
            f"  Spread:           "
            f"{item['spread']:.2f}"
        )

        print(
            f"  Avg Spread 20:    "
            f"{item['avg_spread20']:.2f}"
        )

        print(
            f"  Spread Ratio 20:  "
            f"{item['spread_ratio20']:.2f}"
        )

        print(
            f"  Spread / ATR14:   "
            f"{item['spread_atr']:.2f}"
        )

        print(
            f"  50% midpoint:     "
            f"{item['midpoint']:.2f}"
        )

        print(
            f"  Close position:   "
            f"{item['close_pos'] * 100:.1f}%"
        )

        print(
            f"  RVOL:             "
            f"{item['rvol']:.2f}"
        )

        print(
            f"  TR / ATR14:       "
            f"{item['tr_atr']:.2f}"
        )

        print(
            f"  Candle:           "
            f"{'Bullish' if item['candle_bullish'] else 'Bearish'}"
        )


def print_recent_trigger_details(recent):
    print()
    print("=" * 65)
    print("RECENT TRIGGER DETAILS")
    print("=" * 65)

    if not recent:
        print("None")
        return

    for event in recent:

        print()
        print(
            f"{event['ticker']} "
            f"({event['strategy']}) "
            f"{event['date'].strftime('%Y-%m-%d')} "
            f"— {event['days_since']}d ago"
        )

        print(
            f"  O={event['open']:.2f} "
            f"H={event['high']:.2f} "
            f"L={event['low']:.2f} "
            f"C={event['close']:.2f}"
        )

        print(
            f"  Gap {event['gap_pct']:+.2f}% "
            f"({classify_gap(event['gap_pct'])}) | "
            f"Close vs Prev {event['net_day_pct']:+.2f}% | "
            f"RVOL {event['rvol']:.2f} | "
            f"Spr/20 {event['spread_ratio20']:.2f} | "
            f"ClosePos {event['close_pos']:.2f}"
        )


def print_candidate_calibration(results):
    print()
    print("=" * 65)
    print("CURRENT CANDIDATE CALIBRATION")
    print("=" * 65)

    for setup_name in ["PULLBACK", "BASING", "12/22"]:

        candidates = (
            results[setup_name]["NEW"]
            + results[setup_name]["ACTIVE"]
        )

        print()
        print(setup_name)

        if not candidates:
            print("No current candidates")
            continue

        gap_up = 0
        flat = 0
        gap_down = 0
        bullish_close = 0
        rvol_125 = 0
        rvol_150 = 0
        spread_100 = 0
        spread_125 = 0
        spread_150 = 0

        effort_result_125_100 = 0
        effort_result_150_100 = 0
        effort_result_125_125 = 0

        gap_up_rvol = 0
        gap_up_bullish = 0
        flat_rvol = 0
        flat_spread = 0
        flat_effort_result = 0
        gap_down_bullish = 0
        gap_down_effort_result = 0

        for item in candidates:

            gap_class = classify_gap(
                item["gap_pct"]
            )

            if gap_class == "GAP UP":
                gap_up += 1
            elif gap_class == "GAP DOWN":
                gap_down += 1
            else:
                flat += 1

            if item["close_pos"] > 0.50:
                bullish_close += 1

            if item["rvol"] >= 1.25:
                rvol_125 += 1

            if item["rvol"] >= 1.50:
                rvol_150 += 1

            if item["spread_ratio20"] >= 1.00:
                spread_100 += 1

            if item["spread_ratio20"] >= 1.25:
                spread_125 += 1

            if item["spread_ratio20"] >= 1.50:
                spread_150 += 1

            if (
                item["rvol"] >= 1.25
                and item["spread_ratio20"] >= 1.00
            ):
                effort_result_125_100 += 1

            if (
                item["rvol"] >= 1.50
                and item["spread_ratio20"] >= 1.00
            ):
                effort_result_150_100 += 1

            if (
                item["rvol"] >= 1.25
                and item["spread_ratio20"] >= 1.25
            ):
                effort_result_125_125 += 1

            if gap_class == "GAP UP":

                if item["rvol"] >= 1.00:
                    gap_up_rvol += 1

                if item["close_pos"] > 0.50:
                    gap_up_bullish += 1

            elif gap_class == "FLAT":

                if item["rvol"] >= 1.25:
                    flat_rvol += 1

                if item["spread_ratio20"] >= 1.00:
                    flat_spread += 1

                if (
                    item["rvol"] >= 1.25
                    and item["spread_ratio20"] >= 1.00
                    and item["close_pos"] > 0.50
                ):
                    flat_effort_result += 1

            elif gap_class == "GAP DOWN":

                if item["close_pos"] > 0.50:
                    gap_down_bullish += 1

                if (
                    item["rvol"] >= 1.25
                    and item["spread_ratio20"] >= 1.00
                ):
                    gap_down_effort_result += 1

        print(f"Current candidates: {len(candidates)}")
        print(f"Gap up:   {gap_up}")
        print(f"Flat:     {flat}")
        print(f"Gap down: {gap_down}")
        print(f"Close upper half: {bullish_close}")
        print(f"RVOL >= 1.25: {rvol_125}")
        print(f"RVOL >= 1.50: {rvol_150}")
        print(f"SpreadRatio20 >= 1.00: {spread_100}")
        print(f"SpreadRatio20 >= 1.25: {spread_125}")
        print(f"SpreadRatio20 >= 1.50: {spread_150}")
        print(
            f"RVOL >= 1.25 + SpreadRatio20 >= 1.00: "
            f"{effort_result_125_100}"
        )
        print(
            f"RVOL >= 1.50 + SpreadRatio20 >= 1.00: "
            f"{effort_result_150_100}"
        )
        print(
            f"RVOL >= 1.25 + SpreadRatio20 >= 1.25: "
            f"{effort_result_125_125}"
        )
        print(
            f"Gap-up + RVOL >= 1.00: {gap_up_rvol}"
        )
        print(
            f"Gap-up + close upper half: {gap_up_bullish}"
        )
        print(
            f"Flat + RVOL >= 1.25: {flat_rvol}"
        )
        print(
            f"Flat + SpreadRatio20 >= 1.00: {flat_spread}"
        )
        print(
            f"Flat + effort/result + bullish close: "
            f"{flat_effort_result}"
        )
        print(
            f"Gap-down + close upper half: {gap_down_bullish}"
        )
        print(
            f"Gap-down + effort/result: {gap_down_effort_result}"
        )


def print_full_universe_calibration(ticker_data, tickers):
    rvol_counts = {
        level: 0
        for level in RVOL_LEVELS
    }

    atr_counts = {
        level: 0
        for level in ATR_LEVELS
    }

    spread_counts = {
        level: 0
        for level in SPREAD_LEVELS
    }

    upside_close = 0
    upside_close_bullish_candle = 0
    upside_close_bullish_rvol = 0
    upside_close_bullish_rvol_atr = 0

    twelve22_rvol = 0
    twelve22_rvol_atr = 0
    twelve22_rvol_atr_candle = 0

    effort_result_125 = 0
    effort_result_150 = 0
    effort_result_200 = 0

    for ticker in tickers:

        df = ticker_data.get(ticker)

        if df is None:
            continue

        row = df.iloc[-1]

        required = [
            row["Close"],
            row["SMA12"],
            row["RVOL"],
            row["TR_ATR"],
            row["Spread_ATR"],
            row["SpreadRatio20"],
            row["ClosePos"]
        ]

        if any(pd.isna(value) for value in required):
            continue

        rvol = row["RVOL"]
        tr_atr = row["TR_ATR"]
        spread_atr = row["Spread_ATR"]
        spread_ratio20 = row["SpreadRatio20"]

        close_above_sma12 = (
            row["Close"] > row["SMA12"]
        )

        candle_bullish = (
            row["Close"]
            > ((row["High"] + row["Low"]) / 2)
        )

        for level in RVOL_LEVELS:
            if rvol >= level:
                rvol_counts[level] += 1

        for level in ATR_LEVELS:
            if tr_atr >= level:
                atr_counts[level] += 1

        for level in SPREAD_LEVELS:
            if spread_ratio20 >= level:
                spread_counts[level] += 1

        if close_above_sma12:

            upside_close += 1

            if candle_bullish:

                upside_close_bullish_candle += 1

                if rvol >= 1.50:

                    upside_close_bullish_rvol += 1

                    if tr_atr >= 1.00:
                        upside_close_bullish_rvol_atr += 1

        if rvol >= 1.50:

            twelve22_rvol += 1

            if tr_atr >= 1.00:

                twelve22_rvol_atr += 1

                if candle_bullish:
                    twelve22_rvol_atr_candle += 1

        if (
            rvol >= 1.25
            and spread_atr >= 1.00
        ):
            effort_result_125 += 1

        if (
            rvol >= 1.50
            and spread_atr >= 1.00
        ):
            effort_result_150 += 1

        if (
            rvol >= 2.00
            and spread_atr >= 1.00
        ):
            effort_result_200 += 1

    print()
    print("=" * 65)
    print("TRIGGER CALIBRATION — FULL UNIVERSE")
    print("=" * 65)

    print()
    print("RVOL THRESHOLDS")

    for level in RVOL_LEVELS:
        print(
            f"RVOL >= {level:.2f}: "
            f"{rvol_counts[level]}"
        )

    print()
    print("TRUE RANGE / ATR14 THRESHOLDS")

    for level in ATR_LEVELS:
        print(
            f"TR/ATR14 >= {level:.2f}: "
            f"{atr_counts[level]}"
        )

    print()
    print("SPREAD / PRIOR-20-DAY AVERAGE SPREAD")

    for level in SPREAD_LEVELS:
        print(
            f"SpreadRatio20 >= {level:.2f}: "
            f"{spread_counts[level]}"
        )

    print()
    print("UPSIDE CONFIRMATION")

    print(
        f"Close > SMA12: {upside_close}"
    )

    print(
        f"+ close in upper half: "
        f"{upside_close_bullish_candle}"
    )

    print(
        f"+ RVOL >= 1.50: "
        f"{upside_close_bullish_rvol}"
    )

    print(
        f"+ TR/ATR14 >= 1.00: "
        f"{upside_close_bullish_rvol_atr}"
    )

    print()
    print("12/22 VOLUME / RANGE CONFIRMATION")

    print(
        f"RVOL >= 1.50: "
        f"{twelve22_rvol}"
    )

    print(
        f"+ TR/ATR14 >= 1.00: "
        f"{twelve22_rvol_atr}"
    )

    print(
        f"+ close upper half: "
        f"{twelve22_rvol_atr_candle}"
    )

    print()
    print("WYCKOFF EFFORT / RESULT")

    print("Effort = RVOL")
    print("Result = Spread / ATR14")

    print(
        f"RVOL >= 1.25 + Spread/ATR >= 1.00: "
        f"{effort_result_125}"
    )

    print(
        f"RVOL >= 1.50 + Spread/ATR >= 1.00: "
        f"{effort_result_150}"
    )

    print(
        f"RVOL >= 2.00 + Spread/ATR >= 1.00: "
        f"{effort_result_200}"
    )


# ============================================================
# CHART GENERATION
# ============================================================

TRIGGERS_URL = "https://paulcwarren.github.io/pinescript-libs/scanners/triggers.html"
TRIGGERS_FILE = Path(__file__).resolve().parent / "triggers.html"


def build_trigger_chart(
    ticker,
    strategy,
    trigger_event,
    df
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    trigger_date = pd.Timestamp(trigger_event["date"])

    # Show the most recent chart window so historical triggers remain
    # visible in context. The trigger marker is placed on the actual
    # trigger candle rather than at the right edge of the chart.
    chart_df = df.tail(CHART_LOOKBACK_BARS).copy()

    if chart_df.empty:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.72, 0.28],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}]
        ]
    )

    # --------------------------------------------------------
    # Price / candles
    # --------------------------------------------------------

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Price"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["SMA12"],
            mode="lines",
            name="SMA 12",
            line={"color": "green"}
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["SMA22"],
            mode="lines",
            name="SMA 22",
            line={"color": "orange"}
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["SMA55"],
            mode="lines",
            name="SMA 55",
            line={"color": "red"}
        ),
        row=1,
        col=1
    )

    # Prior-day close at the trigger.
    fig.add_hline(
        y=trigger_event["previous_close"],
        line_dash="dash",
        annotation_text=(
            f"Prior close {trigger_event['previous_close']:.2f}"
        ),
        annotation_position="top left",
        row=1,
        col=1
    )

    # Mark the actual trigger candle with an arrow below its low.
    if trigger_date in chart_df.index:
        trigger_low = float(chart_df.loc[trigger_date, "Low"])
        trigger_atr = float(chart_df.loc[trigger_date, "ATR14"])
        if pd.isna(trigger_atr) or trigger_atr <= 0:
            trigger_atr = max(trigger_low * 0.01, 0.01)
        arrow_y = trigger_low - (trigger_atr * 0.25)

        fig.add_annotation(
            x=trigger_date,
            y=trigger_low,
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            ax=0,
            ay=30,
            xanchor="center",
            yanchor="bottom",
            row=1,
            col=1
        )

    # --------------------------------------------------------
    # Volume / RVOL
    # --------------------------------------------------------

    # Match volume bar colors to candle direction.
    volume_colors = [
        "rgba(0,128,0,0.60)" if close >= open_price
        else "rgba(200,0,0,0.60)"
        for open_price, close in zip(
            chart_df["Open"],
            chart_df["Close"]
        )
    ]

    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            name="Volume",
            marker={"color": volume_colors},
            opacity=0.85
        ),
        row=2,
        col=1,
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["AvgVolume20"],
            mode="lines",
            name="Avg Volume 20",
            line={"color": "grey"}
        ),
        row=2,
        col=1,
        secondary_y=False
    )

    # RVOL is plotted against the right-hand secondary axis.
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["RVOL"],
            mode="lines",
            name="RVOL",
            line={
                "color": "gold",
                "width": 2,
                "dash": "dot"
            }
        ),
        row=2,
        col=1,
        secondary_y=True
    )

    # RVOL = 1.0 reference line.
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=[1.0] * len(chart_df),
            mode="lines",
            name="RVOL = 1.0",
            line={
                "color": "lightgrey",
                "dash": "dash"
            },
            hoverinfo="skip"
        ),
        row=2,
        col=1,
        secondary_y=True
    )

    # --------------------------------------------------------
    # Axes / layout
    # --------------------------------------------------------

    title = (
        f"{ticker} — {strategy} — "
        f"Trigger {trigger_date.strftime('%Y-%m-%d')}"
    )

    fig.update_layout(
        title=title,
        height=850,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
        margin={"l": 60, "r": 60, "t": 80, "b": 50}
    )

    fig.update_yaxes(
        title_text="Price",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Volume",
        row=2,
        col=1,
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="RVOL",
        row=2,
        col=1,
        secondary_y=True
    )

    # Remove Saturday/Sunday gaps.
    weekend_break = [
        dict(bounds=["sat", "mon"])
    ]

    fig.update_xaxes(
        rangebreaks=weekend_break,
        rangeslider_visible=False,
        row=1,
        col=1
    )

    fig.update_xaxes(
        rangebreaks=weekend_break,
        rangeslider_visible=False,
        row=2,
        col=1
    )

    return fig


def order_chart_events(recent_triggers, latest_date):
    """
    Order charts as:

        Today's Pullbacks
        Today's Basings
        Today's 12/22s
        Historical Pullbacks
        Historical Basings
        Historical 12/22s

    Within each strategy, newest trigger first.
    """

    strategy_order = {
        "PULLBACK": 0,
        "BASING": 1,
        "12/22": 2
    }

    ordered = []

    for strategy in ["PULLBACK", "BASING", "12/22"]:

        current = [
            event
            for event in recent_triggers
            if event["strategy"] == strategy
            and event["date"] == latest_date
        ]

        historical = [
            event
            for event in recent_triggers
            if event["strategy"] == strategy
            and event["date"] != latest_date
        ]

        current.sort(
            key=lambda item: item["ticker"]
        )

        historical.sort(
            key=lambda item: (
                item["date"],
                item["ticker"]
            ),
            reverse=True
        )

        ordered.extend(current)
        ordered.extend(historical)

    return ordered


def write_triggers_html(
    recent_triggers,
    ticker_data,
    latest_date
):
    """
    Generate one combined triggers.html page.

    The file is deliberately written directly beside this script
    so it becomes stocks/triggers.html in the GitHub Pages site.
    """

    try:
        import plotly.io as pio
    except ImportError:
        print()
        print(
            "Plotly is not installed. "
            "Skipping HTML chart generation."
        )
        print(
            "Install it with: pip install plotly"
        )
        return False

    ordered = order_chart_events(
        recent_triggers,
        latest_date
    )

    sections = []

    for index, event in enumerate(ordered):

        ticker = event["ticker"]
        strategy = event["strategy"]
        df = ticker_data.get(ticker)

        if df is None:
            continue

        fig = build_trigger_chart(
            ticker=ticker,
            strategy=strategy,
            trigger_event=event,
            df=df
        )

        if fig is None:
            continue

        age = event["days_since"]
        age_text = "TODAY" if age == 0 else f"{age} trading day(s) ago"

        chart_html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn"
        )

        section = (
            f"<section class='chart-section' data-index='{index}'>"
            f"<div class='chart-heading'>"
            f"<strong>{html.escape(ticker)} — "
            f"{html.escape(strategy)}</strong>"
            f"<span>Trigger {event['date'].strftime('%Y-%m-%d')} "
            f"({age_text})</span>"
            f"</div>"
            f"{chart_html}"
            f"</section>"
        )

        sections.append(section)

    chart_count = len(sections)

    if chart_count == 0:
        chart_body = (
            "<div class='empty'>"
            "No triggers in the current five-trading-day window."
            "</div>"
        )
    else:
        chart_body = "".join(sections)

    # --------------------------------------------------------
    # Keyboard paging.
    # ArrowUp    = previous chart
    # ArrowDown  = next chart
    # --------------------------------------------------------

    paging_script = f"""
    <script>
    (function() {{
        const charts = Array.from(
            document.querySelectorAll('.chart-section')
        );
        const counter = document.getElementById('chart-counter');
        const status = document.getElementById('chart-status');
        const total = charts.length;
        let current = 0;

        function showChart(index, behavior) {{
            if (!total) return;

            current = Math.max(0, Math.min(index, total - 1));

            charts[current].scrollIntoView({{
                behavior: behavior || 'smooth',
                block: 'start'
            }});

            if (counter) {{
                counter.textContent =
                    `Chart ${{current + 1}} of ${{total}}`;
            }}

            if (status) {{
                status.textContent =
                    charts[current].querySelector('.chart-heading strong')?.textContent || '';
            }}
        }}

        document.addEventListener('keydown', function(event) {{
            if (event.key === 'ArrowDown') {{
                event.preventDefault();
                showChart(current + 1);
            }} else if (event.key === 'ArrowUp') {{
                event.preventDefault();
                showChart(current - 1);
            }}
        }});

        document.getElementById('prev-chart')?.addEventListener('click', function() {{
            showChart(current - 1);
        }});

        document.getElementById('next-chart')?.addEventListener('click', function() {{
            showChart(current + 1);
        }});

        if (total) {{
            showChart(0, 'auto');
        }} else if (counter) {{
            counter.textContent = 'No charts';
        }}
    }})();
    </script>
    """

    # --------------------------------------------------------
    # Combined HTML page
    # --------------------------------------------------------

    dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Scanner Triggers</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 0;
    background: #f5f5f5;
    color: #222;
}}

#topbar {{
    position: sticky;
    top: 0;
    z-index: 1000;
    background: white;
    border-bottom: 1px solid #ccc;
    padding: 10px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}}

#topbar-inner {{
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}}

button {{
    padding: 6px 12px;
    border: 1px solid #aaa;
    border-radius: 4px;
    background: white;
    cursor: pointer;
}}

button:hover {{
    background: #eee;
}}

#chart-counter {{
    font-weight: bold;
}}

#chart-status {{
    color: #555;
}}

.hint {{
    color: #777;
    font-size: 13px;
}}

.chart-section {{
    background: white;
    margin: 20px auto;
    padding: 12px;
    max-width: 1500px;
    border: 1px solid #ddd;
    border-radius: 6px;
    scroll-margin-top: 90px;
}}

.chart-heading {{
    display: flex;
    justify-content: space-between;
    gap: 15px;
    padding: 4px 8px 10px;
    font-size: 15px;
}}

.chart-heading span {{
    color: #666;
}}

.empty {{
    max-width: 900px;
    margin: 60px auto;
    background: white;
    padding: 30px;
    text-align: center;
    border: 1px solid #ddd;
    border-radius: 6px;
}}
</style>
</head>
<body>
<div id="topbar">
    <div id="topbar-inner">
        <strong>Scanner Triggers</strong>
        <button id="prev-chart">↑ Previous</button>
        <button id="next-chart">↓ Next</button>
        <span id="chart-counter"></span>
        <span id="chart-status"></span>
        <span class="hint">Use ↑ / ↓ to page through charts</span>
    </div>
</div>

<div style="max-width:1500px;margin:12px auto;padding:0 12px;color:#666;">
    Latest market data: {latest_date.strftime('%Y-%m-%d')} &nbsp;|&nbsp;
    Monitoring window: last {RECENT_TRIGGER_SESSIONS} trading sessions
</div>

{chart_body}

{paging_script}
</body>
</html>
"""

    TRIGGERS_FILE.write_text(
        dashboard_html,
        encoding="utf-8"
    )

    return True


# ============================================================
# DEBUG DIAGNOSTICS
# ============================================================

def print_debug_sections(
    results,
    recent_triggers,
    ticker_data,
    tickers
):
    for strategy in ["PULLBACK", "BASING", "12/22"]:

        print_trigger_details(
            strategy,
            results[strategy]["TRIGGERED"]
        )

    print_recent_trigger_details(
        recent_triggers
    )

    print_candidate_calibration(
        results
    )

    print_full_universe_calibration(
        ticker_data,
        tickers
    )


# ============================================================
# SINGLE-TICKER DIAGNOSTIC
# ============================================================

def print_diagnostic(ticker, data):

    df = get_ticker_data(
        data,
        ticker
    )

    if df is None:
        print(
            f"\n{ticker}: insufficient or unavailable data"
        )
        return

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    print()
    print("=" * 70)
    print(f"DIAGNOSTIC: {ticker}")
    print("=" * 70)

    history = df.tail(5)[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "SMA12",
            "SMA22",
            "SMA55",
            "SMA200",
            "RVOL",
            "TR_ATR",
            "Spread_ATR",
            "SpreadRatio20",
            "ClosePos"
        ]
    ].copy()

    print()
    print("LAST 5 TRADING DAYS")
    print()

    print(
        f"{'Date':<12}"
        f"{'Open':>9}"
        f"{'High':>9}"
        f"{'Low':>9}"
        f"{'Close':>10}"
        f"{'SMA12':>10}"
        f"{'SMA22':>10}"
        f"{'RVOL':>8}"
        f"{'TR/ATR':>9}"
        f"{'Spr/ATR':>9}"
        f"{'Spr/20':>9}"
        f"{'ClosePos':>10}"
    )

    for date, row in history.iterrows():

        print(
            f"{date.strftime('%Y-%m-%d'):<12}"
            f"{row['Open']:>9.2f}"
            f"{row['High']:>9.2f}"
            f"{row['Low']:>9.2f}"
            f"{row['Close']:>10.2f}"
            f"{row['SMA12']:>10.2f}"
            f"{row['SMA22']:>10.2f}"
            f"{row['RVOL']:>8.2f}"
            f"{row['TR_ATR']:>9.2f}"
            f"{row['Spread_ATR']:>9.2f}"
            f"{row['SpreadRatio20']:>9.2f}"
            f"{row['ClosePos']:>10.2f}"
        )

    print()
    print(
        f"Latest date:      "
        f"{df.index[-1].strftime('%Y-%m-%d')}"
    )

    print(f"Open:             {latest['Open']:.2f}")
    print(f"High:             {latest['High']:.2f}")
    print(f"Low:              {latest['Low']:.2f}")
    print(f"Close:            {latest['Close']:.2f}")
    print(f"Previous Close:   {previous['Close']:.2f}")

    print(
        f"Gap:              "
        f"{latest['Gap']:.2f}"
    )

    print(
        f"Gap %:            "
        f"{latest['GapPct']:.2f}%"
    )

    print(
        f"Gap / ATR14:      "
        f"{latest['GapATR']:.2f}"
    )

    print(
        f"Close vs Open:    "
        f"{latest['IntradayPct']:.2f}%"
    )

    print(
        f"Close vs Prev:    "
        f"{latest['NetDayPct']:.2f}%"
    )

    print(
        f"Spread:           "
        f"{latest['Spread']:.2f}"
    )

    print(
        f"Avg Spread 20:    "
        f"{latest['AvgSpread20']:.2f}"
    )

    print(
        f"Spread Ratio 20:  "
        f"{latest['SpreadRatio20']:.2f}"
    )

    print(
        f"Spread / ATR14:   "
        f"{latest['Spread_ATR']:.2f}"
    )

    print(
        f"50% midpoint:     "
        f"{((latest['High'] + latest['Low']) / 2):.2f}"
    )

    print(f"SMA 12:           {latest['SMA12']:.2f}")
    print(f"SMA 22:           {latest['SMA22']:.2f}")
    print(f"SMA 55:           {latest['SMA55']:.2f}")
    print(f"SMA 200:          {latest['SMA200']:.2f}")
    print(f"3M Perf:          {latest['Perf3M']:.2f}%")
    print(f"Volume:           {latest['Volume']:,.0f}")
    print(f"Avg Volume 20:    {latest['AvgVolume20']:,.0f}")
    print(f"RVOL:             {latest['RVOL']:.2f}")

    print()
    print(f"ATR 14:           {latest['ATR14']:.2f}")
    print(f"True Range:       {latest['TR']:.2f}")
    print(f"TR / ATR14:       {latest['TR_ATR']:.2f}")
    print(f"Close Position:   {latest['ClosePos']:.2f}")
    print(f"Body / Range:     {latest['BodyPct']:.2f}")
    print(f"Upper Wick/Body:  {latest['UpperWickBody']:.2f}")

    print(
        f"OC Direction:     "
        f"{'Bullish' if latest['OCBullish'] else 'Bearish'}"
    )

    print(
        f"Your Candle Bias: "
        f"{'Bullish' if latest['CandleBullish'] else 'Bearish'}"
    )

    print()
    print("WYCKOFF EFFORT / RESULT")

    print(
        f"Effort (RVOL):     "
        f"{latest['RVOL']:.2f}"
    )

    print(
        f"Result (Spr/ATR):  "
        f"{latest['Spread_ATR']:.2f}"
    )

    print(
        f"Result (Spr/20):   "
        f"{latest['SpreadRatio20']:.2f}"
    )

    print(
        f"Effort >= 1.25:    "
        f"{latest['RVOL'] >= 1.25}"
    )

    print(
        f"Spread/ATR >=1:    "
        f"{latest['Spread_ATR'] >= 1.00}"
    )

    print(
        f"Spread/20 >=1.25:  "
        f"{latest['SpreadRatio20'] >= 1.25}"
    )

    print(
        f"Effort + Result:   "
        f"{latest['RVOL'] >= 1.25 and latest['Spread_ATR'] >= 1.00}"
    )

    print()
    print("TRIGGER GAP RULE")

    print(
        f"Open < previous close: "
        f"{latest['Open'] < previous['Close']}"
    )

    print(
        f"Close < previous close: "
        f"{latest['Close'] < previous['Close']}"
    )

    print(
        f"Gap-down unrecovered:   "
        f"{gap_down_unrecovered(latest, previous)}"
    )

    print(
        f"Trigger gap rule passes: "
        f"{trigger_gap_rule(latest, previous)}"
    )

    print()
    print("SETUP / TRIGGER STATUS")

    pullback = pullback_setup(
        latest,
        previous
    )

    basing = basing_setup(
        latest,
        previous
    )

    crossover = crossover_setup(
        latest,
        previous
    )

    print(f"Pullback setup:  {pullback}")
    print(
        f"Pullback trigger: "
        f"{pullback_trigger(latest, previous)}"
    )

    print(f"Basing setup:    {basing}")
    print(
        f"Basing trigger:  "
        f"{basing_trigger(latest, previous)}"
    )

    print(f"12/22 setup:      {crossover}")
    print(
        f"12/22 trigger:   "
        f"{crossover_trigger(latest, previous)}"
    )


# ============================================================
# MAIN SCAN
# ============================================================

def run_full_scan(data, tickers, debug=False):

    results, ticker_data, data_issues = build_scan_results(
        data,
        tickers
    )

    latest_date = get_common_latest_date(
        ticker_data
    )

    # --------------------------------------------------------
    # Data freshness
    # --------------------------------------------------------

    print()

    if latest_date is not None:
        print(
            f"Latest market data: "
            f"{latest_date.strftime('%Y-%m-%d')}"
        )

    # --------------------------------------------------------
    # Candidate lifecycle
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("CANDIDATE LIFECYCLE")
    print("=" * 65)

    print_lifecycle(
        "PULLBACK",
        results["PULLBACK"]
    )

    print_lifecycle(
        "BASING",
        results["BASING"]
    )

    print_lifecycle(
        "12/22 CROSSOVER",
        results["12/22"]
    )

    # --------------------------------------------------------
    # Recent triggers
    # --------------------------------------------------------

    recent_triggers = collect_recent_triggers(
        ticker_data,
        tickers
    )

    print_today_triggers(
        results
    )

    print_recent_triggers(
        recent_triggers
    )

    # --------------------------------------------------------
    # Combined HTML trigger page
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("TRIGGER CHARTS")
    print("=" * 65)

    if latest_date is not None:
        chart_written = write_triggers_html(
            recent_triggers,
            ticker_data,
            latest_date
        )

        if chart_written:
            print(f"Updated {TRIGGERS_FILE}")
            print(f"Charts: {TRIGGERS_URL}")
        else:
            print("Chart generation skipped.")
    else:
        print("No market data available; chart generation skipped.")

    # --------------------------------------------------------
    # Debug-only detail
    # --------------------------------------------------------

    if debug:

        print_debug_sections(
            results,
            recent_triggers,
            ticker_data,
            tickers
        )

    # --------------------------------------------------------
    # Data issues
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("DATA ISSUES")
    print("=" * 65)

    print(f"{len(data_issues)} stocks")

    if data_issues:
        print(", ".join(data_issues))


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Daily stock setup scanner"
    )

    parser.add_argument(
        "--diagnostic",
        nargs="+",
        metavar="TICKER",
        help="Detailed diagnostic for one or more tickers"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed trigger/calibration diagnostics"
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        help=(
            "Load previously downloaded stock data from this pickle file "
            "instead of contacting Yahoo Finance."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Explicit ticker diagnostic mode.
    # Only requested tickers are downloaded.
    # --------------------------------------------------------

    if args.diagnostic:

        tickers = [
            ticker.upper()
            for ticker in args.diagnostic
        ]

        if args.data_file:
            data = load_data_file(args.data_file)
        else:
            data = download_data(tickers)
            save_data_file(data, DEFAULT_DATA_FILE)

        for ticker in tickers:
            print_diagnostic(
                ticker,
                data
            )

        return

    # --------------------------------------------------------
    # Normal full-universe scan.
    # --------------------------------------------------------

    tickers = load_tickers()

    if args.data_file:
        data = load_data_file(args.data_file)
    else:
        data = download_data(tickers)
        save_data_file(data, DEFAULT_DATA_FILE)

    run_full_scan(
        data,
        tickers,
        debug=args.debug
    )


if __name__ == "__main__":
    main()