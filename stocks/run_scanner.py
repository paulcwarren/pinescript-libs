import pandas as pd
import yfinance as yf
import argparse
from pathlib import Path


INPUT_FILE = Path(__file__).parent.parent / "min10B_listed.csv"

# ------------------------------------------------------------
# Diagnostic thresholds.
#
# These are NOT currently part of the trigger logic.
# ------------------------------------------------------------

RVOL_LEVELS = [1.00, 1.25, 1.50, 2.00]
ATR_LEVELS = [0.75, 1.00, 1.25, 1.50]
SPREAD_LEVELS = [0.75, 1.00, 1.25, 1.50]

# Diagnostic only:
# abs(gap) <= this value is treated as "flat/no meaningful gap".
DIAGNOSTIC_FLAT_GAP_PCT = 0.50

# Remember trigger events for this many trading sessions,
# including the trigger session itself.
RECENT_TRIGGER_SESSIONS = 5


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
    # Today's spread does not influence its own baseline.
    df["AvgSpread20"] = (
        df["Spread"]
        .rolling(20)
        .mean()
        .shift(1)
    )

    # Today's spread relative to its prior 20-session average.
    #
    # 1.00 = normal
    # 1.25 = 25% wider than normal
    # 1.50 = 50% wider than normal
    # 2.00 = twice normal
    df["SpreadRatio20"] = (
        df["Spread"] / df["AvgSpread20"]
    )

    # Spread relative to ATR14.
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

    # Position of close within today's range.
    #
    # 0.00 = close at low
    # 0.50 = close at midpoint
    # 1.00 = close at high
    df["ClosePos"] = (
        (df["Close"] - df["Low"]) / df["DayRange"]
    ).where(df["DayRange"] > 0)

    # User's candle interpretation:
    # bullish when close finishes above midpoint of range.
    df["CandleBullish"] = (
        df["Close"] > ((df["High"] + df["Low"]) / 2)
    )

    # Conventional open/close direction retained separately.
    df["OCBullish"] = df["Close"] > df["Open"]

    # Body as percentage of total range.
    df["BodyPct"] = (
        df["Body"] / df["DayRange"]
    ).where(df["DayRange"] > 0)

    # Upper wick relative to body.
    df["UpperWickBody"] = (
        df["UpperWick"] / df["Body"]
    ).where(df["Body"] > 0)

    return df


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
# Current rules remain unchanged.
#
# Pullback/Basing:
#     Close > SMA12
#     RVOL > 1
#     Close > 50% of day's range
#
# 12/22:
#     RVOL > 1
#     Close > 50% of day's range
#
# Gap, ATR and Wyckoff effort/result remain diagnostic only.
# ============================================================

def pullback_trigger(row, previous=None):
    return (
        row["Close"] > row["SMA12"]
        and row["RVOL"] > 1
        and row["Close"] > ((row["High"] + row["Low"]) / 2)
    )


def basing_trigger(row, previous=None):
    return (
        row["Close"] > row["SMA12"]
        and row["RVOL"] > 1
        and row["Close"] > ((row["High"] + row["Low"]) / 2)
    )


def crossover_trigger(row, previous=None):
    return (
        row["RVOL"] > 1
        and row["Close"] > ((row["High"] + row["Low"]) / 2)
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
        # Candidate currently active
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
        # On the next bar, reset unless a new setup occurs.
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

    # NEW only means the setup occurred today.
    if state == "NEW" and event_date != latest_date:
        state = "ACTIVE"

    # Triggered / invalidated only count as current if the event
    # happened on the latest trading bar.
    if state in ("TRIGGERED", "INVALIDATED") and event_date != latest_date:
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
# RECENT TRIGGER HISTORY
#
# Reconstruct every valid trigger event and retain only the
# most recent 5 trading sessions.
#
# No persistent state file is required.
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

        if state == "NONE":

            if setup_fn(row, previous):

                if trigger_fn(row, previous):
                    events.append(date)
                    state = "NONE"
                else:
                    state = "ACTIVE"

        elif state == "ACTIVE":

            if invalid_fn(row, previous):
                state = "NONE"

            elif trigger_fn(row, previous):
                events.append(date)
                state = "NONE"

        previous = row

    if not events:
        return []

    latest_date = df.index[-1]

    try:
        latest_position = df.index.get_loc(latest_date)
    except KeyError:
        return []

    recent_events = []

    for event_date in events:

        try:
            event_position = df.index.get_loc(event_date)
        except KeyError:
            continue

        days_since = latest_position - event_position

        if days_since >= sessions:
            continue

        row = df.loc[event_date]
        previous_date = df.index[event_position - 1]

        if event_position == 0:
            continue

        previous_row = df.loc[previous_date]

        recent_events.append(
            {
                "date": event_date,
                "days_since": days_since,
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "previous_close": previous_row["Close"],
                "gap": row["Gap"],
                "gap_pct": row["GapPct"],
                "gap_atr": row["GapATR"],
                "intraday_pct": row["IntradayPct"],
                "net_day_pct": row["NetDayPct"],
                "spread": row["Spread"],
                "avg_spread20": row["AvgSpread20"],
                "spread_ratio20": row["SpreadRatio20"],
                "spread_atr": row["Spread_ATR"],
                "close_pos": row["ClosePos"],
                "rvol": row["RVOL"],
                "tr_atr": row["TR_ATR"],
                "candle_bullish": row["CandleBullish"]
            }
        )

    # Most recent trigger first.
    recent_events.sort(
        key=lambda item: item["date"],
        reverse=True
    )

    return recent_events


# ============================================================
# REPORTING
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
            print(", ".join(
                format_ticker(item)
                for item in items
            ))


def print_trigger_details(title, items):
    print()
    print("=" * 65)
    print(f"{title} — TODAY'S TRIGGERS")
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


def print_recent_trigger_details(
    title,
    ticker,
    events
):
    print()
    print(f"{ticker}")

    for event in events:

        print(
            f"  Trigger:          "
            f"{event['date'].strftime('%Y-%m-%d')} "
            f"({event['days_since']}d ago)"
        )

        print(
            f"  O={event['open']:.2f} "
            f"H={event['high']:.2f} "
            f"L={event['low']:.2f} "
            f"C={event['close']:.2f}"
        )

        print(
            f"  Gap:              "
            f"{event['gap_pct']:.2f}% "
            f"({event['gap_atr']:.2f} ATR)"
        )

        print(
            f"  Close vs Prev:    "
            f"{event['net_day_pct']:.2f}%"
        )

        print(
            f"  Close position:   "
            f"{event['close_pos'] * 100:.1f}%"
        )

        print(
            f"  RVOL:             "
            f"{event['rvol']:.2f}"
        )

        print(
            f"  Spread Ratio 20:  "
            f"{event['spread_ratio20']:.2f}"
        )

        print(
            f"  Spread / ATR14:   "
            f"{event['spread_atr']:.2f}"
        )


# ============================================================
# CURRENT CANDIDATE CALIBRATION
#
# Uses current NEW + ACTIVE candidates only.
# Triggered and invalidated events are excluded.
# ============================================================

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

        print(
            f"Current candidates: {len(candidates)}"
        )

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

            gap_pct = item["gap_pct"]

            if gap_pct > DIAGNOSTIC_FLAT_GAP_PCT:
                gap_class = "GAP UP"
            elif gap_pct < -DIAGNOSTIC_FLAT_GAP_PCT:
                gap_class = "GAP DOWN"
            else:
                gap_class = "FLAT"

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

        print(
            f"Gap up   (>{DIAGNOSTIC_FLAT_GAP_PCT:.2f}%): "
            f"{gap_up}"
        )

        print(
            f"Flat     (±{DIAGNOSTIC_FLAT_GAP_PCT:.2f}%): "
            f"{flat}"
        )

        print(
            f"Gap down (<-{DIAGNOSTIC_FLAT_GAP_PCT:.2f}%): "
            f"{gap_down}"
        )

        print()
        print(
            f"Close in upper half:           "
            f"{bullish_close}"
        )

        print(
            f"RVOL >= 1.25:                  "
            f"{rvol_125}"
        )

        print(
            f"RVOL >= 1.50:                  "
            f"{rvol_150}"
        )

        print(
            f"SpreadRatio20 >= 1.00:         "
            f"{spread_100}"
        )

        print(
            f"SpreadRatio20 >= 1.25:         "
            f"{spread_125}"
        )

        print(
            f"SpreadRatio20 >= 1.50:         "
            f"{spread_150}"
        )

        print()
        print(
            f"RVOL >= 1.25 + "
            f"SpreadRatio20 >= 1.00:        "
            f"{effort_result_125_100}"
        )

        print(
            f"RVOL >= 1.50 + "
            f"SpreadRatio20 >= 1.00:        "
            f"{effort_result_150_100}"
        )

        print(
            f"RVOL >= 1.25 + "
            f"SpreadRatio20 >= 1.25:        "
            f"{effort_result_125_125}"
        )

        print()
        print("GAP-SPECIFIC OBSERVATIONS")

        print(
            f"Gap-up + RVOL >= 1.00:         "
            f"{gap_up_rvol}"
        )

        print(
            f"Gap-up + close upper half:     "
            f"{gap_up_bullish}"
        )

        print(
            f"Flat + RVOL >= 1.25:           "
            f"{flat_rvol}"
        )

        print(
            f"Flat + SpreadRatio20 >= 1.00: "
            f"{flat_spread}"
        )

        print(
            f"Flat + effort/result + bullish close:"
            f"{flat_effort_result:>12}"
        )

        print(
            f"Gap-down + close upper half:   "
            f"{gap_down_bullish}"
        )

        print(
            f"Gap-down + effort/result:      "
            f"{gap_down_effort_result}"
        )


# ============================================================
# FULL-UNIVERSE CALIBRATION
# ============================================================

def print_trigger_calibration(data, tickers):
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

        df = get_ticker_data(data, ticker)

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
            f"RVOL >= {level:.2f}:"
            f"{rvol_counts[level]:>5}"
        )

    print()
    print("TRUE RANGE / ATR14 THRESHOLDS")

    for level in ATR_LEVELS:
        print(
            f"TR/ATR14 >= {level:.2f}:"
            f"{atr_counts[level]:>5}"
        )

    print()
    print("SPREAD / PRIOR-20-DAY AVERAGE SPREAD")

    for level in SPREAD_LEVELS:
        print(
            f"SpreadRatio20 >= {level:.2f}:"
            f"{spread_counts[level]:>5}"
        )

    print()
    print("UPSIDE CONFIRMATION")

    print(
        f"Close > SMA12:"
        f"{upside_close:>27}"
    )

    print(
        f"+ close in upper half of range:"
        f"{upside_close_bullish_candle:>10}"
    )

    print(
        f"+ RVOL >= 1.50:"
        f"{upside_close_bullish_rvol:>22}"
    )

    print(
        f"+ TR/ATR14 >= 1.00:"
        f"{upside_close_bullish_rvol_atr:>17}"
    )

    print()
    print("12/22 VOLUME / RANGE CONFIRMATION")

    print(
        f"RVOL >= 1.50:"
        f"{twelve22_rvol:>28}"
    )

    print(
        f"+ TR/ATR14 >= 1.00:"
        f"{twelve22_rvol_atr:>17}"
    )

    print(
        f"+ close in upper half of range:"
        f"{twelve22_rvol_atr_candle:>10}"
    )

    print()
    print("WYCKOFF EFFORT / RESULT")

    print("Effort = RVOL")
    print("Result = Spread / ATR14")

    print(
        f"RVOL >= 1.25 and Spread/ATR >= 1.00:"
        f"{effort_result_125:>8}"
    )

    print(
        f"RVOL >= 1.50 and Spread/ATR >= 1.00:"
        f"{effort_result_150:>8}"
    )

    print(
        f"RVOL >= 2.00 and Spread/ATR >= 1.00:"
        f"{effort_result_200:>8}"
    )


# ============================================================
# FULL UNIVERSE SCAN
# ============================================================

def run_scan(data, tickers):

    results, ticker_data, data_issues = build_scan_results(
        data,
        tickers
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
    # Today's exact trigger details
    # --------------------------------------------------------

    print_trigger_details(
        "PULLBACK",
        results["PULLBACK"]["TRIGGERED"]
    )

    print_trigger_details(
        "BASING",
        results["BASING"]["TRIGGERED"]
    )

    print_trigger_details(
        "12/22",
        results["12/22"]["TRIGGERED"]
    )

    # --------------------------------------------------------
    # Recent trigger history
    # --------------------------------------------------------

    print_recent_triggers(
        ticker_data,
        tickers
    )

    # --------------------------------------------------------
    # Candidate-specific calibration
    # --------------------------------------------------------

    print_candidate_calibration(
        results
    )

    # --------------------------------------------------------
    # Full-universe calibration
    # --------------------------------------------------------

    print_trigger_calibration(
        data,
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
# BUILD CURRENT LIFECYCLE RESULTS
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

    ticker_data = {}

    for ticker in tickers:

        df = get_ticker_data(data, ticker)

        if df is None:
            data_issues.append(ticker)
            continue

        ticker_data[ticker] = df

        for setup_name, definitions in lifecycle_definitions.items():

            setup_fn = definitions[0]
            trigger_fn = definitions[1]
            invalid_fn = definitions[2]

            state = replay_lifecycle(
                df,
                setup_fn,
                trigger_fn,
                invalid_fn
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
                    latest["High"]
                    + latest["Low"]
                ) / 2,

                "close_pos": latest["ClosePos"],
                "rvol": latest["RVOL"],
                "tr_atr": latest["TR_ATR"],

                "candle_bullish": (
                    latest["Close"]
                    > (
                        (
                            latest["High"]
                            + latest["Low"]
                        ) / 2
                    )
                )
            }

            results[setup_name][state["status"]].append(
                result
            )

    return results, ticker_data, data_issues


# ============================================================
# GAP CLASSIFICATION
# ============================================================

def classify_gap(gap_pct):
    if pd.isna(gap_pct):
        return "UNKNOWN"

    if gap_pct > DIAGNOSTIC_FLAT_GAP_PCT:
        return "GAP UP"

    if gap_pct < -DIAGNOSTIC_FLAT_GAP_PCT:
        return "GAP DOWN"

    return "FLAT"


# ============================================================
# RECENT TRIGGERS REPORT
# ============================================================

def print_recent_triggers(ticker_data, tickers):
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

    recent = {
        "PULLBACK": [],
        "BASING": [],
        "12/22": []
    }

    for ticker in tickers:

        df = ticker_data.get(ticker)

        if df is None:
            continue

        for setup_name, definitions in lifecycle_definitions.items():

            events = collect_trigger_events(
                df,
                definitions[0],
                definitions[1],
                definitions[2]
            )

            if not events:
                continue

            # Only the most recent trigger for a ticker/setup
            # is needed in the 5-day monitoring list.
            event = events[0].copy()
            event["ticker"] = ticker

            recent[setup_name].append(event)

    # --------------------------------------------------------
    # Sort newest triggers first.
    # --------------------------------------------------------

    for setup_name in recent:
        recent[setup_name].sort(
            key=lambda item: item["date"],
            reverse=True
        )

    print()
    print("=" * 65)
    print(
        f"RECENT TRIGGERS — "
        f"LAST {RECENT_TRIGGER_SESSIONS} TRADING DAYS"
    )
    print("=" * 65)

    for setup_name in ["PULLBACK", "BASING", "12/22"]:

        print()
        print(setup_name)

        if not recent[setup_name]:
            print("None")
            continue

        for item in recent[setup_name]:

            gap_class = classify_gap(
                item["gap_pct"]
            )

            print(
                f"{item['ticker']}: "
                f"{item['date'].strftime('%Y-%m-%d')} "
                f"({item['days_since']}d ago) "
                f"{gap_class}"
            )

            print(
                f"  Close {item['close']:.2f} | "
                f"Gap {item['gap_pct']:+.2f}% | "
                f"RVOL {item['rvol']:.2f} | "
                f"Spr/20 {item['spread_ratio20']:.2f} | "
                f"ClosePos {item['close_pos']:.2f}"
            )


# ============================================================
# CURRENT CANDIDATE CALIBRATION
# ============================================================

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

        print(
            f"Current candidates: {len(candidates)}"
        )

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

        print(
            f"Gap up   (>{DIAGNOSTIC_FLAT_GAP_PCT:.2f}%): "
            f"{gap_up}"
        )

        print(
            f"Flat     (±{DIAGNOSTIC_FLAT_GAP_PCT:.2f}%): "
            f"{flat}"
        )

        print(
            f"Gap down (<-{DIAGNOSTIC_FLAT_GAP_PCT:.2f}%): "
            f"{gap_down}"
        )

        print()
        print(
            f"Close in upper half:           "
            f"{bullish_close}"
        )

        print(
            f"RVOL >= 1.25:                  "
            f"{rvol_125}"
        )

        print(
            f"RVOL >= 1.50:                  "
            f"{rvol_150}"
        )

        print(
            f"SpreadRatio20 >= 1.00:         "
            f"{spread_100}"
        )

        print(
            f"SpreadRatio20 >= 1.25:         "
            f"{spread_125}"
        )

        print(
            f"SpreadRatio20 >= 1.50:         "
            f"{spread_150}"
        )

        print()
        print(
            f"RVOL >= 1.25 + "
            f"SpreadRatio20 >= 1.00:        "
            f"{effort_result_125_100}"
        )

        print(
            f"RVOL >= 1.50 + "
            f"SpreadRatio20 >= 1.00:        "
            f"{effort_result_150_100}"
        )

        print(
            f"RVOL >= 1.25 + "
            f"SpreadRatio20 >= 1.25:        "
            f"{effort_result_125_125}"
        )

        print()
        print("GAP-SPECIFIC OBSERVATIONS")

        print(
            f"Gap-up + RVOL >= 1.00:         "
            f"{gap_up_rvol}"
        )

        print(
            f"Gap-up + close upper half:     "
            f"{gap_up_bullish}"
        )

        print(
            f"Flat + RVOL >= 1.25:           "
            f"{flat_rvol}"
        )

        print(
            f"Flat + SpreadRatio20 >= 1.00: "
            f"{flat_spread}"
        )

        print(
            f"Flat + effort/result + bullish close:"
            f"{flat_effort_result:>12}"
        )

        print(
            f"Gap-down + close upper half:   "
            f"{gap_down_bullish}"
        )

        print(
            f"Gap-down + effort/result:      "
            f"{gap_down_effort_result}"
        )


# ============================================================
# DIAGNOSTIC
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
    print("=" * 65)
    print(f"DIAGNOSTIC: {ticker}")
    print("=" * 65)

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
        f"Latest date:     "
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

    # --------------------------------------------------------
    # Pullback
    # --------------------------------------------------------

    print()
    print("PULLBACK CONDITIONS")

    print(
        f"3M > +15%:          "
        f"{latest['Perf3M'] > 15}"
    )

    print(
        f"Close < SMA12:      "
        f"{latest['Close'] < latest['SMA12']}"
    )

    print(
        f"Close >= SMA22:     "
        f"{latest['Close'] >= latest['SMA22']}"
    )

    print(
        f"SMA55 > SMA200:     "
        f"{latest['SMA55'] > latest['SMA200']}"
    )

    print()
    print("PULLBACK TRIGGER")

    print(
        f"Close > SMA12:      "
        f"{latest['Close'] > latest['SMA12']}"
    )

    print(
        f"RVOL > 1:           "
        f"{latest['RVOL'] > 1}"
    )

    print(
        f"Close > 50% range:  "
        f"{latest['Close'] > ((latest['High'] + latest['Low']) / 2)}"
    )

    print(
        f"TRIGGER:            "
        f"{pullback_trigger(latest)}"
    )

    # --------------------------------------------------------
    # Basing
    # --------------------------------------------------------

    print()
    print("BASING CONDITIONS")

    print(
        f"3M < -15%:          "
        f"{latest['Perf3M'] < -15}"
    )

    print(
        f"SMA22 < SMA55:      "
        f"{latest['SMA22'] < latest['SMA55']}"
    )

    print(
        f"SMA12 > SMA22:      "
        f"{latest['SMA12'] > latest['SMA22']}"
    )

    print()
    print("BASING TRIGGER")

    print(
        f"Close > SMA12:      "
        f"{latest['Close'] > latest['SMA12']}"
    )

    print(
        f"RVOL > 1:           "
        f"{latest['RVOL'] > 1}"
    )

    print(
        f"Close > 50% range:  "
        f"{latest['Close'] > ((latest['High'] + latest['Low']) / 2)}"
    )

    print(
        f"TRIGGER:            "
        f"{basing_trigger(latest)}"
    )

    # --------------------------------------------------------
    # 12/22
    # --------------------------------------------------------

    print()
    print("12/22 CONDITIONS")

    print(
        f"SMA22 < SMA55:       "
        f"{latest['SMA22'] < latest['SMA55']}"
    )

    print(
        f"Yesterday 12 <= 22:  "
        f"{previous['SMA12'] <= previous['SMA22']}"
    )

    print(
        f"Today 12 > 22:       "
        f"{latest['SMA12'] > latest['SMA22']}"
    )

    print()
    print("12/22 TRIGGER")

    print(
        f"RVOL > 1:             "
        f"{latest['RVOL'] > 1}"
    )

    print(
        f"Close > 50% range:    "
        f"{latest['Close'] > ((latest['High'] + latest['Low']) / 2)}"
    )

    print(
        f"TRIGGER:              "
        f"{crossover_trigger(latest)}"
    )

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    print()
    print("CURRENT LIFECYCLE STATE")

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

    for name, definitions in lifecycle_definitions.items():

        state = replay_lifecycle(
            df,
            definitions[0],
            definitions[1],
            definitions[2]
        )

        if state["status"] == "NONE":

            print(f"{name}: NONE")

        else:

            setup_date = (
                state["setup_date"].strftime("%Y-%m-%d")
                if state["setup_date"] is not None
                else "-"
            )

            print(
                f"{name}: {state['status']} "
                f"(setup {setup_date}, "
                f"age {state['age']}d)"
            )


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
        help="Show detailed diagnostics for one or more tickers"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Diagnostic mode
    # Only requested tickers are downloaded.
    # --------------------------------------------------------

    if args.diagnostic:

        tickers = [
            ticker.upper()
            for ticker in args.diagnostic
        ]

        data = download_data(tickers)

        for ticker in tickers:
            print_diagnostic(
                ticker,
                data
            )

    # --------------------------------------------------------
    # Normal full-universe scan
    # --------------------------------------------------------

    else:

        tickers = load_tickers()

        data = download_data(
            tickers
        )

        run_scan(
            data,
            tickers
        )


if __name__ == "__main__":
    main()
