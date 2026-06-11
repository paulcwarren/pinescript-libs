import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# === Configuration ===
TICKERS = [
    "IGV", "XLY", "XLP", "XLF", "XLV", "XLI", "XRT", "XME", "XLC", 
    "XBI", "ROBO", "XLE", "QTUM", "URA", "SOXX", "UFO", "MAGS",
    "DRAM", "IYT", "DTCR"
]
BENCHMARK = "SPY"
TRAIL_LENGTH = 10 

RS_LEN = 63
CENTER_LEN = 252
LOOKBACK_3M = 63
LOOKBACK_1M = 21

def fetch_and_calculate(tickers, benchmark):
    print("Fetching data from Yahoo Finance...")
    all_symbols = tickers + [benchmark]
    data = yf.download(all_symbols, period="3y", interval="1d")['Close']
    data = data.dropna(axis=1, how='all').ffill()
    
    bench = data[benchmark]
    table_results = []
    graph_data = {}

    print("Processing tickers for Table and Graph...")
    for sym in tickers:
        if sym not in data.columns:
            continue
            
        sec = data[sym]
        ratio = sec / bench
        
        # RS and RM Components
        rs_comp = ratio.ewm(span=RS_LEN, adjust=False).mean()
        roc = ratio.pct_change() * 100
        roc = roc.fillna(0)
        rm_comp_smoothed = roc.ewm(span=RS_LEN, adjust=False).mean()
        
        # Center Lines
        rs_center = rs_comp.ewm(span=CENTER_LEN, adjust=False).mean()
        rm_center = rm_comp_smoothed.ewm(span=CENTER_LEN, adjust=False).mean()
        
        # Bucket Logic
        is_rs_strong = rs_comp > rs_center
        is_rm_improving = rm_comp_smoothed > rm_center
        
        conditions = [
            (is_rs_strong) & (is_rm_improving),
            (is_rs_strong) & (~is_rm_improving),
            (~is_rs_strong) & (~is_rm_improving),
            (~is_rs_strong) & (is_rm_improving)
        ]
        choices = ["Leading", "Weakening", "Lagging", "Improving"]
        buckets = np.select(conditions, choices, default="Neutral")
        
        x_plot = rs_comp - rs_center
        y_plot = rm_comp_smoothed - rm_center
        
        graph_data[sym] = {
            "x": x_plot.iloc[-TRAIL_LENGTH:].tolist(),
            "y": y_plot.iloc[-TRAIL_LENGTH:].tolist(),
            "bucket": buckets[-1]
        }
        
        prio_map = {"Leading": 4, "Weakening": 3, "Lagging": 2, "Improving": 1, "Neutral": 0}
        
        # Consecutive Days Logic
        current_bucket = buckets[-1]
        days = 1
        for b in reversed(buckets[:-1]):
            if b == current_bucket:
                days += 1
            else:
                break
                
        # Performance Metrics
        p1, p2 = np.nan, np.nan
        if len(sec) > LOOKBACK_3M:
            p1 = (sec.iloc[-1] / sec.iloc[-LOOKBACK_3M - 1]) / (bench.iloc[-1] / bench.iloc[-LOOKBACK_3M - 1]) - 1
            
        if len(sec) > LOOKBACK_1M:
            p2 = (sec.iloc[-1] / sec.iloc[-LOOKBACK_1M - 1]) / (bench.iloc[-1] / bench.iloc[-LOOKBACK_1M - 1]) - 1

        table_results.append({
            "Ticker": sym, "3M Perf": p1, "1M Perf": p2, "RRG Bucket": current_bucket,
            "Days": days, "Prio": prio_map.get(current_bucket, 0),
            "RS Val": rs_comp.iloc[-1], "RM Val": rm_comp_smoothed.iloc[-1]
        })

    # Build Table DataFrame
    df = pd.DataFrame(table_results)
    df['RS Rnk'] = df['RS Val'].rank(ascending=False, method='min').astype(int)
    df['RM Rnk'] = df['RM Val'].rank(ascending=False, method='min').astype(int)
    df['Avg Rnk'] = (df['RS Rnk'] + df['RM Rnk']) / 2.0
    
    df = df.sort_values(by=['Prio', 'Avg Rnk'], ascending=[False, True]).reset_index(drop=True)
    
    # Format text for HTML
    df['3M Perf'] = (df['3M Perf'] * 100).map("{:.1f}%".format)
    df['1M Perf'] = (df['1M Perf'] * 100).map("{:.1f}%".format)
    df['RS Val'] = df['RS Val'].map("{:.4f}".format)
    df['RM Val'] = df['RM Val'].map("{:.2f}".format)
    
    display_df = df[['Ticker', '3M Perf', '1M Perf', 'RRG Bucket', 'Days', 'RS Rnk', 'RM Rnk', 'Avg Rnk', 'RS Val', 'RM Val']]
    
    return display_df, graph_data

def build_plotly_figure(graph_data):
    fig = go.Figure()
    
    bucket_colors = {
        "Leading": "#26a69a", "Weakening": "#ffa726", 
        "Lagging": "#ef5350", "Improving": "#9ccc65"
    }

    for sym, data in graph_data.items():
        x_vals, y_vals = data["x"], data["y"]
        color = bucket_colors.get(data["bucket"], "#ffffff")
        
        # Add trail line
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers',
            name=sym,
            line=dict(color=color, width=2),
            marker=dict(
                size=[4] * (len(x_vals)-1) + [10], 
                color=color
            ),
            hovertemplate=f"<b>{sym}</b><br>RS: %{{x:.4f}}<br>RM: %{{y:.4f}}<extra></extra>"
        ))

    # Add quadrant backgrounds and crosshairs
    fig.add_hline(y=0, line_width=2, line_color="#363a45")
    fig.add_vline(x=0, line_width=2, line_color="#363a45")
    
    # Update layout to match dark mode theme
    fig.update_layout(
        title="Interactive Relative Rotation Graph (RRG)",
        xaxis_title="Relative Strength (RS)",
        yaxis_title="Relative Momentum (RM)",
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc"),
        showlegend=True,
        height=700,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    # CORRECTED QUADRANT LABELS
    fig.add_annotation(x=0.95, y=0.95, text="LEADING", showarrow=False, font=dict(color="rgba(38, 166, 154, 0.3)", size=30), xref="x domain", yref="y domain")
    fig.add_annotation(x=0.95, y=0.05, text="WEAKENING", showarrow=False, font=dict(color="rgba(255, 167, 38, 0.3)", size=30), xref="x domain", yref="y domain")
    fig.add_annotation(x=0.05, y=0.05, text="LAGGING", showarrow=False, font=dict(color="rgba(239, 83, 80, 0.3)", size=30), xref="x domain", yref="y domain")
    fig.add_annotation(x=0.05, y=0.95, text="IMPROVING", showarrow=False, font=dict(color="rgba(156, 204, 101, 0.3)", size=30), xref="x domain", yref="y domain")

    # Generate HTML div block for the graph
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def generate_dashboard(df, graph_html, filename):
    print(f"Building HTML Dashboard at {filename}...")
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Sector RRG Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: #131722; color: #d1d4dc; padding: 30px; margin: 0; }
        .container { max-width: 1200px; margin: 0 auto; }
        h2 { color: #ffffff; border-bottom: 2px solid #2a2e39; padding-bottom: 10px; margin-bottom: 20px;}
        .graph-container { margin-bottom: 40px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        table { border-collapse: collapse; width: 100%; font-size: 14px; background-color: #1e222d; border-radius: 8px; overflow: hidden; }
        th { background-color: #181c25; color: #787b86; font-weight: 600; text-transform: uppercase; padding: 14px 15px; text-align: left; border-bottom: 1px solid #2a2e39; }
        td { padding: 12px 15px; border-bottom: 1px solid #2a2e39; }
        tr:last-child td { border-bottom: none; }
        tr:hover { background-color: #2a2e39; }
        
        .bucket-Leading { background-color: rgba(38, 166, 154, 0.15); color: #26a69a; font-weight: 600; border-radius: 4px; padding: 4px 8px; }
        .bucket-Improving { background-color: rgba(156, 204, 101, 0.15); color: #9ccc65; font-weight: 600; border-radius: 4px; padding: 4px 8px; }
        .bucket-Weakening { background-color: rgba(255, 167, 38, 0.15); color: #ffa726; font-weight: 600; border-radius: 4px; padding: 4px 8px; }
        .bucket-Lagging { background-color: rgba(239, 83, 80, 0.15); color: #ef5350; font-weight: 600; border-radius: 4px; padding: 4px 8px; }
        .bucket-Neutral { background-color: rgba(158, 158, 158, 0.15); color: #9e9e9e; font-weight: 600; border-radius: 4px; padding: 4px 8px; }
    </style>
    </head>
    <body>
        <div class="container">
            <h2>Sector Relative Rotation Graph (RRG) vs SPY</h2>
            <div class="graph-container">
                {graph}
            </div>
            {table}
        </div>
    </body>
    </html>
    """

    def color_bucket(val):
        return f'<span class="bucket-{val}">{val}</span>'

    df_html = df.copy()
    df_html['RRG Bucket'] = df_html['RRG Bucket'].apply(color_bucket)
    table_html = df_html.to_html(index=False, escape=False, border=0)
    
    final_html = html_template.replace('{graph}', graph_html).replace('{table}', table_html)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"Success! Dashboard saved as {filename}")

if __name__ == "__main__":
    # Ensure the rrg directory exists
    os.makedirs("rrg", exist_ok=True)
    
    final_table, graph_data = fetch_and_calculate(TICKERS, BENCHMARK)
    graph_html = build_plotly_figure(graph_data)
    
    # Output path updated to point to the correct folder
    generate_dashboard(final_table, graph_html, "rrg/rrg_dashboard.html")