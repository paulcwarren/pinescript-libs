import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# === Configuration ===
TICKERS = [
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"
#    "XRT", "XME", "XBI", "ROBO", "QTUM", "URA", "SOXX", "UFO", "MAGS", "TAN", "IGV", 
#    "DRAM", "IYT", "DTCR"
]
BENCHMARK = "SPY"
TRAIL_LENGTH = 36

RS_LEN = 63
CENTER_LEN = 252
LOOKBACK_3M = 63
LOOKBACK_1M = 21

def fetch_and_calculate(tickers, benchmark):
    print("Fetching data from Yahoo Finance...")
    all_symbols = tickers + [benchmark]
    data = yf.download(all_symbols, period="3y", interval="1d", progress=False)['Close']
    data = data.dropna(axis=1, how='all').ffill()
    
    bench = data[benchmark]
    table_results = []
    graph_data = {}

    print("Processing metrics...")
    for sym in tickers:
        if sym not in data.columns:
            continue
            
        sec = data[sym]
        ratio = sec / bench
        
        rs_comp = ratio.ewm(span=RS_LEN, adjust=False).mean()
        roc = ratio.pct_change() * 100
        roc = roc.fillna(0)
        rm_comp_smoothed = roc.ewm(span=RS_LEN, adjust=False).mean()
        
        rs_center = rs_comp.ewm(span=CENTER_LEN, adjust=False).mean()
        rm_center = rm_comp_smoothed.ewm(span=CENTER_LEN, adjust=False).mean()
        
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
        current_bucket = buckets[-1]
        days = 1
        for b in reversed(buckets[:-1]):
            if b == current_bucket:
                days += 1
            else:
                break
                
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

    df = pd.DataFrame(table_results)
    df['RS Rnk'] = df['RS Val'].rank(ascending=False, method='min').astype(int)
    df['RM Rnk'] = df['RM Val'].rank(ascending=False, method='min').astype(int)
    df['Avg Rnk'] = (df['RS Rnk'] + df['RM Rnk']) / 2.0
    
    df = df.sort_values(by=['Prio', 'Avg Rnk'], ascending=[False, True]).reset_index(drop=True)
    
    df['3M Perf'] = (df['3M Perf'] * 100).map("{:.1f}%".format)
    df['1M Perf'] = (df['1M Perf'] * 100).map("{:.1f}%".format)
    df['RS Val'] = df['RS Val'].map("{:.4f}".format)
    df['RM Val'] = df['RM Val'].map("{:.2f}".format)
    
    display_df = df[['Ticker', '3M Perf', '1M Perf', 'RRG Bucket', 'Days', 'RS Rnk', 'RM Rnk', 'Avg Rnk', 'RS Val', 'RM Val']]
    return display_df, graph_data

def build_plotly_figure(graph_data, sorted_tickers):
    fig = go.Figure()
    
    bucket_colors = {
        "Leading": "#26a69a", "Weakening": "#ffa726", 
        "Lagging": "#ef5350", "Improving": "#9ccc65"
    }

    for sym in sorted_tickers:
        if sym not in graph_data:
            continue
            
        data = graph_data[sym]
        x_vals, y_vals = data["x"], data["y"]
        color = bucket_colors.get(data["bucket"], "#ffffff")
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers',
            name=sym,
            line=dict(color=color, width=1.5),
            opacity=0.4, 
            marker=dict(
                size=[3] * (len(x_vals)-1) + [8], 
                color=color,
                line=dict(width=1, color='#ffffff')
            ),
            customdata=[sym] * len(x_vals),
            hovertemplate=f"<b>{sym}</b><br>RS: %{{x:.4f}}<br>RM: %{{y:.4f}}<extra></extra>"
        ))

    # Center Crosshairs
    fig.add_hline(y=0, line_width=1.5, line_color="#363a45", line_dash="solid")
    fig.add_vline(x=0, line_width=1.5, line_color="#363a45", line_dash="solid")
    
    # RESTORED AND UPGRADED AXIS CONFIGURATION
    fig.update_layout(
        title=dict(text="Systematic Rotation Tracker (36-Day Macro Tails)", font=dict(size=18, color="#ffffff")),
        xaxis=dict(
            title=dict(text="Relative Strength (RS)", font=dict(size=14, color="#d1d4dc")),
            gridcolor="#2a2e39",
            zeroline=False, # Handled by our custom hline/vline
            tickfont=dict(color="#787b86")
        ),
        yaxis=dict(
            title=dict(text="Relative Momentum (RM)", font=dict(size=14, color="#d1d4dc")),
            gridcolor="#2a2e39",
            zeroline=False,
            tickfont=dict(color="#787b86")
        ),
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc"),
        showlegend=True,
        height=680,
        margin=dict(l=60, r=50, t=80, b=60), # Slightly widened left/bottom margins for axis text
        hovermode="closest",
        legend=dict(itemsizing='constant', title=dict(text="Hover to Isolate", font=dict(size=12, color="#787b86")))
    )
    
    fig.add_annotation(x=0.95, y=0.95, text="LEADING", showarrow=False, font=dict(color="rgba(38, 166, 154, 0.15)", size=28, weight="bold"), xref="x domain", yref="y domain")
    fig.add_annotation(x=0.95, y=0.05, text="WEAKENING", showarrow=False, font=dict(color="rgba(255, 167, 38, 0.15)", size=28, weight="bold"), xref="x domain", yref="y domain")
    fig.add_annotation(x=0.05, y=0.05, text="LAGGING", showarrow=False, font=dict(color="rgba(239, 83, 80, 0.15)", size=28, weight="bold"), xref="x domain", yref="y domain")
    fig.add_annotation(x=0.05, y=0.95, text="IMPROVING", showarrow=False, font=dict(color="rgba(156, 204, 101, 0.15)", size=28, weight="bold"), xref="x domain", yref="y domain")

    return fig.to_html(full_html=False, include_plotlyjs='cdn', div_id="rrg-plotly-chart")
    
def generate_dashboard(df, graph_html, filename):
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Institutional RRG Risk Architecture</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: #131722; color: #d1d4dc; padding: 30px; margin: 0; }
        .container { max-width: 1400px; margin: 0 auto; }
        h2 { color: #ffffff; border-bottom: 2px solid #2a2e39; padding-bottom: 10px; margin-top: 0; margin-bottom: 20px;}
        .layout-grid { display: grid; grid-template-columns: 100%; gap: 25px; margin-bottom: 25px; }
        
        .matrix-card { background-color: #1e222d; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #2a2e39; }
        .matrix-title { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 15px; color: #787b86; }
        .matrix-table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
        .matrix-table th { background-color: #181c25; color: #d1d4dc; font-weight: 600; padding: 10px; border: 1px solid #2a2e39; text-transform: uppercase; font-size: 11px; }
        .matrix-table td { padding: 12px 10px; border: 1px solid #2a2e39; font-weight: 500; }
        
        .status-hold { background-color: rgba(38, 166, 154, 0.2); color: #26a69a; }
        .status-observe { background-color: rgba(255, 167, 38, 0.2); color: #ffa726; border-left: 4px solid #ffa726 !important; }
        .status-exit { background-color: rgba(239, 83, 80, 0.25); color: #ef5350; font-weight: bold; animation: pulse 2s infinite; }
        .status-avoid { color: #787b86; background-color: rgba(30, 34, 45, 0.5); }
        .status-alpha { background-color: rgba(156, 204, 101, 0.15); color: #9ccc65; }

        .data-card { background-color: #1e222d; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #2a2e39; }
        
        /* Strict Data Table Alignment Rules */
        .data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
        .data-table th { background-color: #181c25; color: #787b86; font-weight: 600; text-transform: uppercase; padding: 14px 15px; border-bottom: 1px solid #2a2e39; font-size: 11px; }
        .data-table td { padding: 12px 15px; border-bottom: 1px solid #2a2e39; vertical-align: middle; }
        .data-table tr:hover { background-color: #2a2e39; }
        
        /* Force Center by default to fix Pandas overrides */
        .data-table th, .data-table td { text-align: center !important; }
        
        /* 1: Ticker (Left) */
        .data-table th:nth-child(1), .data-table td:nth-child(1) { text-align: left !important; }
        
        /* 2 & 3: Percentages (Right for decimal alignment) */
        .data-table th:nth-child(2), .data-table td:nth-child(2),
        .data-table th:nth-child(3), .data-table td:nth-child(3) { text-align: right !important; }
        
        /* 9 & 10: Raw RS/RM Values (Right for decimal alignment) */
        .data-table th:nth-child(9), .data-table td:nth-child(9),
        .data-table th:nth-child(10), .data-table td:nth-child(10) { text-align: right !important; }

        .bucket-Leading { background-color: rgba(38, 166, 154, 0.15); color: #26a69a; font-weight: 600; border-radius: 4px; padding: 4px 8px; display: inline-block; width: 80px; text-align: center;}
        .bucket-Improving { background-color: rgba(156, 204, 101, 0.15); color: #9ccc65; font-weight: 600; border-radius: 4px; padding: 4px 8px; display: inline-block; width: 80px; text-align: center;}
        .bucket-Weakening { background-color: rgba(255, 167, 38, 0.15); color: #ffa726; font-weight: 600; border-radius: 4px; padding: 4px 8px; display: inline-block; width: 80px; text-align: center;}
        .bucket-Lagging { background-color: rgba(239, 83, 80, 0.15); color: #ef5350; font-weight: 600; border-radius: 4px; padding: 4px 8px; display: inline-block; width: 80px; text-align: center;}
        
        .legendtext { cursor: pointer; }
    </style>
    </head>
    <body>
        <div class="container">
            <h2>Systematic Alpha Matrix & Execution Framework</h2>
            
            <div class="data-card" style="margin-bottom: 25px;">
                {graph}
            </div>
            
            <div class="layout-grid">
                <div class="matrix-card">
                    <div class="matrix-title">Systematic Rule Logic Reference (Weekly Matrix Setup)</div>
                    <table class="matrix-table">
                        <thead>
                            <tr>
                                <th>RRG Quadrant Condition</th>
                                <th>Price Structure &gt; 36SMA</th>
                                <th>Price Structure &lt; 36SMA</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="font-weight:bold; color:#26a69a;">LEADING</td>
                                <td class="status-hold">STRONG ASSET ALLOCATION / CORE HOLD</td>
                                <td class="status-observe">RISK WARNING / PREPARE ASSET LIQUIDATION</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:#ffa726;">WEAKENING</td>
                                <td class="status-observe">MONITORING BIAS (STAY IN ASSET)</td>
                                <td class="status-exit">AND CONDITION VALIDATED &rarr; HARD EXIT SYSTEM CONSTRAINTS</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:#ef5350;">LAGGING</td>
                                <td class="status-alpha">EARLY LONG CONVICTION DISCOVERY ZONE</td>
                                <td class="status-avoid">CORE SYSTEM AVOIDANCE ARCHITECTURE</td>
                            </tr>
                            <tr>
                                <td style="font-weight:bold; color:#9ccc65;">IMPROVING</td>
                                <td class="status-hold">SYSTEM ENTRY CONFIRMATION WINDOW</td>
                                <td class="status-avoid">MOMENTUM NOISE / UNCONFIRMED BY TREND</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="data-card">
                {table}
            </div>
        </div>

        <script>
            window.addEventListener('load', function() {
                var plotDiv = document.getElementById('rrg-plotly-chart');
                if (!plotDiv) return;

                function resetTraces() {
                    Plotly.restyle(plotDiv, {
                        'opacity': plotDiv.data.map(() => 0.4),
                        'line.width': plotDiv.data.map(() => 1.5)
                    });
                }

                function highlightTrace(index) {
                    Plotly.restyle(plotDiv, {
                        'opacity': plotDiv.data.map((_, i) => (i === index) ? 1.0 : 0.08),
                        'line.width': plotDiv.data.map((_, i) => (i === index) ? 3.5 : 1.5)
                    });
                }

                plotDiv.on('plotly_hover', function(data) {
                    if(data.points.length > 0) highlightTrace(data.points[0].curveNumber);
                });
                plotDiv.on('plotly_unhover', resetTraces);

                plotDiv.addEventListener('mouseover', function(e) {
                    let el = e.target;
                    let traceNode = null;
                    while (el && el !== plotDiv) {
                        if (el.classList && el.classList.contains('traces')) {
                            traceNode = el;
                            break;
                        }
                        el = el.parentNode;
                    }

                    if (traceNode) {
                        let textNode = traceNode.querySelector('.legendtext');
                        if (textNode) {
                            let traceName = textNode.textContent;
                            let traceIndex = plotDiv.data.findIndex(t => t.name === traceName);
                            if (traceIndex > -1) highlightTrace(traceIndex);
                        }
                    }
                });

                plotDiv.addEventListener('mouseout', function(e) {
                    let el = e.target;
                    while (el && el !== plotDiv) {
                        if (el.classList && el.classList.contains('traces')) {
                            resetTraces();
                            break;
                        }
                        el = el.parentNode;
                    }
                });
            });
        </script>
    </body>
    </html>
    """

    def color_bucket(val):
        return f'<span class="bucket-{val}">{val}</span>'

    df_html = df.copy()
    df_html['RRG Bucket'] = df_html['RRG Bucket'].apply(color_bucket)
    
    # Strip Pandas' inline alignments and inject our class
    table_html = df_html.to_html(index=False, escape=False, border=0)
    table_html = table_html.replace('class="dataframe"', 'class="data-table"')
    
    final_html = html_template.replace('{graph}', graph_html).replace('{table}', table_html)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"Operational Dashboard deployed successfully at: {filename}")

if __name__ == "__main__":
    os.makedirs("rrg", exist_ok=True)
    
    # 1. Fetch data and calculate table
    final_table, graph_data = fetch_and_calculate(TICKERS, BENCHMARK)
    
    # 2. Extract the sorted list of tickers directly from the final DataFrame
    sorted_tickers_list = final_table['Ticker'].tolist()
    
    # 3. Pass the sorted list into the Plotly builder
    graph_html = build_plotly_figure(graph_data, sorted_tickers_list)
    
    # 4. Generate the dashboard
    generate_dashboard(final_table, graph_html, "rrg/rrg_dashboard.html")
