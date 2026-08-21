import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json

# === Configuration ===
# These will always show on the dashboard by default
CORE_TICKERS = [
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"
]

# These are pre-calculated, embedded in the HTML, and get a checkbox toggle
EXTENDED_UNIVERSE = [
    "IGV", "CLOU", "XBI", "IYT", "KRE", "XHB", "XRT", "XME", "ROBO", "UFO", "SOXX", "TAN"
]

BENCHMARK = "SPY"
TRAIL_LENGTH = 36
RS_LEN = 63
CENTER_LEN = 252
LOOKBACK_3M = 63
LOOKBACK_1M = 21

def fetch_and_calculate(tickers, benchmark):
    print("Fetching data from Yahoo Finance...")
    all_symbols = list(dict.fromkeys(tickers + [benchmark]))
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
            "bucket": buckets[-1],
            "is_core": sym in CORE_TICKERS
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
        
        # Determine visibility for initialization (Core = True, Extended = False)
        is_visible = True if data["is_core"] else False
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers',
            name=sym,
            visible=is_visible,
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

    fig.add_hline(y=0, line_width=1.5, line_color="#363a45", line_dash="solid")
    fig.add_vline(x=0, line_width=1.5, line_color="#363a45", line_dash="solid")
    
    fig.update_layout(
        title=dict(text="Systematic Rotation Tracker (36-Day Macro Tails)", font=dict(size=18, color="#ffffff")),
        xaxis=dict(title=dict(text="Relative Strength (RS)", font=dict(size=14, color="#d1d4dc")), gridcolor="#2a2e39", zeroline=False, tickfont=dict(color="#787b86")),
        yaxis=dict(title=dict(text="Relative Momentum (RM)", font=dict(size=14, color="#d1d4dc")), gridcolor="#2a2e39", zeroline=False, tickfont=dict(color="#787b86")),
        plot_bgcolor="#131722", paper_bgcolor="#131722", font=dict(color="#d1d4dc"),
        showlegend=True, height=680, margin=dict(l=60, r=50, t=80, b=60), hovermode="closest",
        legend=dict(itemsizing='constant', title=dict(text="Hover to Isolate", font=dict(size=12, color="#787b86")))
    )
    
    fig.add_annotation(x=0.95, y=0.95, text="LEADING", showarrow=False, font=dict(color="rgba(38, 166, 154, 0.15)", size=28, weight="bold"), xref="x domain", yref="y domain")
    fig.add_annotation(x=0.95, y=0.05, text="WEAKENING", showarrow=False, font=dict(color="rgba(255, 167, 38, 0.15)", size=28, weight="bold"), xref="x domain", yref="y domain")
    fig.add_annotation(x=0.05, y=0.05, text="LAGGING", showarrow=False, font=dict(color="rgba(239, 83, 80, 0.15)", size=28, weight="bold"), xref="x domain", yref="y domain")
    fig.add_annotation(x=0.05, y=0.95, text="IMPROVING", showarrow=False, font=dict(color="rgba(156, 204, 101, 0.15)", size=28, weight="bold"), xref="x domain", yref="y domain")

    return fig.to_html(full_html=False, include_plotlyjs='https://unpkg.com/plotly.js@3.7.0/dist/plotly.min.js', div_id="rrg-plotly-chart")

def generate_dashboard(df, graph_html, core_tickers, extended_tickers, filename):
    
    # Dynamically build the checkbox HTML based on the EXTENDED_UNIVERSE list
    checkboxes_html = ""
    for ticker in extended_tickers:
        checkboxes_html += f'<label class="cb-label"><input type="checkbox" class="ticker-cb" value="{ticker}"> {ticker}</label>\n'

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Institutional RRG Risk Architecture</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: #131722; color: #d1d4dc; padding: 30px; margin: 0; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header-bar { border-bottom: 2px solid #2a2e39; padding-bottom: 15px; margin-bottom: 20px; }
        h2 { color: #ffffff; margin: 0 0 10px 0;}
        
        .checkbox-container { display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px; }
        .cb-label { display: flex; align-items: center; gap: 6px; cursor: pointer; color: #787b86; font-size: 13px; font-weight: 600; transition: color 0.2s; }
        .cb-label input { cursor: pointer; accent-color: #26a69a; width: 15px; height: 15px; margin: 0; }
        .cb-label:hover { color: #d1d4dc; }
        
        .data-card { background-color: #1e222d; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #2a2e39; }
        .data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
        .data-table th { background-color: #181c25; color: #787b86; font-weight: 600; text-transform: uppercase; padding: 14px 15px; border-bottom: 1px solid #2a2e39; font-size: 11px; }
        .data-table td { padding: 12px 15px; border-bottom: 1px solid #2a2e39; vertical-align: middle; }
        .data-table tr:hover { background-color: #2a2e39; }
        .data-table th, .data-table td { text-align: center !important; }
        .data-table th:nth-child(1), .data-table td:nth-child(1) { text-align: left !important; }
        .data-table th:nth-child(2), .data-table td:nth-child(2), .data-table th:nth-child(3), .data-table td:nth-child(3) { text-align: right !important; }
        .data-table th:nth-child(9), .data-table td:nth-child(9), .data-table th:nth-child(10), .data-table td:nth-child(10) { text-align: right !important; }

        .bucket-Leading { background-color: rgba(38, 166, 154, 0.15); color: #26a69a; font-weight: 600; border-radius: 4px; padding: 4px 8px; display: inline-block; width: 80px; text-align: center;}
        .bucket-Improving { background-color: rgba(156, 204, 101, 0.15); color: #9ccc65; font-weight: 600; border-radius: 4px; padding: 4px 8px; display: inline-block; width: 80px; text-align: center;}
        .bucket-Weakening { background-color: rgba(255, 167, 38, 0.15); color: #ffa726; font-weight: 600; border-radius: 4px; padding: 4px 8px; display: inline-block; width: 80px; text-align: center;}
        .bucket-Lagging { background-color: rgba(239, 83, 80, 0.15); color: #ef5350; font-weight: 600; border-radius: 4px; padding: 4px 8px; display: inline-block; width: 80px; text-align: center;}
        
        .legendtext { cursor: pointer; }
    </style>
    </head>
    <body>
        <div class="container">
            <div class="header-bar">
                <h2>Systematic Alpha Matrix</h2>
                <div class="checkbox-container">
                    {checkboxes}
                </div>
            </div>
            
            <div class="data-card" style="margin-bottom: 25px;">
                {graph}
            </div>
            
            <div class="data-card">
                {table}
            </div>
        </div>

        <script>
            const coreTickers = {core_tickers_json};
            
            window.addEventListener('load', function() {
                const plotDiv = document.getElementById('rrg-plotly-chart');
                const checkboxes = document.querySelectorAll('.ticker-cb');
                const tableRows = document.querySelectorAll('.data-table tbody tr');
                
                // 1. Tag HTML Table Rows with data attributes
                tableRows.forEach(row => {
                    const tickerCell = row.querySelector('td:first-child');
                    if(tickerCell) {
                        row.setAttribute('data-ticker', tickerCell.textContent.trim());
                    }
                });

                // 2. Main function to update UI state
                function updateVisibility(activeExtendedTickers) {
                    // Update Plotly
                    if(plotDiv && plotDiv.data) {
                        const visibilityUpdate = plotDiv.data.map(trace => {
                            return (coreTickers.includes(trace.name) || activeExtendedTickers.includes(trace.name)) ? true : false;
                        });
                        Plotly.restyle(plotDiv, {'visible': visibilityUpdate});
                    }
                    
                    // Update Table
                    tableRows.forEach(row => {
                        const ticker = row.getAttribute('data-ticker');
                        if(coreTickers.includes(ticker) || activeExtendedTickers.includes(ticker)) {
                            row.style.display = 'table-row';
                        } else {
                            row.style.display = 'none';
                        }
                    });
                }

                // 3. Update URL without reloading the page
                function updateUrlParams(activeTickers) {
                    const url = new URL(window.location);
                    if (activeTickers.length > 0) {
                        url.searchParams.set('tickers', activeTickers.join(','));
                    } else {
                        url.searchParams.delete('tickers');
                    }
                    window.history.replaceState({}, '', url);
                }

                // 4. Initialize state from URL on page load
                const urlParams = new URLSearchParams(window.location.search);
                const urlTickersParam = urlParams.get('tickers');
                let initialActiveTickers = [];
                
                if (urlTickersParam) {
                    initialActiveTickers = urlTickersParam.split(',').map(t => t.trim().toUpperCase());
                    // Check the corresponding boxes
                    checkboxes.forEach(cb => {
                        if (initialActiveTickers.includes(cb.value)) {
                            cb.checked = true;
                        }
                    });
                }
                
                // Run initial visibility check
                updateVisibility(initialActiveTickers);

                // 5. Handle Checkbox Toggles
                checkboxes.forEach(cb => {
                    cb.addEventListener('change', function() {
                        const checkedVals = Array.from(document.querySelectorAll('.ticker-cb:checked')).map(box => box.value);
                        updateVisibility(checkedVals);
                        updateUrlParams(checkedVals);
                    });
                });

                // 6. Plotly Hover Effects
                if(plotDiv) {
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
                }
            });
        </script>
    </body>
    </html>
    """

    def color_bucket(val):
        return f'<span class="bucket-{val}">{val}</span>'

    df_html = df.copy()
    df_html['RRG Bucket'] = df_html['RRG Bucket'].apply(color_bucket)
    table_html = df_html.to_html(index=False, escape=False, border=0)
    table_html = table_html.replace('class="dataframe"', 'class="data-table"')
    
    # Inject variables into the HTML template
    final_html = html_template.replace('{graph}', graph_html)\
                              .replace('{table}', table_html)\
                              .replace('{checkboxes}', checkboxes_html)\
                              .replace('{core_tickers_json}', json.dumps(core_tickers))
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"Operational Dashboard deployed successfully at: {filename}")

if __name__ == "__main__":
    os.makedirs("rrg", exist_ok=True)
    
    # Combine lists to pre-fetch everything at once
    combined_universe = CORE_TICKERS + EXTENDED_UNIVERSE
    print(f"Calculating {len(combined_universe)} total tickers for static embedding...")
    
    final_table, graph_data = fetch_and_calculate(combined_universe, BENCHMARK)
    sorted_tickers_list = final_table['Ticker'].tolist()
    graph_html = build_plotly_figure(graph_data, sorted_tickers_list)
    
    generate_dashboard(final_table, graph_html, CORE_TICKERS, EXTENDED_UNIVERSE, "rrg/rrg_dashboard.html")