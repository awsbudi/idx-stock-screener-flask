import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta
from flask import Flask, render_template_string, request
import numpy as np
import json
from plotly.utils import PlotlyJSONEncoder

# --- Konfigurasi Ticker Awal ---
# Daftar Tickers yang lebih luas (Simulasi IDX Active Stocks)
# Pengguna dapat mengganti atau menambah daftar ini via input form.
DEFAULT_IDX_TICKERS = [
    "BBCA.JK", "TLKM.JK", "ASII.JK", "ACES.JK", "UNVR.JK", "GOTO.JK", "ADRO.JK", "BUMI.JK", 
    "ANTM.JK", "MDKA.JK", "BRIS.JK", "ARTO.JK", "BMRI.JK", "BBNI.JK", "BBRI.JK", "UNTR.JK",
    "ITMG.JK", "INDF.JK", "ICBP.JK", "SRIL.JK", "CPIN.JK", "EXCL.JK", "FREN.JK", "HMSP.JK",
    "INCO.JK", "KLBF.JK", "PGAS.JK", "PTBA.JK", "SMGR.JK", "TINS.JK", "SMMA.JK", "TPIA.JK"
]
IHSG_TICKER = "^JKSE"

# --- Fungsi Pengambilan Data ---
def load_historical_data(tickers, period="6mo"):
    """Mengambil data historis dari yfinance."""
    print(f"Loading data for {tickers}...")
    try:
        # Menambahkan IHSG jika belum ada untuk analisis makro
        download_tickers = list(set(tickers + [IHSG_TICKER]))
        
        data = yf.download(download_tickers, period=period)
        
        stock_data_cols = [t for t in tickers if t != IHSG_TICKER]
        
        # Penanganan MultiIndex vs SingleIndex data
        if isinstance(data.columns, pd.MultiIndex):
            stock_data = data.loc[:, (slice(None), stock_data_cols)]
            ihsg_data = data.loc[:, (slice(None), IHSG_TICKER)].droplevel(1, axis=1) if IHSG_TICKER in data.columns.get_level_values(1) else None
        else:
            stock_data = data.copy()
            ihsg_data = None 

        return stock_data, ihsg_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- Fungsi Perhitungan Indikator Teknis ---
def calculate_indicators(df, sma_short, sma_long, rsi_period, vol_period, hist_days):
    """Menghitung indikator teknis yang diminta."""
    
    # Indikator yang diperlukan untuk sorting (30 hari return)
    df['Hist_Return_30D'] = (df['Close'] / df['Close'].shift(hist_days) - 1) * 100
    df['Avg_Volume'] = df['Volume'].rolling(window=vol_period).mean()
    
    # Indikator lain (optional, tapi dihitung)
    df['SMA_Short'] = df['Close'].rolling(window=sma_short).mean()
    df['SMA_Long'] = df['Close'].rolling(window=sma_long).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

# --- Fungsi Screening dan Ekstraksi Data ---
def screen_and_extract(full_data, ihsg_data, criteria_params, indicator_params):
    """Menerapkan kriteria screening dan mengumpulkan hasil."""
    
    results = []
    chart_jsons = {}
    
    # List kolom/ticker yang ada di MultiIndex (jika ada)
    if isinstance(full_data.columns, pd.MultiIndex):
        tickers_available = full_data.columns.get_level_values(1).unique().tolist()
    elif not full_data.empty:
        # Jika hanya satu ticker yang diunduh, kolomnya hanya 1 level
        tickers_available = [full_data.columns[0]] if len(full_data.columns.names) == 1 and full_data.columns.names[0] == 'Ticker' else [IHSG_TICKER]
    else:
        return [], {}, 'N/A', 0
    
    # Analisis IHSG
    ihsg_status = "N/A"
    ihsg_change = 0
    if ihsg_data is not None and not ihsg_data.empty:
        ihsg_latest = ihsg_data['Close'].iloc[-1]
        ihsg_prev_close = ihsg_data['Close'].iloc[-2] if len(ihsg_data) > 1 else ihsg_latest
        ihsg_change = ((ihsg_latest / ihsg_prev_close) - 1) * 100
        ihsg_status = f"{ihsg_latest:,.2f} ({ihsg_change:+.2f}%)"

    tickers_to_process = [t for t in tickers_available if t != IHSG_TICKER]

    for ticker in tickers_to_process:
        try:
            # Ekstrak data untuk ticker ini
            if isinstance(full_data.columns, pd.MultiIndex):
                stock_df = full_data.loc[:, (slice(None), ticker)].droplevel(1, axis=1).copy()
            else:
                 # Jika hanya satu ticker yang diunduh (meski di multi-download), yfinance kadang meratakan kolom
                stock_df = full_data.copy()
                if stock_df.columns.name == 'Ticker' and stock_df.columns[0] != ticker:
                     continue # Lewati jika data ini bukan untuk ticker yang dimaksud

            # 1. Hitung Indikator (terutama return 30D & Volume Average)
            stock_df = calculate_indicators(stock_df, **indicator_params)
            
            # 2. Ambil data terbaru
            latest = stock_df.iloc[-1]
            yesterday = stock_df.iloc[-2] if len(stock_df) > 1 else latest

            current_close = latest['Close']
            yesterday_close = yesterday['Close']
            current_open = latest['Open']
            current_volume = latest['Volume']
            # avg_volume = latest.get('Avg_Volume', 1) # Not strictly needed for hard filter
            avg_volume = latest.get('Avg_Volume', 1) 
            return_30d = latest.get('Hist_Return_30D', -999) # Default return rendah jika N/A

            # --- Kriteria Screening ---
            
            # A. Open Price Ratio: open price > X * previous close (Customizable via criteria_params['open_ratio'])
            open_ratio_ok = current_open > (criteria_params['open_ratio'] * yesterday_close)
            
            # B. Min Price: Price > X (Customizable via criteria_params['min_price'])
            min_price_ok = current_close >= criteria_params['min_price']

            # C. Min Volume (Shares): volume > X shares (Customizable via criteria_params['min_volume_shares'])
            min_volume_ok = current_volume > criteria_params['min_volume_shares']
            
            # D. Min Relative Volume (Tambahan optional, bisa dicustom)
            # relative_volume_ok = current_volume > (1.5 * avg_volume)
            
            # --- Cek Kelayakan ---
            if open_ratio_ok and min_price_ok and min_volume_ok:
                
                # --- Ekstraksi Data untuk Hasil ---
                result = {
                    'Ticker': ticker,
                    'Harga Terakhir': f"{current_close:,.2f}",
                    'Gain 30D (%)': return_30d, # Ini akan digunakan untuk sorting
                    'Open/Close Ratio': f"{(current_open / yesterday_close):.3f}x",
                    'Open Hari Ini': f"{current_open:,.2f}",
                    'Close Kemarin': f"{yesterday_close:,.2f}",
                    'Volume Hari Ini': f"{current_volume:,.0f}",
                    'Avg Volume 20D': f"{avg_volume:,.0f}",
                    'RSI': f"{latest.get('RSI', np.nan):.2f}",
                }
                results.append(result)
                
                # Buat JSON Chart untuk ticker yang lolos
                chart_jsons[ticker] = create_plotly_json(stock_df, ticker, indicator_params)

        except Exception as e:
            print(f"Skipping {ticker} due to detailed analysis error: {e}")
            continue

    return results, chart_jsons, ihsg_status, ihsg_change

# --- Fungsi Plotting (Mengubah Plotly ke JSON) ---
# Dibiarkan sama seperti versi sebelumnya
def create_plotly_json(df, ticker, indicators):
    """Membuat chart harga (OHLC) dan indikator teknis menggunakan Plotly, diubah ke JSON."""
    
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Candlestick',
            increasing_line_color='#10b981', 
            decreasing_line_color='#ef4444' 
        )
    ])
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Short'], line=dict(color='#3b82f6', width=1), name=f'SMA {indicators["sma_short"]}'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Long'], line=dict(color='#f97316', width=1), name=f'SMA {indicators["sma_long"]}'))

    fig.update_layout(
        title=f'Chart Harga & Indikator: {ticker}',
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#facc15")
    )
    
    # Chart Volume
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='#6b7280'))
    fig_volume.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=0, b=20),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#facc15")
    )
    
    # Chart RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#a855f7', width=1)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981")
    fig_rsi.update_layout(
        title='RSI (Relative Strength Index)',
        height=150,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#facc15")
    )

    return {
        'price': json.dumps(fig, cls=PlotlyJSONEncoder),
        'volume': json.dumps(fig_volume, cls=PlotlyJSONEncoder),
        'rsi': json.dumps(fig_rsi, cls=PlotlyJSONEncoder)
    }

# --- FLASK APP Setup & HTML Template ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'jagoan_saham_secret' 

# Template HTML lengkap (diperbarui untuk Screener)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Screener Saham IDX (Flask)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: white; }
        .card { background-color: #1e293b; border-radius: 0.75rem; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .btn-submit { background-color: #facc15; color: #1f2937; font-weight: bold; border-radius: 0.5rem; transition: background-color 0.2s; }
        .btn-submit:hover { background-color: #ca8a04; }
        .table-header { background-color: #334155; }
        .table-row:nth-child(even) { background-color: #1f2937; }
        .table-row:nth-child(odd) { background-color: #1e293b; }
        .ihsg-positive { color: #10b981; }
        .ihsg-negative { color: #ef4444; }
        .positive-gain { color: #10b981; }
        .negative-gain { color: #ef4444; }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-7xl mx-auto">
        <h1 class="text-3xl font-bold mb-6 text-[#facc15]">Screener Saham IDX Kustom 🎯</h1>
        <p class="mb-6 text-gray-400">Filter cepat saham-saham berdasarkan kriteria harga dan volume, diurutkan berdasarkan Gain % Bulanan.</p>

        <!-- Form Konfigurasi Screening -->
        <div class="card mb-8">
            <form method="POST" action="/analyze">
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <!-- Kolom 1: Universe & Periode -->
                    <div class="col-span-1 md:col-span-2">
                        <h2 class="text-xl font-semibold mb-3 text-white">1. Stock Universe & Data</h2>
                        <label class="block text-gray-400 mb-1">Daftar Ticker (Pisahkan dengan koma/spasi/baris baru, contoh: BBCA.JK, TLKM.JK, ...)</label>
                        <textarea name="tickers_list" rows="5" class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white focus:ring-[#facc15] focus:border-[#facc15]" required>{{ input_tickers_str }}</textarea>
                        <p class="text-xs text-gray-500 mt-1">Menggunakan suffix **.JK** untuk saham IDX. Daftar di atas adalah simulasi saham aktif IDX.</p>
                        <div class="mt-4">
                            <label class="block text-gray-400 mb-1">Periode Data Historis</label>
                            <select name="period" class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white">
                                <option value="3mo" {% if period == '3mo' %}selected{% endif %}>3 Bulan</option>
                                <option value="6mo" {% if period == '6mo' %}selected{% endif %}>6 Bulan</option>
                                <option value="1y" {% if period == '1y' %}selected{% endif %}>1 Tahun</option>
                            </select>
                        </div>
                    </div>

                    <!-- Kolom 2 & 3: Kriteria Filter Keras -->
                    <div class="col-span-1 md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                            <h2 class="text-xl font-semibold mb-3 text-white">2. Kriteria Harga & Volume (Dapat Disesuaikan)</h2>
                            
                            <label class="block text-gray-400 mt-2 mb-1">Min Open Ratio (Open / Prev Close > X)</label>
                            <input type="number" name="open_ratio" value="{{ criteria_params.open_ratio }}" step="0.001" required
                                   class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white focus:ring-[#facc15] focus:border-[#facc15]">
                            <p class="text-xs text-gray-500">Contoh: **1.015** (Open > 1.5% dari Close kemarin)</p>

                            <label class="block text-gray-400 mt-4 mb-1">Min Harga Saham (Price >= X)</label>
                            <input type="number" name="min_price" value="{{ criteria_params.min_price }}" step="1" required
                                   class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white focus:ring-[#facc15] focus:border-[#facc15]">
                            <p class="text-xs text-gray-500">Contoh: **50** (Harga minimal)</p>

                            <label class="block text-gray-400 mt-4 mb-1">Min Volume Harian (Shares, X >)</label>
                            <input type="number" name="min_volume_shares" value="{{ criteria_params.min_volume_shares }}" step="100000" required
                                   class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white focus:ring-[#facc15] focus:border-[#facc15]">
                            <p class="text-xs text-red-400">Perhatian: Menggunakan Volume SHARES. Contoh: **5000000** (5 juta saham).</p>
                        </div>

                        <!-- Kolom 4: Parameter Indikator Lanjutan (untuk Chart) -->
                        <div>
                             <h2 class="text-xl font-semibold mb-3 text-white">3. Indikator Chart (Opsional)</h2>
                             {% for label, name, value in indicator_inputs %}
                                 <label class="block text-gray-400 mt-2 mb-1">{{ label }}</label>
                                 <input type="number" name="{{ name }}" value="{{ value }}" step="1" required
                                        class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white focus:ring-[#facc15] focus:border-[#facc15]">
                             {% endfor %}
                        </div>
                    </div>
                </div>

                <button type="submit" class="btn-submit w-full p-3 mt-6 text-lg">
                    ▶️ RUN SCREENER & SORT (By 30D Gain)
                </button>
            </form>
        </div>

        <!-- Area Hasil Analisis -->
        {% if results %}
        <div class="card mb-8">
            <h2 class="text-2xl font-bold mb-4 text-[#facc15]">Hasil Screening ({{ results | length }} Saham Lolos)</h2>

            <!-- IHSG Status -->
            <div class="mb-6 p-4 rounded-lg border-l-4 
                {% if ihsg_change > 0 %}border-[#10b981] bg-[#1a2e38]{% elif ihsg_change < 0 %}border-[#ef4444] bg-[#3e2723]{% else %}border-gray-500 bg-gray-700{% endif %}">
                <span class="text-lg font-semibold block mb-1">Status IHSG</span>
                <span class="text-xl font-bold {% if ihsg_change > 0 %}text-[#10b981]{% elif ihsg_change < 0 %}text-[#ef4444]{% else %}text-gray-400{% endif %}">
                    {{ ihsg_status }}
                </span>
            </div>

            <!-- Tabel Hasil -->
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead>
                        <tr class="table-header text-xs uppercase tracking-wider">
                            <th class="p-3">#</th>
                            <th class="p-3">Ticker</th>
                            <th class="p-3">Gain 30D (Sort)</th>
                            <th class="p-3">Harga Terakhir</th>
                            <th class="p-3">Open/Close Ratio</th>
                            <th class="p-3">Volume Hari Ini</th>
                            <th class="p-3">Open Hari Ini</th>
                            <th class="p-3">RSI (14)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in results %}
                        <tr class="table-row">
                            <td class="p-3">{{ loop.index }}</td>
                            <td class="p-3 font-semibold">{{ row.Ticker }}</td>
                            <td class="p-3 font-bold 
                                {% if row['Gain 30D (%)'] | float > 0 %}positive-gain{% elif row['Gain 30D (%)'] | float < 0 %}negative-gain{% endif %}">
                                {{ row['Gain 30D (%)'] | round(2) }}%
                            </td>
                            <td class="p-3">{{ row['Harga Terakhir'] }}</td>
                            <td class="p-3">{{ row['Open/Close Ratio'] }}</td>
                            <td class="p-3">{{ row['Volume Hari Ini'] }}</td>
                            <td class="p-3">{{ row['Open Hari Ini'] }}</td>
                            <td class="p-3">{{ row.RSI }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <h2 class="text-2xl font-bold my-6 text-[#facc15]">Chart Harga & Indikator</h2>
            <select id="chart-selector" onchange="updateChart()" class="mb-4 p-2 rounded bg-gray-700 border border-gray-600 text-white">
                {% for row in results %}
                    <option value="{{ row.Ticker }}">{{ row.Ticker }}</option>
                {% endfor %}
            </select>
            
            <div id="plotly-chart-price" class="mb-4"></div>
            <div id="plotly-chart-volume" class="mb-4"></div>
            <div id="plotly-chart-rsi"></div>
        </div>

        <script>
            const chartData = {{ chart_jsons | safe }};
            const defaultTicker = document.getElementById('chart-selector').value;

            function updateChart() {
                const selectedTicker = document.getElementById('chart-selector').value;
                const data = chartData[selectedTicker];
                if (data) {
                    Plotly.react('plotly-chart-price', JSON.parse(data.price).data, JSON.parse(data.price).layout, {responsive: true});
                    Plotly.react('plotly-chart-volume', JSON.parse(data.volume).data, JSON.parse(data.volume).layout, {responsive: true});
                    Plotly.react('plotly-chart-rsi', JSON.parse(data.rsi).data, JSON.parse(data.rsi).layout, {responsive: true});
                }
            }
            
            // Render chart saat pertama kali dimuat
            if (chartData[defaultTicker]) {
                updateChart();
            }
        </script>
        {% elif error_message %}
            <div class="card bg-red-800 text-white p-4">
                <p class="font-bold">⚠️ Error Analisis:</p>
                <p>{{ error_message }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Halaman utama dengan form default."""
    # Default parameters for a strong gap-up filter
    default_criteria = {
        'open_ratio': 1.015,         # Open > 1.015 * Prev Close
        'min_price': 50,             # Price > 50
        'min_volume_shares': 5000000, # Volume > 5,000,000 shares
    }

    # Parameters for Chart Indicators (default)
    default_indicators = {
        'sma_short': 20, 'sma_long': 60, 'rsi_period': 14, 'vol_period': 20, 'hist_days': 30
    }
    
    indicator_inputs = [
        ("SMA Pendek (Hari)", "sma_short", default_indicators['sma_short']),
        ("SMA Panjang (Hari)", "sma_long", default_indicators['sma_long']),
        ("RSI Periode", "rsi_period", default_indicators['rsi_period']),
    ]
    
    # Concatenate default list to string for textarea
    input_tickers_str = ", ".join(DEFAULT_IDX_TICKERS)

    return render_template_string(
        HTML_TEMPLATE,
        input_tickers_str=input_tickers_str,
        ihsg_ticker=IHSG_TICKER,
        criteria_params=default_criteria,
        indicator_inputs=indicator_inputs,
        period='6mo',
        results=None,
        error_message=None
    )


@app.route('/analyze', methods=['POST'])
def analyze():
    """Route untuk menjalankan screening berdasarkan input form."""
    selected_tickers_list = []
    period = '6mo'
    try:
        form_data = request.form
        
        # --- 1. Ambil Parameter dari Form ---
        
        # Ambil dan bersihkan daftar ticker dari textarea
        tickers_str = form_data.get('tickers_list', "").replace(',', ' ').replace('\n', ' ').replace('\r', ' ')
        selected_tickers_list = [t.strip().upper() for t in tickers_str.split() if t.strip()]
        period = form_data.get('period', '6mo')

        # Parameter Kriteria Filter (Diambil langsung dari input form)
        criteria_params = {
            'open_ratio': float(form_data.get('open_ratio', 1.015)),
            'min_price': int(form_data.get('min_price', 50)),
            'min_volume_shares': int(form_data.get('min_volume_shares', 5000000)),
        }
        
        # Parameter Indikator Chart
        indicator_params = {
            'sma_short': int(form_data.get('sma_short', 20)),
            'sma_long': int(form_data.get('sma_long', 60)),
            'rsi_period': int(form_data.get('rsi_period', 14)),
            'vol_period': 20, # Fixed at 20 for Vol Avg
            'hist_days': 30,  # Fixed at 30 for sorting
        }

        if not selected_tickers_list:
            raise ValueError("Harap masukkan minimal satu ticker saham (contoh: BBCA.JK) untuk di-screen.")

        all_data_tickers = list(set(selected_tickers_list + [IHSG_TICKER]))
        
        # --- 2. Ambil Data ---
        full_data, ihsg_data = load_historical_data(all_data_tickers, period=period)
        
        # --- 3. Running Screener ---
        
        # Passing: full_data (MultiIndex DF), ihsg_data (Single Index DF), criteria_params, indicator_params
        results, chart_jsons, ihsg_status, ihsg_change = screen_and_extract(
            full_data, ihsg_data, criteria_params, indicator_params
        )

        if not results:
             raise ValueError("Tidak ada saham yang lolos kriteria screening yang Anda tentukan.")
             
        results_df = pd.DataFrame(results)
        
        # Terapkan Sorting: Diurutkan berdasarkan Gain 30D (%) secara DESCENDING
        results_df['Gain 30D (%)'] = pd.to_numeric(results_df['Gain 30D (%)'], errors='coerce')
        results_df = results_df.sort_values(by=['Gain 30D (%)'], ascending=False).dropna(subset=['Gain 30D (%)'])
        
        # Konversi DataFrame ke list of dicts untuk render_template_string
        final_results = results_df.to_dict('records')
        
        # --- 4. Render Hasil ---
        indicator_inputs = [
            ("SMA Pendek (Hari)", "sma_short", indicator_params['sma_short']),
            ("SMA Panjang (Hari)", "sma_long", indicator_params['sma_long']),
            ("RSI Periode", "rsi_period", indicator_params['rsi_period']),
        ]
        
        return render_template_string(
            HTML_TEMPLATE,
            input_tickers_str=", ".join(selected_tickers_list),
            ihsg_ticker=IHSG_TICKER,
            criteria_params=criteria_params,
            indicator_inputs=indicator_inputs,
            period=period,
            results=final_results,
            chart_jsons=chart_jsons,
            ihsg_status=ihsg_status,
            ihsg_change=ihsg_change,
            error_message=None
        )

    except ValueError as e:
        # Re-render form dengan error message
        # Pastikan kita menggunakan nilai yang dikirimkan user jika ada
        default_criteria = {
            'open_ratio': float(form_data.get('open_ratio', 1.015)) if 'form_data' in locals() else 1.015,
            'min_price': int(form_data.get('min_price', 50)) if 'form_data' in locals() else 50,
            'min_volume_shares': int(form_data.get('min_volume_shares', 5000000)) if 'form_data' in locals() else 5000000,
        }
        default_indicators = {'sma_short': 20, 'sma_long': 60, 'rsi_period': 14}
        
        indicator_inputs = [
            ("SMA Pendek (Hari)", "sma_short", default_indicators['sma_short']),
            ("SMA Panjang (Hari)", "sma_long", default_indicators['sma_long']),
            ("RSI Periode", "rsi_period", default_indicators['rsi_period']),
        ]
        
        return render_template_string(HTML_TEMPLATE, 
                                      input_tickers_str=", ".join(selected_tickers_list) if selected_tickers_list else ", ".join(DEFAULT_IDX_TICKERS),
                                      ihsg_ticker=IHSG_TICKER,
                                      criteria_params=default_criteria,
                                      indicator_inputs=indicator_inputs,
                                      period=period,
                                      results=None,
                                      error_message=str(e))
    except Exception as e:
        # Re-render form dengan error message umum
        default_criteria = {
            'open_ratio': float(form_data.get('open_ratio', 1.015)) if 'form_data' in locals() else 1.015,
            'min_price': int(form_data.get('min_price', 50)) if 'form_data' in locals() else 50,
            'min_volume_shares': int(form_data.get('min_volume_shares', 5000000)) if 'form_data' in locals() else 5000000,
        }
        default_indicators = {'sma_short': 20, 'sma_long': 60, 'rsi_period': 14}
        
        indicator_inputs = [
            ("SMA Pendek (Hari)", "sma_short", default_indicators['sma_short']),
            ("SMA Panjang (Hari)", "sma_long", default_indicators['sma_long']),
            ("RSI Periode", "rsi_period", default_indicators['rsi_period']),
        ]
        
        return render_template_string(HTML_TEMPLATE, 
                                      input_tickers_str=", ".join(selected_tickers_list) if selected_tickers_list else ", ".join(DEFAULT_IDX_TICKERS),
                                      ihsg_ticker=IHSG_TICKER,
                                      criteria_params=default_criteria,
                                      indicator_inputs=indicator_inputs,
                                      period=period,
                                      results=None,
                                      error_message=f"Terjadi kesalahan tak terduga saat memproses data: {e}. Cek kembali koneksi internet dan daftar ticker.")


if __name__ == '__main__':
    # Untuk local development, gunakan mode debug
    app.run(debug=True)
    
