import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

# Load patterns from JSON file
def load_patterns():
    with open("c:\\Projects\\Patterns\\Assets\\chart_patterns_dataset.json", "r") as file:
        return json.load(file)

# Function to fetch OHLCV data
def fetch_data(symbol, timeframe):
    try:
        # Map timeframes to yfinance intervals and periods
        timeframe_map = {
            '4h': ('1h', '60d'),   # Get 60 days of hourly data for 4h resampling
            '6h': ('1h', '60d'),   # Get 60 days of hourly data for 6h resampling
            '12h': ('1h', '60d'),  # Get 60 days of hourly data for 12h resampling
            '1d': ('1d', '1y'),    # Get 1 year of daily data
            '1wk': ('1wk', '2y'),  # Get 2 years of weekly data
            '1mo': ('1mo', '5y')   # Get 5 years of monthly data
        }

        interval, period = timeframe_map[timeframe]
        
        # Download data with appropriate interval and period
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        if data.empty:
            st.error(f"No data found for the symbol: {symbol}")
            return None
          
        # For intraday timeframes, resample the hourly data
        if timeframe in ['4h', '6h', '12h']:
            timeframe_hours = int(timeframe.replace('h', ''))
            data = data.resample(f'{timeframe_hours}H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        data.reset_index(inplace=True)
        # Rename 'Datetime' or 'Date' column to 'Date'
        date_column = 'Datetime' if 'Datetime' in data.columns else 'Date'
        data.rename(columns={date_column: 'Date'}, inplace=True)
        
        # Convert dates for candlestick chart
        data['Date'] = data['Date'].apply(mdates.date2num)
        
        # Ensure we have enough data for pattern detection
        if len(data) < 50:
            st.warning(f"Insufficient data points for reliable pattern detection. Got {len(data)} points, need at least 50.")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to detect patterns based on JSON rules
def detect_patterns(data, patterns):
    detected_patterns = []
    if "High" not in data.columns or "Low" not in data.columns:
        st.error("The data does not contain the required 'High' and 'Low' columns.")
        return detected_patterns

    for pattern in patterns:
        try:
            window_size = 50  # Increased window size for better pattern detection
            if len(data) > window_size:
                # Get the last window_size periods of data
                recent_data = data.iloc[-window_size:]
                
                # Calculate average volume
                avg_volume = recent_data['Volume'].mean().iloc[0] if not recent_data.empty else 0
                
                if pattern["pattern"] == "Head and Shoulders":
                    # Check for head and shoulders pattern
                    highs = recent_data['High'].values
                    if len(highs) >= 30:
                        left_shoulder = max(highs[-30:-20])
                        head = max(highs[-20:-10])
                        right_shoulder = max(highs[-10:])
                        # Head should be higher than shoulders
                        # Shoulders should be roughly equal height
                        if (head > left_shoulder and head > right_shoulder and 
                            abs(left_shoulder - right_shoulder) < 0.1 * head):
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Head & Shoulders (Inverted)":
                    # Check for inverted head and shoulders pattern
                    lows = recent_data['Low'].values
                    if len(lows) >= 30:
                        left_shoulder = min(lows[-30:-20])
                        head = min(lows[-20:-10])
                        right_shoulder = min(lows[-10:])
                        # Head should be lower than shoulders
                        # Shoulders should be roughly equal height
                        if (head < left_shoulder and head < right_shoulder and 
                            abs(left_shoulder - right_shoulder) < 0.1 * abs(head)):
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Double Bottom":
                    # Check for double bottom pattern
                    lows = recent_data['Low'].values
                    closes = recent_data['Close'].values
                    if len(lows) >= 20:
                        first_bottom = min(lows[-20:-10])
                        second_bottom = min(lows[-10:])                        # Bottoms should be roughly equal
                        # Price should show recovery after second bottom
                        if (abs(first_bottom - second_bottom) < 0.02 * first_bottom and
                            closes[-1] > second_bottom):
                            detected_patterns.append(pattern)
                            
                elif pattern["pattern"] == "Cup and Handle":
                    # Check for cup and handle pattern
                    closes = recent_data['Close'].values
                    volumes = recent_data['Volume'].values
                    if len(closes) >= 40:
                        # Split the window into cup (30 periods) and handle (10 periods)
                        cup_data = closes[-40:-10]
                        handle_data = closes[-10:]
                        
                        # Check for U-shaped cup
                        left_high = cup_data[0]
                        cup_low = min(cup_data)
                        right_high = cup_data[-1]
                        
                        # Calculate cup depth and symmetry
                        cup_depth = (left_high - cup_low) / left_high
                        cup_symmetry = abs(left_high - right_high) / left_high
                        
                        # Check handle characteristics
                        handle_high = max(handle_data)
                        handle_low = min(handle_data)
                        handle_depth = (handle_high - handle_low) / handle_high
                        
                        # Volume pattern check
                        vol_left = np.mean(volumes[-40:-30])  # Volume at left lip
                        vol_bottom = np.mean(volumes[-30:-20]) # Volume at cup bottom
                        vol_right = np.mean(volumes[-20:-10])  # Volume at right lip
                        
                        # Pattern conditions:
                        # 1. Cup is U-shaped (measured by depth and symmetry)
                        # 2. Handle droops downward but not too deep
                        # 3. Volume follows the expected pattern
                        if (0.1 < cup_depth < 0.5 and  # Cup should be well-defined but not too deep
                            cup_symmetry < 0.1 and      # Left and right highs should be similar
                            handle_depth < cup_depth * 0.5 and  # Handle shouldn't be too deep
                            vol_bottom < vol_left and   # Volume diminishes in cup bottom
                            vol_right > vol_bottom):    # Volume increases toward right lip
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Triple Bottoms":
                    # Check for triple bottom pattern
                    lows = recent_data['Low'].values
                    if len(lows) >= 30:
                        bottom1 = min(lows[-30:-20])
                        bottom2 = min(lows[-20:-10])
                        bottom3 = min(lows[-10:])
                        # All bottoms should be roughly equal
                        if (abs(bottom1 - bottom2) < 0.02 * bottom1 and 
                            abs(bottom2 - bottom3) < 0.02 * bottom2):
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Triple Tops":
                    # Check for triple top pattern
                    highs = recent_data['High'].values
                    if len(highs) >= 30:
                        top1 = max(highs[-30:-20])
                        top2 = max(highs[-20:-10])
                        top3 = max(highs[-10:])
                        # All tops should be roughly equal
                        if (abs(top1 - top2) < 0.02 * top1 and 
                            abs(top2 - top3) < 0.02 * top2):
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Flags (Bullish)":
                    # Check for bullish flag pattern
                    closes = recent_data['Close'].values
                    if len(closes) >= 20:
                        # Check for prior uptrend
                        uptrend = all(closes[i] <= closes[i+1] for i in range(len(closes)-20, len(closes)-10))
                        # Check for flag (short-term pullback)
                        consolidation = all(abs(closes[i] - closes[i+1]) < 0.01 * closes[i] 
                                         for i in range(len(closes)-10, len(closes)-1))
                        if uptrend and consolidation:
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Triangles (Ascending)":
                    # Check for ascending triangle pattern
                    highs = recent_data['High'].values
                    lows = recent_data['Low'].values
                    if len(highs) >= 20:
                        # Check for flat resistance and rising support
                        resistance_line = max(highs[-20:])
                        support_slope = (lows[-1] - lows[-20]) / 19
                        if support_slope > 0 and all(h <= resistance_line * 1.02 for h in highs[-5:]):
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Triangles (Descending)":
                    # Check for descending triangle pattern
                    highs = recent_data['High'].values
                    lows = recent_data['Low'].values
                    if len(highs) >= 20:
                        # Check for flat support and falling resistance
                        support_line = min(lows[-20:])
                        resistance_slope = (highs[-1] - highs[-20]) / 19
                        if resistance_slope < 0 and all(l >= support_line * 0.98 for l in lows[-5:]):
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Triangles (Symmetrical)":
                    # Check for symmetrical triangle pattern
                    highs = recent_data['High'].values
                    lows = recent_data['Low'].values
                    if len(highs) >= 20:
                        # Check for converging trendlines
                        high_slope = (highs[-1] - highs[-20]) / 19
                        low_slope = (lows[-1] - lows[-20]) / 19
                        if high_slope < 0 and low_slope > 0:
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Wedges (Falling)":
                    # Check for falling wedge pattern
                    highs = recent_data['High'].values
                    lows = recent_data['Low'].values
                    if len(highs) >= 20:
                        # Both trendlines falling, but support falling slower than resistance
                        high_slope = (highs[-1] - highs[-20]) / 19
                        low_slope = (lows[-1] - lows[-20]) / 19
                        if high_slope < low_slope < 0:
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Wedges (Rising)":
                    # Check for rising wedge pattern
                    highs = recent_data['High'].values
                    lows = recent_data['Low'].values
                    if len(highs) >= 20:
                        # Both trendlines rising, but resistance rising faster than support
                        high_slope = (highs[-1] - highs[-20]) / 19
                        low_slope = (lows[-1] - lows[-20]) / 19
                        if 0 < low_slope < high_slope:
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Rounding Bottoms":
                    # Check for rounding bottom pattern
                    closes = recent_data['Close'].values
                    if len(closes) >= 30:
                        # Calculate shape similarity to a parabola
                        x = np.array(range(len(closes[-30:])))
                        y = closes[-30:]
                        z = np.polyfit(x, y, 2)
                        if z[0] > 0:  # Check if parabola opens upward
                            detected_patterns.append(pattern)

                elif pattern["pattern"] == "Rounding Tops":
                    # Check for rounding top pattern
                    closes = recent_data['Close'].values
                    if len(closes) >= 30:
                        # Calculate shape similarity to an inverted parabola
                        x = np.array(range(len(closes[-30:])))
                        y = closes[-30:]
                        z = np.polyfit(x, y, 2)
                        if z[0] < 0:  # Check if parabola opens downward
                            detected_patterns.append(pattern)

        except Exception as e:
            st.warning(f"Error detecting {pattern.get('pattern', 'unknown pattern')}: {str(e)}")
            continue
    
    return detected_patterns

# Function to suggest trades based on detected patterns
def suggest_trade(data: pd.DataFrame, detected_patterns: list) -> dict | None:
    """
    Generate trade suggestions based on detected chart patterns.
    
    Args:
        data (pd.DataFrame): OHLCV price data with required 'Close' column
        detected_patterns (list): List of detected chart patterns with their properties
        
    Returns:
        dict | None: Trade suggestion containing pattern name, entry price, stop loss,
                    target price, and confidence score. Returns None if no patterns detected.
    """
    if not isinstance(data, pd.DataFrame) or "Close" not in data.columns:
        st.error("Invalid price data format")
        return None
        
    if not detected_patterns or not isinstance(detected_patterns, list):
        return None

    try:
        pattern = detected_patterns[0]
        
        # Get the latest close price and convert to float to remove Series metadata
        entry_price = float(data["Close"].values[-1])
        
        # Calculate risk-reward based on pattern direction
        if pattern.get("directional_bias") == "Bullish":
            stop_loss = float(entry_price * 0.95)  # 5% below entry
            target_price = float(entry_price * 1.10)  # 10% above entry
        else:
            stop_loss = float(entry_price * 1.05)  # 5% above entry
            target_price = float(entry_price * 0.90)  # 10% below entry
          # Calculate dynamic confidence score based on multiple factors
        confidence_factors = []
        
        # 1. Volume confirmation (higher volume increases confidence)
        avg_volume = np.mean(data['Volume'].values[-10:])  # Last 10 periods
        prev_avg_volume = np.mean(data['Volume'].values[-20:-10])  # Previous 10 periods
        volume_factor = min(1.0, avg_volume / prev_avg_volume if prev_avg_volume > 0 else 0.5)
        confidence_factors.append(volume_factor)
        
        # 2. Price momentum alignment with pattern
        price_change = (entry_price - data['Close'].values[-5]) / data['Close'].values[-5]
        momentum_factor = 0.7
        if (pattern.get("directional_bias") == "Bullish" and price_change > 0) or \
           (pattern.get("directional_bias") == "Bearish" and price_change < 0):
            momentum_factor = 0.9
        confidence_factors.append(momentum_factor)
        
        # 3. Pattern completion factor
        if pattern["pattern"] in ["Head and Shoulders", "Triple Tops", "Triple Bottoms"]:
            completion_factor = 0.95  # Complex patterns that are fully formed are more reliable
        else:
            completion_factor = 0.85
        confidence_factors.append(completion_factor)
        
        # 4. Market trend alignment
        trend_direction = np.mean(data['Close'].values[-20:] - data['Close'].values[-21:-1])
        trend_factor = 0.7
        if (pattern.get("directional_bias") == "Bullish" and trend_direction > 0) or \
           (pattern.get("directional_bias") == "Bearish" and trend_direction < 0):
            trend_factor = 0.9
        confidence_factors.append(trend_factor)
          # Calculate weighted confidence score
        confidence_score = round(sum(confidence_factors) / len(confidence_factors), 2)
        
        # Determine qualitative confidence level and trade type
        if confidence_score >= 0.85:
            confidence_level = "High"
            trade_type = "Buy" if pattern.get("directional_bias") == "Bullish" else "Sell"
        elif confidence_score >= 0.75:
            confidence_level = "Medium"
            trade_type = "Buy" if pattern.get("directional_bias") == "Bullish" else "Sell"
        else:
            confidence_level = "Low"
            trade_type = "No Trade"  # Don't trade when confidence is low
            
        # Determine trade duration based on pattern type and market conditions
        if pattern["pattern"] in ["Cup and Handle", "Rounding Bottoms", "Rounding Tops", 
                                "Head and Shoulders", "Head & Shoulders (Inverted)"]:
            trade_duration = "Long-term"  # These patterns typically play out over longer periods
        elif pattern["pattern"] in ["Flags (Bullish)", "Triangles (Ascending)", 
                                  "Triangles (Descending)", "Triangles (Symmetrical)"]:
            trade_duration = "Medium-term"  # These patterns typically resolve in medium timeframe
        else:
            trade_duration = "Short-term"  # Other patterns tend to have shorter duration
            
        # Adjust duration based on timeframe and trend strength
        if abs(trend_direction) > 0.02:  # Strong trend
            trade_duration = "Long-term" if trade_duration != "Short-term" else "Medium-term"
        
        # Determine trade type based on pattern bias and confidence
        if confidence_score < 0.75:
            trade_type = "No Trade"  # Don't trade if confidence is too low
        else:
            trade_type = "Buy" if pattern.get("directional_bias") == "Bullish" else "Sell"

        # Prepare the return dictionary with all calculations
        return {
            "pattern_name": pattern["pattern"],
            "trade_type": trade_type,
            "entry_price": round(entry_price, 4),
            "stop_loss": round(stop_loss, 4),
            "target_price": round(target_price, 4),
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "trade_duration": trade_duration,
            "risk_reward_ratio": round(abs(target_price - entry_price) / abs(stop_loss - entry_price), 2),
            "confidence_details": {
                "volume_confirmation": round(volume_factor, 2),
                "momentum_alignment": round(momentum_factor, 2),
                "pattern_completion": round(completion_factor, 2),
                "trend_alignment": round(trend_factor, 2)
            }
        }
    except Exception as e:
        st.error(f"Error generating trade suggestion: {str(e)}")
        return None

# Function to plot candlestick chart
def plot_candlestick(data):
    fig, ax = plt.subplots()
    ohlc = data[["Date", "Open", "High", "Low", "Close"]].values
    candlestick_ohlc(ax, ohlc, width=0.6, colorup="green", colordown="red")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.title("Candlestick Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(fig)

# Streamlit UI
st.title("📊 Forex and Stock Market Pattern Analyzer")

# Load patterns from JSON
patterns = load_patterns()

# Predefined list of symbols for autocomplete
predefined_symbols = ["AAPL", "MSFT", "GOOGL", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]

# Upload chart or analyze symbol
option = st.selectbox("Choose an option", ["Analyze Symbol", "Upload Excel"])

if option == "Analyze Symbol":
    symbol = st.text_input("Enter symbol (e.g., AAPL, EURUSD=X):", value="", help="Start typing to see suggestions")
    timeframe = st.selectbox("Select timeframe", ["1d", "1wk", "1mo", "4h", "6h", "12h"])
    if st.button("Analyze"):
        data = fetch_data(symbol, timeframe)
        if data is not None:
            st.success("✅ Data fetched successfully!")
            plot_candlestick(data)  # Display candlestick chart
            detected_patterns = detect_patterns(data, patterns)
            if detected_patterns:
                st.subheader("📌 Detected Patterns")
                for pattern in detected_patterns:
                    st.markdown(f"**{pattern['pattern']}**: {pattern['description']}")
            else:
                st.info("No patterns detected.")
            trade_suggestion = suggest_trade(data, detected_patterns)
            if trade_suggestion:
                st.subheader("💡 Trade Suggestion")
                
                # Create three columns
                col1, col2, col3 = st.columns([2, 2, 1])
                
                # Column 1: Basic trade information
                with col1:
                    st.write("**Trade Details:**")
                    # Color code the trade type
                    trade_type = trade_suggestion['trade_type']
                    trade_color = {
                        "Buy": "green",
                        "Sell": "red",
                        "No Trade": "orange"
                    }[trade_type]
                    st.markdown(f"**Trade Action:** :{trade_color}[{trade_type}]")
                    st.write(f"Pattern: {trade_suggestion['pattern_name']}")
                    st.write(f"Entry Price: {trade_suggestion['entry_price']}")
                    st.write(f"Stop Loss: {trade_suggestion['stop_loss']}")
                    st.write(f"Target Price: {trade_suggestion['target_price']}")
                    st.write(f"Risk/Reward: {trade_suggestion['risk_reward_ratio']}")
                
                # Column 2: Confidence analysis
                with col2:
                    st.write("**Confidence Analysis:**")
                    details = trade_suggestion['confidence_details']
                    st.write("Contributing Factors:")
                    st.write(f"• Volume: {details['volume_confirmation']}")
                    st.write(f"• Momentum: {details['momentum_alignment']}")
                    st.write(f"• Pattern: {details['pattern_completion']}")
                    st.write(f"• Trend: {details['trend_alignment']}")
                
                # Column 3: Trade summary
                with col3:
                    st.write("**Trade Summary:**")
                    # Color-coded confidence level
                    confidence_color = {
                        "High": "green",
                        "Medium": "orange",
                        "Low": "red"
                    }[trade_suggestion['confidence_level']]
                    st.markdown(f"Confidence: :{confidence_color}[{trade_suggestion['confidence_level']}]")
                    # Trade duration with emoji
                    duration_emoji = {
                        "Long-term": "📈",
                        "Medium-term": "⚡",
                        "Short-term": "⚡⚡"
                    }[trade_suggestion['trade_duration']]
                    st.write(f"Duration: {duration_emoji} {trade_suggestion['trade_duration']}")

elif option == "Upload Excel":
    uploaded_file = st.file_uploader("Upload an Excel file with symbols", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded Symbols:")
        st.dataframe(df)
        timeframes = st.multiselect("Select timeframes to analyze", ["1d", "1wk", "1mo", "4h", "6h", "12h"], default=["1d"])
        if st.button("Analyze All Symbols"):
            for symbol in df["Symbol"]:
                st.subheader(f"Analyzing {symbol}")
                for timeframe in timeframes:
                    st.write(f"Timeframe: {timeframe}")
                    data = fetch_data(symbol, timeframe)
                    if data is not None:
                        plot_candlestick(data)
                        detected_patterns = detect_patterns(data, patterns)
                        if detected_patterns:
                            for pattern in detected_patterns:
                                st.markdown(f"**{pattern['pattern']}**: {pattern['description']}")
                        else:
                            st.info("No patterns detected.")
