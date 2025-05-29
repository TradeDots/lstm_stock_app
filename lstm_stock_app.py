import streamlit as st
import yfinance as yf
from curl_cffi import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="LSTM Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 LSTM Stock Price Predictor by TradeDots</h1>', unsafe_allow_html=True)
st.markdown("**Predict stock prices using advanced AI neural networks**")

# Sidebar for inputs
st.sidebar.header("📊 Configuration")

# Ticker input
ticker = st.sidebar.text_input(
    "Stock Ticker Symbol", 
    value="AAPL", 
    help="Enter any valid stock ticker (e.g., AAPL, GOOGL, TSLA)"
).upper()

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=1825),  # 5 years ago
        help="Start date for historical data"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        help="End date for historical data"
    )

# Model parameters
st.sidebar.subheader("🔧 Model Parameters")
sequence_length = st.sidebar.slider("Sequence Length (days)", 30, 120, 60, help="Number of previous days to use for prediction")
epochs = st.sidebar.slider("Training Epochs", 20, 100, 50, help="Number of training iterations")
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

# Prediction horizon
prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 5, help="Number of future days to predict")

# Functions
@st.cache_data
def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        session = requests.Session(impersonate="chrome")
        stock = yf.Ticker(ticker, session=session)
        
        # Try to get info first to verify the ticker is valid
        try:
            info = stock.info
            if not info:
                st.error(f"Could not get info for {ticker}")
                return None, None
        except Exception as e:
            st.error(f"Error getting info for {ticker}: {str(e)}")
            return None, None
        
        # Download the data
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error(f"No data returned for {ticker}")
            return None, None
        
        # Get company info
        company_name = info.get('longName', ticker)
        
        return data, company_name
    except Exception as e:
        return None, None

def prepare_data(data, sequence_length):
    """Prepare data for LSTM training - improved version"""
    # Ensure we have enough data
    if len(data) < sequence_length + 1:
        raise ValueError(f"Not enough data. Need at least {sequence_length + 1} days, got {len(data)}")
    
    prices = data['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Split data - ensure we have enough data for both sets
    split_point = max(int(len(X) * 0.8), 1)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    st.write(f"Data preparation: {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test, scaler, scaled_data

def build_lstm_model(sequence_length):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def calculate_metrics(actual, predicted):
    """Calculate prediction metrics"""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def predict_future_prices(model, last_sequence, scaler, days):
    """Predict future prices"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Predict next day
        next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # Convert back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Main app logic
if st.sidebar.button("🚀 Run Prediction", type="primary"):
    if ticker:
        # Download data
        with st.spinner(f"📈 Downloading data for {ticker}..."):
            data, company_name = download_stock_data(ticker, start_date, end_date)
        
        if data is not None and not data.empty and 'Close' in data.columns:
            st.success(f"✅ Successfully downloaded {len(data)} days of data for {company_name}")
            
            # Quick Data Test
            st.write("**Quick Data Test:**")
            
            try:
                # Method 1: Streamlit's native line chart (simplest)
                st.write(f"**{company_name} Stock Price Over Time**")
                st.line_chart(data['Close'])
                
            except Exception as e:
                # Method 2: Matplotlib backup
                try:
                    st.write(f"**{company_name} Stock Price Over Time**")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(data.index, data['Close'], color='blue', linewidth=2)
                    ax.set_title('Stock Price Chart (Matplotlib)')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price ($)')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e2:
                    # Method 3: Simple table view as last resort
                    st.write(f"**{company_name} Stock Price Over Time**")
                    chart_data = pd.DataFrame({
                        'Date': data.index,
                        'Close Price': data['Close']
                    })
                    st.dataframe(chart_data.tail(20))  # Show last 20 days
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if len(data) > 0:
                    current_price = float(data['Close'].iloc[-1])
                    st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                if len(data) > 1:
                    current_price = float(data['Close'].iloc[-1])
                    prev_price = float(data['Close'].iloc[-2])
                    daily_change = current_price - prev_price
                    st.metric("Daily Change", f"${daily_change:.2f}", f"{(daily_change/prev_price*100):+.2f}%")
            with col3:
                high_price = float(data['Close'].max())
                st.metric("Period High", f"${high_price:.2f}")
            with col4:
                low_price = float(data['Close'].min())
                st.metric("Period Low", f"${low_price:.2f}")
            
            # Prepare data
            with st.spinner("🔧 Preparing data for training..."):
                X_train, X_test, y_train, y_test, scaler, scaled_data = prepare_data(data, sequence_length)
            
            st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
            
            # Build and train model
            with st.spinner("🧠 Building and training LSTM model..."):
                model = build_lstm_model(sequence_length)
                
                # Add progress bar for training
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Custom callback to update progress
                class StreamlitCallback:
                    def __init__(self, progress_bar, status_text, total_epochs):
                        self.progress_bar = progress_bar
                        self.status_text = status_text
                        self.total_epochs = total_epochs
                    
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / self.total_epochs
                        self.progress_bar.progress(progress)
                        self.status_text.text(f"Training epoch {epoch + 1}/{self.total_epochs} - Loss: {logs.get('loss', 0):.4f}")
                
                # Train model
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                
                history = model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    verbose=0,
                    callbacks=[early_stopping]
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Training completed!")
            
            # Make predictions
            with st.spinner("🔮 Making predictions..."):
                train_predictions = model.predict(X_train, verbose=0)
                test_predictions = model.predict(X_test, verbose=0)
                
                # Convert back to original scale
                train_predictions = scaler.inverse_transform(train_predictions)
                test_predictions = scaler.inverse_transform(test_predictions)
                y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Calculate metrics
                train_metrics = calculate_metrics(y_train_actual, train_predictions)
                test_metrics = calculate_metrics(y_test_actual, test_predictions)
                
                # Predict future prices
                last_sequence = scaled_data[-sequence_length:]
                future_predictions = predict_future_prices(model, last_sequence, scaler, prediction_days)
            
            # Display results
            st.header("📊 Prediction Results")
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Training Metrics")
                st.metric("MAPE", f"{train_metrics['MAPE']:.2f}%")
                st.metric("RMSE", f"${train_metrics['RMSE']:.2f}")
            
            with col2:
                st.subheader("🎯 Testing Metrics")
                st.metric("MAPE", f"{test_metrics['MAPE']:.2f}%")
                st.metric("RMSE", f"${test_metrics['RMSE']:.2f}")
            
            # Accuracy interpretation
            if test_metrics['MAPE'] < 5:
                st.markdown('<div class="success-box" style="color: black">🎉 <strong>Excellent accuracy!</strong> MAPE < 5% indicates very reliable predictions.</div>', unsafe_allow_html=True)
            elif test_metrics['MAPE'] < 10:
                st.markdown('<div class="success-box" style="color: black">✅ <strong>Good accuracy!</strong> MAPE < 10% indicates reliable predictions.</div>', unsafe_allow_html=True)
            elif test_metrics['MAPE'] < 20:
                st.markdown('<div class="warning-box" style="color: black">⚠️ <strong>Fair accuracy.</strong> Use predictions with caution.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box" style="color: black">❌ <strong>Poor accuracy.</strong> Consider using more data or different parameters.</div>', unsafe_allow_html=True)
            # Visualizations
            st.header("📈 Visualizations")
            
            try:
                # Method 1: Create chart with Streamlit + Matplotlib
                st.subheader(f"📊 {company_name} ({ticker}) - LSTM Price Predictions")
                
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # Plot historical prices
                ax.plot(data.index, data['Close'], label='Historical Prices', color='#1f77b4', linewidth=2)
                
                # Add training predictions if available
                if 'train_predictions' in locals() and train_predictions is not None and len(train_predictions) > 0:
                    try:
                        train_flat = train_predictions.flatten()
                        pred_start_idx = sequence_length
                        pred_end_idx = min(pred_start_idx + len(train_flat), len(data))
                        
                        if pred_end_idx > pred_start_idx:
                            pred_dates = data.index[pred_start_idx:pred_end_idx]
                            pred_values = train_flat[:len(pred_dates)]
                            ax.plot(pred_dates, pred_values, label='Training Predictions', 
                                   color='#ff7f0e', linewidth=2, linestyle='--', alpha=0.8)
                    except:
                        pass
                
                # Add test predictions if available
                if 'test_predictions' in locals() and test_predictions is not None and len(test_predictions) > 0:
                    try:
                        test_flat = test_predictions.flatten()
                        split_point = int(len(data) * 0.8)
                        test_start_idx = max(sequence_length, split_point)
                        test_end_idx = min(test_start_idx + len(test_flat), len(data))
                        
                        if test_end_idx > test_start_idx:
                            test_dates = data.index[test_start_idx:test_end_idx]
                            test_values = test_flat[:len(test_dates)]
                            ax.plot(test_dates, test_values, label='Test Predictions', 
                                   color='#d62728', linewidth=2)
                    except:
                        pass
                
                # Add future predictions if available
                if 'future_predictions' in locals() and future_predictions is not None and len(future_predictions) > 0:
                    try:
                        last_date = data.index[-1]
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                                   periods=len(future_predictions), freq='D')
                        future_values = future_predictions.flatten()
                        
                        ax.plot(future_dates, future_values, label='Future Predictions', 
                               color='#2ca02c', linewidth=3, marker='o', markersize=6)
                    except:
                        pass
                
                # Customize the plot
                ax.set_title(f'{company_name} ({ticker}) - LSTM Price Predictions', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Price ($)', fontsize=12)
                ax.legend(loc='upper left', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Display the chart
                st.pyplot(fig)
                
            except Exception as e:
                # Fallback: Simple separate charts
                try:
                    # Historical prices chart
                    st.line_chart(data['Close'])
                    
                    # Predictions comparison if available
                    if 'test_predictions' in locals() and test_predictions is not None:
                        # Create simple comparison DataFrame
                        split_point = int(len(data) * 0.8)
                        test_start = max(sequence_length, split_point)
                        test_data = data.iloc[test_start:test_start+len(test_predictions)]
                        
                        comparison_df = pd.DataFrame({
                            'Actual': test_data['Close'].values[:len(test_predictions)],
                            'Predicted': test_predictions.flatten()
                        }, index=test_data.index[:len(test_predictions)])
                        
                        st.line_chart(comparison_df)
                    
                except Exception as fallback_error:
                    st.dataframe(data.tail(10))
            
            # Secondary chart: Training history (if available)
            try:
                if 'history' in locals() and history is not None:
                    st.subheader("📊 Model Training Progress")
                    
                    training_fig = go.Figure()
                    training_fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                    if 'val_loss' in history.history:
                        training_fig.add_trace(go.Scatter(
                            y=history.history['val_loss'],
                            mode='lines',
                            name='Validation Loss',
                            line=dict(color='red')
                        ))
                    
                    training_fig.update_layout(
                        title='Model Training Loss Over Time',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(training_fig, use_container_width=True)
            except:
                pass
            
            # Future predictions table
            st.header("🔮 Future Price Predictions")
            try:
                if len(future_predictions) > 0 and not data.empty:
                    current_price = float(data['Close'].iloc[-1])
                    
                    # Use business days for the table too
                    future_dates = pd.date_range(
                        start=data.index[-1] + pd.Timedelta(days=1), 
                        periods=prediction_days, 
                        freq='B'
                    )
                    
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': [f"${price:.2f}" for price in future_predictions],
                        'Change from Current': [f"{((price - current_price) / current_price * 100):+.2f}%" for price in future_predictions]
                    })
                    
                    st.dataframe(future_df, use_container_width=True)
                else:
                    st.error("No future predictions to display")
                
            except Exception as e:
                st.error(f"Error creating predictions table: {str(e)}")
            
            # Download results - SIMPLIFIED VERSION
            st.header("💾 Download Results")

            try:
                # Helper function to safely convert to float
                def safe_float_convert(value):
                    """Safely convert value to float, return NaN if conversion fails"""
                    try:
                        if pd.isna(value) or value == '' or value is None:
                            return np.nan
                        return float(value)
                    except (ValueError, TypeError):
                        return np.nan
                
                # Get the base data length
                data_length = len(data)
                
                # Create aligned data arrays - fix the Close column processing
                dates_list = [str(d) for d in data.index]
                
                # Fix: Properly convert the Close column to list
                close_values = data['Close'].values  # Get numpy array first
                actual_prices_list = []
                for price in close_values:
                    converted_price = safe_float_convert(price)
                    actual_prices_list.append(converted_price)
                
                # Initialize predictions list
                predictions_list = [np.nan] * data_length
                
                # Add training predictions at correct positions
                if 'train_predictions' in locals() and train_predictions is not None:
                    try:
                        train_flat = train_predictions.flatten()
                        # Ensure we don't go beyond array bounds
                        max_train_idx = min(sequence_length + len(train_flat), data_length)
                        for i, pred in enumerate(train_flat):
                            idx = sequence_length + i
                            if idx < max_train_idx:
                                predictions_list[idx] = safe_float_convert(pred)
                    except:
                        pass
                
                # Add test predictions at correct positions
                if 'test_predictions' in locals() and test_predictions is not None:
                    try:
                        test_flat = test_predictions.flatten()
                        # Calculate where test predictions should start
                        split_point = int(len(data) * 0.8)
                        test_start_idx = max(sequence_length, split_point)
                        
                        # Ensure we don't go beyond array bounds
                        max_test_idx = min(test_start_idx + len(test_flat), data_length)
                        for i, pred in enumerate(test_flat):
                            idx = test_start_idx + i
                            if idx < max_test_idx:
                                predictions_list[idx] = safe_float_convert(pred)
                    except:
                        pass
                
                # Create DataFrame with verified equal-length arrays
                if len(dates_list) == len(actual_prices_list) == len(predictions_list):
                    results_df = pd.DataFrame({
                        'Date': dates_list,
                        'Actual_Price': actual_prices_list,
                        'Predicted_Price': predictions_list
                    })
                    
                    # Create CSV and download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"{ticker}_lstm_predictions.csv",
                        mime="text/csv"
                    )
                    
                    st.success("✅ Download ready!")
                    
                    # Show preview
                    with st.expander("📊 Preview Download Data"):
                        st.dataframe(results_df.head(10))
                        
                else:
                    st.error("Unable to prepare download data - array length mismatch")
                
            except Exception as e:
                st.error(f"Download error: {str(e)}")
            
            # Model summary
            with st.expander("🔍 View Model Architecture"):
                try:
                    # Capture model summary as string
                    summary_list = []
                    model.summary(print_fn=lambda x: summary_list.append(x))
                    summary_str = '\n'.join(summary_list)
                    
                    if summary_str:
                        st.text(summary_str)
                    else:
                        st.error("Could not generate model summary")
                        
                    # Show basic model info
                    st.write("**Model Configuration:**")
                    st.write(f"- Input shape: {model.input_shape}")
                    st.write(f"- Output shape: {model.output_shape}")
                    st.write(f"- Total parameters: {model.count_params():,}")
                    
                    # Show layer information
                    st.write("**Model Layers:**")
                    for i, layer in enumerate(model.layers):
                        st.write(f"Layer {i+1}: {layer.name} ({layer.__class__.__name__})")
                        if hasattr(layer, 'units'):
                            st.write(f"  - Units: {layer.units}")
                        if hasattr(layer, 'activation'):
                            st.write(f"  - Activation: {layer.activation}")
                        if hasattr(layer, 'rate'):
                            st.write(f"  - Dropout rate: {layer.rate}")
                
                except Exception as e:
                    st.error(f"Error displaying model architecture: {str(e)}")
                    st.write("**Basic Model Info:**")
                    try:
                        st.write(f"Model type: {type(model).__name__}")
                        st.write(f"Number of layers: {len(model.layers)}")
                        st.write("Model exists and was compiled successfully")
                    except:
                        st.write("Model information unavailable")
            
            # Training history
            with st.expander("📊 View Training History"):
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(
                    y=history.history['loss'],
                    mode='lines',
                    name='Training Loss'
                ))
                fig_history.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Validation Loss'
                ))
                fig_history.update_layout(
                    title='Model Training History',
                    xaxis_title='Epoch',
                    yaxis_title='Loss'
                )
                st.plotly_chart(fig_history)
            
        else:
            st.error(f"❌ Could not download data for ticker '{ticker}'. Please check if the ticker symbol is valid.")
    else:
        st.warning("⚠️ Please enter a ticker symbol.")

# Information section
st.header("ℹ️ How It Works")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 LSTM Neural Networks")
    st.write("""
    LSTM (Long Short-Term Memory) networks are a type of recurrent neural network 
    capable of learning long-term dependencies. They're particularly well-suited for 
    time series prediction because they can:
    
    - Remember important patterns from the past
    - Forget irrelevant information
    - Make predictions based on sequential data
    """)

with col2:
    st.subheader("📊 Model Features")
    st.write("""
    Our LSTM model uses:
    
    - **2 LSTM layers** with 50 neurons each
    - **Dropout layers** to prevent overfitting  
    - **Dense layers** for final prediction
    - **60-day sequences** as default input
    - **Min-Max scaling** for data normalization
    """)

# Disclaimer
st.markdown("---")
st.markdown("""
**⚠️ Disclaimer:** This tool is for educational purposes only. Stock market predictions are inherently uncertain, 
and this model should not be used as the sole basis for investment decisions. Always conduct your own research 
and consider consulting with financial professionals before making investment choices. Past performance does not 
guarantee future results.
""")

# Footer
st.markdown("---")
st.markdown("**Made with ❤️ using Streamlit and TensorFlow**")
