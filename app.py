# === Import Libraries ===
import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
from newsapi import NewsApiClient
from alpha_vantage.timeseries import TimeSeries
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# === Load your trained Random Forest Model ===
with open('best_rf.pkl', 'rb') as f:
    best_rf = pickle.load(f)

# === Load FinBERT for Sentiment Analysis ===

model_name = "yiyanghkust/finbert-tone"

# Load model and tokenizer separately
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)  # Add this!

# Now create pipeline manually
finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# === Initialize News API and Alpha Vantage API ===
newsapi = NewsApiClient(api_key='75fe9a978b73483eb03e708e767eab83')
ts = TimeSeries(key='75fe9a978b73483eb03e708e767eab83', output_format='pandas')

# === Streamlit UI ===
st.set_page_config(page_title="📈 Live Stock Sentiment Predictor", layout="wide")
st.title('📈 Live Stock Sentiment & Movement Prediction Dashboard')

ticker = st.text_input('Enter Stock Symbol (e.g., TSLA, AAPL)', 'TSLA')

if st.button('Predict Now'):
    with st.spinner('Fetching live data and predicting...'):
        
        # === 1. Fetch latest stock data from Alpha Vantage ===
        try:
            stock_data, _ = ts.get_intraday(symbol=ticker, interval='5min', outputsize='compact')
            latest_stock = stock_data.iloc[-1]
            open_price = latest_stock['1. open']
            prev_close_price = latest_stock['4. close']
            volume = latest_stock['5. volume']
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            st.stop()

        # === 2. Fetch last 10 days stock history for moving averages ===
        try:
            hist = yf.Ticker(ticker).history(period="10d")
            hist['ma_5'] = hist['Close'].rolling(window=5).mean()
            hist['ma_10'] = hist['Close'].rolling(window=10).mean()
            ma_5 = hist['ma_5'].iloc[-1]
            ma_10 = hist['ma_10'].iloc[-1]
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            st.stop()

        # === 3. Fetch live news headlines from News API ===
        try:
            articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
            news_headlines = [article['title'] for article in articles['articles']]
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            st.stop()

        # === 4. Sentiment Analysis using FinBERT ===
        try:
            sentiment_results = [finbert_pipeline(title[:512])[0] for title in news_headlines]
            sentiment_score = sum([r['score'] for r in sentiment_results]) / len(sentiment_results)
            prev_sentiment_score = sentiment_score  # simple assumption for now
        except Exception as e:
            st.error(f"Error during sentiment analysis: {e}")
            st.stop()

        # === 5. Prepare live input for Random Forest Prediction ===
        live_input = pd.DataFrame({
            'open_price': [open_price],
            'sentiment_score': [sentiment_score],
            'prev_sentiment': [prev_sentiment_score],
            'prev_close_price': [prev_close_price],
            'ma_5': [ma_5],
            'ma_10': [ma_10],
            'volume': [volume]
        })

        # === 6. Predict using Random Forest ===
        prediction = best_rf.predict(live_input)[0]
        label_map = {0: 'Down', 1: 'Stable', 2: 'Up'}
        predicted_movement = label_map.get(prediction, "Unknown")

    # === 7. Display Results ===
    st.success(f"🎯 Predicted Movement for {ticker}: {predicted_movement}")
    st.metric(label="Sentiment Score", value=f"{sentiment_score:.2f}")
    st.metric(label="Open Price", value=f"${open_price:.2f}")
    st.metric(label="Previous Close Price", value=f"${prev_close_price:.2f}")
    st.metric(label="5-day Moving Avg", value=f"${ma_5:.2f}")
    st.metric(label="10-day Moving Avg", value=f"${ma_10:.2f}")
    st.metric(label="Volume", value=f"{volume:,}")

    st.subheader("Latest News Headlines Analyzed")
    for i, headline in enumerate(news_headlines):
        st.write(f"📰 {i+1}. {headline}")

