# 📈 Stock Market Sentiment Analysis and Prediction

Welcome to our project repository for **Stock Market Sentiment Analysis and Price Movement Prediction**.  
This project combines **Web Mining**, **Natural Language Processing (NLP)**, and **Machine Learning** to predict short-term stock movements based on real-time news sentiment and technical stock indicators.

---

## 🚀 Project Overview

- **Goal**: Predict whether a stock (Apple or Tesla) will move **Up**, **Down**, or **Stay Stable** using live financial news and stock metrics.
- **Model Used**: Random Forest Classifier
- **Deployment**: Streamlit Dashboard (real-time, API-driven)
- **Sentiment Engine**: FinBERT (Finance-specific BERT model)
- **Live APIs Integrated**:
  - Alpha Vantage (Stock Prices)
  - Yahoo Finance (`yfinance` library for moving averages)
  - NewsAPI (Real-time News Headlines)

---

## 📁 Repository Structure

| File/Folder            | Description                                          |
|-------------------------|------------------------------------------------------|
| `best_rf.pkl`           | Trained Random Forest model (serialized with pickle) |
| `stock_sentiment.ipynb` | Jupyter Notebook (data collection, preprocessing, model training) |
| `app.py`                | Streamlit application for live prediction           |
| `final_report.docx/pdf` | Full project report                                 |
| `presentation.pptx`     | PowerPoint slides for presentation                  |
| `README.md`             | Project documentation (this file)                   |

---

## 🛠️ How to Run the Streamlit Dashboard Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/stock-sentiment-analysis.git
   cd stock-sentiment-analysis
