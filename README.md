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

   🛠️ How to Run the Streamlit Dashboard Locally
----------------------------------------------

1. Clone this repository:
    ```bash
    git clone https://github.com/your-repo/stock-sentiment-analysis.git
    cd stock-sentiment-analysis
    ```

2. Install required Python libraries:
    ```bash
    pip install streamlit pandas yfinance newsapi-python alpha_vantage transformers scikit-learn xgboost
    ```

3. **Configure API Keys**:
    * Open `app.py`
    * Replace the placeholder API keys:
        * `NewsAPI` Key
        * `Alpha Vantage` Key

4. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

* * *

🧠 Project Highlights
---------------------

*   Real-time stock data fetching (5-min intraday)
*   Sentiment analysis using FinBERT for financial headlines
*   Moving averages calculation for 5-day and 10-day periods
*   Live feature generation and input to pre-trained Random Forest model
*   Dashboard display of prediction, sentiment score, stock metrics, and news headlines
*   Error handling for API failures and missing data

* * *

🧪 Machine Learning Details
---------------------------

*   **Feature Engineering**:
    *   `open_price`, `close_price`, `volume`
    *   `sentiment_score`, `prev_sentiment`
    *   `ma_5`, `ma_10` (5 and 10-day moving averages)

*   **Model Comparison**:

    | Model | Accuracy (%) |
    | --- | --- |
    | Logistic Regression | 46.78% |
    | Random Forest Classifier | **69.97%** |
    | XGBoost Classifier | 62.72% |
    | Support Vector Machine | 56.93% |
    | K-Nearest Neighbors | 64.17% |

*   **Best Model**: Random Forest (saved as `best_rf.pkl`)

* * *

📜 License
----------

This project is developed solely for academic purposes as part of the **Web Mining (IS688102)** course at [Your University Name].

* * *

🙏 Acknowledgements
-------------------

*   [HuggingFace - FinBERT Tone Model](https://huggingface.co/yiyanghkust/finbert-tone)
*   [NewsAPI](https://newsapi.org/)
*   [Alpha Vantage API](https://www.alphavantage.co/)
*   [Yahoo Finance](https://finance.yahoo.com/)

* * *


