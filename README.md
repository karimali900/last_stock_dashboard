# 📈 Last Stock Dashboard

**Real-time Stock Market Analysis Dashboard**  
Built with **PySpark + Streamlit + PostgreSQL + yfinance**

A powerful, automated stock monitoring system with live prices, technical indicators, sentiment analysis, Prophet forecasts, and price alerts.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B)

---

## ✨ Features

- **Live Stock Prices** with real-time percentage changes
- **Advanced Technical Analysis** (RSI, MACD, Bollinger Bands)
- **Sentiment Analysis** from latest news
- **30-Day Price Forecast** using Prophet
- **Price Alert System** (±X% alerts via email)
- **Real-time Mode** (updates every 60 minutes)
- **Beautiful Interactive Dashboard** with Streamlit
- **Persistent Storage** in PostgreSQL
- **Parallel Data Fetching** with PySpark

---

## 🛠 Tech Stack

- **Python 3.11**
- **PySpark** – Parallel data fetching
- **Streamlit** – Web dashboard
- **PostgreSQL** – Data storage
- **yfinance** – Stock data & news
- **Prophet** – Forecasting
- **Plotly** – Interactive charts

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/karimali900/last_stock_dashboard.git
cd last_stock_dashboard
## pip install -r requirements.txt
## EMAIL_PASSWORD=your_gmail_app_password
DB_PASSWORD=

then run streamlit run stock_dashboard.py
 Author:Karim El-Masri
Stock Market Automation Project


