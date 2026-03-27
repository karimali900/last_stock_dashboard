#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last_stock_enhace1_spark.py
FULL PySpark VERSION – FIXED (query_db + DB index error gone)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import argparse
import logging
import time
import smtplib
import sqlalchemy as sa
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from sqlalchemy import text

# PySpark
from pyspark.sql import SparkSession

# Optional
try:
    from textblob import TextBlob
    TEXTBLOB_OK = True
except ImportError:
    TEXTBLOB_OK = False

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-5s | %(message)s')

# ====================== 1. LOAD .env ======================
def load_dotenv_manual():
    if os.path.exists('.env'):
        print("✅ .env file found")
        with open('.env', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'').replace(' ', '')
                    os.environ[key] = value
                    if key in ('EMAIL_PASSWORD', 'DB_PASSWORD'):
                        print(f"   {key} loaded")
    else:
        print("⚠️  No .env file")

load_dotenv_manual()

# ====================== 2. CONFIG ======================
DEFAULT_TICKERS = ["AAPL", "TSLA", "COMI.CA", "HRHO.CA", "2222.SR", "VOD.L", "BP.L"]

EMAIL_FROM      = "90.karim@gmail.com"
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD")
EMAIL_TO        = ["karimali900@yahoo.ca", "90.karim@gmail.com"]
SMTP_SERVER     = "smtp.gmail.com"
SMTP_PORT       = 587

if not EMAIL_PASSWORD:
    print("❌ EMAIL_PASSWORD missing!")
    exit(1)

# ====================== 3. SPARK ======================
spark = SparkSession.builder \
    .appName("Karim_Stock_Analyzer") \
    .master("local[*]") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
print("🚀 PySpark initialized")

# ====================== 4. DATABASE (NOW FULLY FIXED) ======================
class DatabaseManager:
    def __init__(self):
        password = os.getenv("DB_PASSWORD", "karim")
        conn_str = f'postgresql://postgres:{password}@localhost:5432/stocks_2026'
        self.engine = sa.create_engine(conn_str, echo=False)
        self._create_table()

    def _create_table(self):
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_history (
                    Ticker TEXT, Datetime TIMESTAMP, Open FLOAT, High FLOAT,
                    Low FLOAT, Close FLOAT, Volume BIGINT,
                    PRIMARY KEY (Ticker, Datetime)
                )
            """))
            logging.info("✅ PostgreSQL table ready")

    def load_to_db(self, df: pd.DataFrame):
        if df.empty: return
        try:
            df = df.copy()
            df = df.reset_index(drop=True)                  # Remove any hidden index
            if isinstance(df.index, pd.MultiIndex) or any(c in df.columns for c in ['index', 'level_0']):
                df = df.reset_index()
                df = df.drop(columns=[c for c in df.columns if c in ['index', 'level_0']], errors='ignore')

            # Force exact columns
            df = df[['Ticker', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            df = df.drop_duplicates(subset=['Ticker', 'Datetime'], keep='last')

            df.to_sql('stock_history', self.engine, if_exists='append', index=False, method='multi')
            logging.info(f"✅ Saved {len(df)} rows to PostgreSQL")
        except Exception as e:
            logging.error(f"DB error: {e}")

    # ←←← THIS WAS MISSING ←←←
    def query_db(self, query, params=None):
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            logging.error(f"Query error: {e}")
            return pd.DataFrame()

# ====================== TECHNICAL INDICATORS ======================
def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(df, window=20):
    mb = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    return mb + 2*std, mb, mb - 2*std

# ====================== 5. STOCK REPORTER ======================
class StockReporter:
    def __init__(self, df: pd.DataFrame, db: DatabaseManager):
        self.df = df
        self.db = db
        self.summary = ""
        self.alerts = []

    def advanced_sentiment_analysis(self):
        self.summary += "\n=== SENTIMENT & TECHNICAL ANALYSIS ===\n"
        latest = self.db.query_db(text("""
            SELECT DISTINCT ON ("Ticker") "Ticker", "Datetime", "Close", "Open", "Volume", "High", "Low"
            FROM stock_history ORDER BY "Ticker", "Datetime" DESC
        """))
        avg_vol_df = self.db.query_db(text('SELECT "Ticker", AVG("Volume") AS avg FROM stock_history GROUP BY "Ticker"'))
        avg_vol = avg_vol_df.set_index("Ticker")["avg"].to_dict() if not avg_vol_df.empty else {}

        for ticker in sorted(self.df['Ticker'].unique()):
            news_score = None
            try:
                news = yf.Ticker(ticker).news or []
                scores = [TextBlob(a.get('title','')).sentiment.polarity for a in news[:5] if a.get('title')]
                news_score = float(np.mean(scores)) if scores else None
            except:
                pass

            status = "Positive 😊" if news_score and news_score > 0.15 else "Negative 😟" if news_score and news_score < -0.15 else "Neutral 😐"
            score_display = f"{news_score:.2f}" if news_score is not None else "N/A"
            self.summary += f"• {ticker}: {status} | Score: {score_display}\n"

            if news_score and news_score < -0.15:
                self.alerts.append(f"{ticker}: Strong negative ({news_score:.2f})")

            df_t = self.db.query_db(text("""
                SELECT * FROM stock_history WHERE "Ticker" = :t ORDER BY "Datetime" DESC LIMIT 30
            """), {"t": ticker})
            if len(df_t) < 14:
                self.summary += f"   Technical: Insufficient data\n\n"
                continue

            rsi = compute_rsi(df_t).iloc[0] if not compute_rsi(df_t).empty else None
            macd, sig = compute_macd(df_t)
            macd_val = macd.iloc[0] if not macd.empty else None
            sig_val = sig.iloc[0] if not sig.empty else None
            ub, mb, lb = compute_bollinger_bands(df_t)
            row = df_t.iloc[0]
            pct = ((row['Close'] - row['Open']) / row['Open'] * 100) if row['Open'] else 0

            rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
            macd_str = f"{macd_val:.3f}/{sig_val:.3f}" if macd_val is not None and sig_val is not None else "N/A"
            bb_str = f"${row['Close']:.2f} vs U:{ub.iloc[0]:.2f} M:{mb.iloc[0]:.2f} L:{lb.iloc[0]:.2f}" if not ub.empty else "N/A"

            self.summary += f"   RSI: {rsi_str} | MACD: {macd_str} | BB: {bb_str}\n"
            self.summary += f"   Price: ${row['Close']:.2f} ({pct:+.2f}%) | Vol: {row['Volume']:,}\n\n"

        if PROPHET_OK:
            self.summary += "\n=== 30-DAY PROPHET FORECAST ===\n"
            for t in sorted(self.df['Ticker'].unique()):
                df_p = self.db.query_db(text('SELECT "Datetime" AS ds, "Close" AS y FROM stock_history WHERE "Ticker" = :t ORDER BY "Datetime"'), {"t": t})
                if len(df_p) < 30: continue
                df_p['ds'] = pd.to_datetime(df_p['ds']).dt.tz_localize(None)
                try:
                    m = Prophet()
                    m.fit(df_p)
                    future = m.make_future_dataframe(periods=30)
                    forecast = m.predict(future)
                    self.summary += f"• {t} → ${forecast['yhat'].iloc[-1]:.2f}\n"
                except:
                    pass

    def build_report(self):
        today = datetime.now().strftime("%Y-%m-%d %H:%M EET")
        self.summary = f"Stock Report ─ {today}\n\nLIVE PRICES:\n"
        for t in sorted(self.df['Ticker'].unique()):
            p, ch = get_live_price(t)
            arrow = "▲" if ch > 0 else "▼" if ch < 0 else "─"
            self.summary += f"{t:8}  ${p:10,.2f}  {arrow} {ch:+6.2f}%\n" if p else f"{t:8}  N/A\n"

        self.advanced_sentiment_analysis()

        if self.alerts:
            self.summary += "\n🚨 ALERTS:\n" + "\n".join(self.alerts) + "\n"

        self.html_body = f"""
        <html><head><meta charset="utf-8"><title>Stock Report {today}</title></head>
        <body style="font-family:Arial,sans-serif;margin:30px;">
            <h1>Stock Report — {today}</h1>
            <pre style="background:#f8f9fa;padding:20px;border-radius:8px;white-space:pre-wrap;">{self.summary}</pre>
        </body></html>
        """

    def save_reports(self):
        fn = f"stock_report_{datetime.now():%Y%m%d_%H%M}.html"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(self.html_body)
        print(f"✅ HTML report saved → {fn}")

    def send_email(self):
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = EMAIL_FROM
            msg['To'] = ", ".join(EMAIL_TO)
            msg['Subject'] = f"Stock Report {datetime.now():%Y-%m-%d %H:%M}"
            msg['Date'] = formatdate(localtime=True)
            msg.attach(MIMEText(self.summary, 'plain', 'utf-8'))
            msg.attach(MIMEText(self.html_body, 'html', 'utf-8'))
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_PASSWORD)
                server.send_message(msg)
            print("✅ Full report email sent")
        except Exception as e:
            print(f"❌ Email failed: {e}")

    def send_price_alert(self, ticker, old_price, new_price, change_pct):
        emoji = "📈" if change_pct > 0 else "📉"
        subject = f"{emoji} PRICE ALERT: {ticker} {change_pct:+.2f}%"
        alert_html = f"<html><body><h2>🚨 PRICE ALERT — {ticker}</h2><p>Moved <strong>{change_pct:+.2f}%</strong></p><table border='1'><tr><td>Previous</td><td>${old_price:,.4f}</td></tr><tr><td>Current</td><td>${new_price:,.4f}</td></tr></table></body></html>"
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = EMAIL_FROM
            msg['To'] = ", ".join(EMAIL_TO)
            msg['Subject'] = subject
            msg['Date'] = formatdate(localtime=True)
            msg.attach(MIMEText(f"PRICE ALERT: {ticker} {change_pct:+.2f}%", 'plain'))
            msg.attach(MIMEText(alert_html, 'html'))
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_PASSWORD)
                server.send_message(msg)
            print(f"✅ ALERT SENT → {ticker} ({change_pct:+.2f}%)")
        except Exception as e:
            print(f"❌ Alert failed: {e}")

# ====================== HELPERS ======================
def get_live_price(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        p = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        ch = info.get('regularMarketChangePercent') or 0.0
        return round(p, 4) if p else None, round(ch, 2)
    except:
        return None, 0.0

def fetch_ticker_data(ticker):
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if df.empty: return pd.DataFrame()
        df = df.reset_index().rename(columns={'Date': 'Datetime'})
        df['Ticker'] = ticker
        return df[['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except:
        return pd.DataFrame()

def fetch_all_data_spark(tickers):
    print(f"🔥 Fetching {len(tickers)} tickers with PySpark...")
    rdd = spark.sparkContext.parallelize(tickers)
    pandas_dfs = rdd.map(fetch_ticker_data).filter(lambda df: not df.empty).collect()
    return pd.concat(pandas_dfs, ignore_index=True) if pandas_dfs else pd.DataFrame()

# ====================== MAIN ======================
class StockPipeline:
    def __init__(self):
        self.db = DatabaseManager()

    def run(self, tickers=None, email=False, real_time=False, alert_threshold=None):
        if tickers is None: tickers = DEFAULT_TICKERS

        df = fetch_all_data_spark(tickers)
        self.db.load_to_db(df)

        reporter = StockReporter(df, self.db)
        reporter.build_report()
        reporter.save_reports()
        if email:
            reporter.send_email()

        print("\n" + reporter.summary)

        if real_time:
            print(f"\n🔄 REAL-TIME MODE (60 min) – alerts at ±{alert_threshold}%")
            last_prices = {t: None for t in tickers}
            while True:
                time.sleep(3600)
                print(f"\n─── UPDATE {datetime.now():%H:%M:%S} ───")
                reporter.build_report()
                print(reporter.summary.split("LIVE PRICES")[1].split("\n\n")[0])

                if alert_threshold is not None:
                    for t in tickers:
                        p, _ = get_live_price(t)
                        if p is None: continue
                        old = last_prices.get(t)
                        if old and abs((p - old) / old * 100) >= alert_threshold:
                            reporter.send_price_alert(t, old, p, (p - old) / old * 100)
                        last_prices[t] = p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', type=str)
    parser.add_argument('--email', action='store_true')
    parser.add_argument('--real-time', action='store_true')
    parser.add_argument('--alert-threshold', type=float, default=None)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in (args.tickers or '').split(',') if t.strip()] or DEFAULT_TICKERS

    StockPipeline().run(
        tickers=tickers,
        email=args.email,
        real_time=args.real_time,
        alert_threshold=args.alert_threshold
    )
