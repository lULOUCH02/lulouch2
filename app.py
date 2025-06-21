from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import ta
import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from urllib.parse import urlparse
from datetime import datetime
import pytz
from dateutil import parser

app = Flask(__name__)

local_tz = pytz.timezone("Asia/Riyadh")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
labels = ["سلبي", "محايد", "إيجابي"]

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    return labels[scores.argmax().item()]

def is_bitcoin_related(title, summary, link=""):
    text = (title + " " + summary).lower()
    parsed_url = urlparse(link)
    path_text = parsed_url.path.lower()
    return any(keyword in (text + " " + path_text) for keyword in ["bitcoin", "btc"])

@app.route("/")
def index():
    now = datetime.now(local_tz)
    start_time = local_tz.localize(datetime(now.year, now.month, now.day, 0, 0))
    end_time = now

    rss_feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://cryptoslate.com/feed/",
        "https://news.bitcoin.com/feed/",
        "https://decrypt.co/feed",
        "https://u.today/rss",
        "https://bitcoinmagazine.com/.rss/full/",
        "https://www.newsbtc.com/feed/",
        "https://www.investing.com/rss/news_285.rss",
        "https://www.fxstreet.com/rss/cryptocurrencies",
        "https://ambcrypto.com/feed/",
        "https://cryptobriefing.com/feed/",
        "https://www.cryptopolitan.com/feed/",
        "https://coinjournal.net/feed/",
        "https://www.ccn.com/crypto/feed/",
        "https://www.blockonomi.com/feed/",
        "https://www.livebitcoinnews.com/feed/",
        "https://cryptonews.com/news/feed/",
    ]

    positive_news, negative_news = 0, 0
    for url in rss_feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            published_str = getattr(entry, 'published', None)
            if not published_str:
                continue
            published = parser.parse(published_str).astimezone(local_tz)
            if not (start_time <= published <= end_time):
                continue
            title = getattr(entry, 'title', '')
            link = getattr(entry, 'link', '')
            summary = getattr(entry, 'summary', title)
            if not is_bitcoin_related(title, summary, link):
                continue
            sentiment = classify_sentiment(summary)
            if sentiment == "إيجابي":
                positive_news += 1
            elif sentiment == "سلبي":
                negative_news += 1

    btc = yf.download("BTC-USD", period="2d", interval="1h")
    close = btc["Close"].squeeze()
    price_now = close.iloc[-1]
    price_prev = close.iloc[-2]
    price_change_pct = ((price_now - price_prev) / price_prev) * 100

    rsi = ta.momentum.RSIIndicator(close=close).rsi().iloc[-1]
    macd = ta.trend.MACD(close=close)
    macd_line = macd.macd().iloc[-1]
    macd_signal = macd.macd_signal().iloc[-1]

    score = 50
    if price_change_pct > 2:
        score += 10
    elif price_change_pct < -2:
        score -= 10

    if rsi > 70:
        score -= 10
    elif rsi < 30:
        score += 10

    score += 5 if macd_line > macd_signal else -5
    score += (positive_news - negative_news) * 3
    score = max(0, min(score, 100))

    if score > 70:
        recommendation = "🚀 السوق متفائل جدًا (فكر بالبيع)"
    elif score < 30:
        recommendation = "🧊 السوق خائف (فرصة شراء)"
    else:
        recommendation = "⏳ السوق متوازن أو غير واضح"

    return jsonify({
        "التوقيت": now.strftime("%Y-%m-%d %H:%M"),
        "السعر الحالي": f"${price_now:,.2f}",
        "تغير السعر": f"{price_change_pct:+.2f}%",
        "RSI": f"{rsi:.2f}",
        "MACD": f"{macd_line:.2f} / {macd_signal:.2f}",
        "أخبار إيجابية": positive_news,
        "أخبار سلبية": negative_news,
        "المؤشر": f"{score}/100",
        "التوصية": recommendation
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
