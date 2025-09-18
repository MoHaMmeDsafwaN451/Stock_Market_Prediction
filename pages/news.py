# pages/news.py
import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Stock News", layout="wide")

# Finnhub API (with your real API key)
API_KEY = "d360nbhr01qumnp3lp3gd360nbhr01qumnp3lp40"
BASE_URL = "https://finnhub.io/api/v1/news?category=general&token=" + API_KEY

def fetch_news():
    try:
        response = requests.get(BASE_URL, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def main():
    st.title("📰 Real-time Stock Market News")
    
    with st.spinner("Fetching latest market news..."):
        news_data = fetch_news()
    
    if not news_data:
        st.error("Failed to fetch news. Try again later.")
        return

    st.info(f"📈 Showing latest {min(10, len(news_data))} market news articles")
    
    for i, article in enumerate(news_data[:10], 1):  # Show top 10
        with st.expander(f"📄 {i}. {article.get('headline', 'No headline')}"):
            st.write(article.get('summary', 'No summary available'))
            st.caption(f"Source: {article.get('source', 'Unknown')} | {datetime.utcfromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')}")
            if article.get('url'):
                st.markdown(f"[🔗 Read Full Article]({article['url']})")
        st.divider()

if __name__ == "__main__":
    main()
