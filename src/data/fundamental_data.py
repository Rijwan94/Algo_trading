import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re

def fetch_forexfactory_calendar(start_date, end_date):
    """
    Fetches fundamental news calendar from ForexFactory for a given date range.
    Note: ForexFactory usually only allows scraping for recent/upcoming data without a session cookie,
    but this provides a good structural implementation.

    Args:
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: DataFrame containing news events and their impact.
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        events = []
        for item in data:
            # Impact mapping
            impact = item.get('impact', 'Low')
            impact_score = 0
            if impact == 'High':
                impact_score = 3
            elif impact == 'Medium':
                impact_score = 2
            elif impact == 'Low':
                impact_score = 1

            events.append({
                'title': item.get('title', ''),
                'country': item.get('country', ''),
                'date': item.get('date', ''),
                'impact': impact,
                'impact_score': impact_score,
                'forecast': item.get('forecast', ''),
                'previous': item.get('previous', '')
            })

        df = pd.DataFrame(events)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            # Filter by date range
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df = df.loc[mask]

        return df

    except Exception as e:
        print(f"Error fetching fundamental data: {e}")
        return pd.DataFrame()


def apply_news_impact_to_pair(df_market, df_news, pair_symbol):
    """
    Applies fundamental news impact to a forex pair's historical data.
    Takes into account base vs quote currency.
    """
    if len(pair_symbol) == 6:
        base_currency = pair_symbol[:3]
        quote_currency = pair_symbol[3:]
    else:
        # Default fallback for things like XAUUSD
        base_currency = pair_symbol.replace("USD", "")
        quote_currency = "USD"

    # Initialize impact column
    df_market['news_impact'] = 0.0

    if df_news.empty:
        return df_market

    # For each hour in market data, find events that happened in that hour
    for idx in df_market.index:
        start_time = idx
        end_time = idx + timedelta(hours=1)

        # Get news in this window
        mask = (df_news['date'] >= start_time) & (df_news['date'] < end_time)
        events = df_news[mask]

        net_impact = 0.0
        for _, event in events.iterrows():
            # If news is for base currency
            if event['country'] == base_currency:
                net_impact += event['impact_score']
            # If news is for quote currency, it has inverse effect on pair
            elif event['country'] == quote_currency:
                net_impact -= event['impact_score']

        df_market.loc[idx, 'news_impact'] = net_impact

    return df_market

if __name__ == "__main__":
    start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    end = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    print(f"Fetching news from {start} to {end}")
    df_news = fetch_forexfactory_calendar(start, end)
    print("Found news events:")
    print(df_news.head() if not df_news.empty else "No news found")
