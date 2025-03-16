import os
import csv
from datetime import datetime
import requests
import pandas as pd
import google.generativeai as genai
from textblob import TextBlob
import tweepy
import yfinance as yf

class BitcoinDataAnalyzer:
    def __init__(self, 
                 twitter_api_key, 
                 twitter_api_secret, 
                 twitter_access_token, 
                 twitter_access_token_secret,
                 gemini_api_key,
                 news_api_key):
        """
        Initialize the Bitcoin Data Analyzer with API credentials
        
        Args:
            twitter_api_key (str): Twitter API key
            twitter_api_secret (str): Twitter API secret
            twitter_access_token (str): Twitter access token
            twitter_access_token_secret (str): Twitter access token secret
            gemini_api_key (str): Google Gemini API key
            news_api_key (str): NewsAPI key
        """
        # Twitter API Authentication
        auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
        auth.set_access_token(twitter_access_token, twitter_access_token_secret)
        self.twitter_api = tweepy.API(auth)
        
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        
        # Store NewsAPI key
        self.news_api_key = news_api_key
        
        # Ensure output directory exists
        os.makedirs('bitcoin_sentiment_data', exist_ok=True)
        os.makedirs('bitcoin_summaries', exist_ok=True)

    def fetch_and_save_twitter_sentiment(self, query='Bitcoin', max_tweets=100):
        """
        Fetch and save Twitter sentiment data
        
        Args:
            query (str): Search query for tweets
            max_tweets (int): Maximum number of tweets to fetch
        
        Returns:
            list: List of tweet sentiment data
        """
        sentiment_data = []
        
        try:
            # Fetch tweets
            tweets = tweepy.Cursor(self.twitter_api.search_tweets, 
                                   q=query, 
                                   lang='en', 
                                   tweet_mode='extended').items(max_tweets)
            
            for tweet in tweets:
                # Perform sentiment analysis
                full_text = tweet.full_text
                blob = TextBlob(full_text)
                
                sentiment_data.append({
                    'timestamp': tweet.created_at,
                    'text': full_text,
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
            
            # Save to CSV
            filepath = os.path.join('bitcoin_sentiment_data', 'twitter_sentiment.csv')
            df = pd.DataFrame(sentiment_data)
            df.to_csv(filepath, index=False)
            print(f"Twitter sentiment data saved to {filepath}")
        
        except Exception as e:
            print(f"Error fetching Twitter sentiment: {e}")
        
        return sentiment_data

    def fetch_and_save_news_sentiment(self):
        """
        Fetch and save news sentiment data
        
        Returns:
            list: List of news sentiment data
        """
        news_sentiment_data = []
        
        try:
            url = f'https://newsapi.org/v2/everything?q=Bitcoin&language=en&sortBy=publishedAt&apiKey={self.news_api_key}'
            
            response = requests.get(url)
            news_data = response.json()
            
            for article in news_data.get('articles', []):
                # Perform basic sentiment analysis on article title
                blob = TextBlob(article['title'])
                
                news_sentiment_data.append({
                    'timestamp': article['publishedAt'],
                    'title': article['title'],
                    'source': article['source']['name'],
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
            
            # Save to CSV
            filepath = os.path.join('bitcoin_sentiment_data', 'news_sentiment.csv')
            df = pd.DataFrame(news_sentiment_data)
            df.to_csv(filepath, index=False)
            print(f"News sentiment data saved to {filepath}")
        
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
        
        return news_sentiment_data

    def fetch_and_save_fear_and_greed_index(self):
        """
        Fetch and save the Crypto Fear & Greed Index
        
        Returns:
            dict or None: Fear and Greed Index data
        """
        try:
            # Alternative API for Fear & Greed Index
            url = 'https://api.alternative.me/fng/'
            response = requests.get(url)
            data = response.json()
            
            fear_greed_data = [{
                'timestamp': datetime.now(),
                'value': data['data'][0]['value'],
                'value_classification': data['data'][0]['value_classification']
            }]
            
            # Save to CSV
            filepath = os.path.join('bitcoin_sentiment_data', 'fear_greed_index.csv')
            df = pd.DataFrame(fear_greed_data)
            df.to_csv(filepath, index=False)
            print(f"Fear & Greed Index data saved to {filepath}")
            
            return fear_greed_data[0]
        
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return None

    def fetch_and_save_bitcoin_price(self):
        """
        Fetch and save current Bitcoin price
        
        Returns:
            dict or None: Bitcoin price data
        """
        try:
            # Fetch Bitcoin price using yfinance
            bitcoin = yf.Ticker('BTC-USD')
            history = bitcoin.history(period='1d')
            
            price_data = [{
                'timestamp': datetime.now(),
                'close_price': history['Close'].iloc[-1],
                'volume': history['Volume'].iloc[-1]
            }]
            
            # Save to CSV
            filepath = os.path.join('bitcoin_sentiment_data', 'bitcoin_price.csv')
            df = pd.DataFrame(price_data)
            df.to_csv(filepath, index=False)
            print(f"Bitcoin price data saved to {filepath}")
            
            return price_data[0]
        
        except Exception as e:
            print(f"Error fetching Bitcoin price: {e}")
            return None

    def generate_bitcoin_summary(self):
        """
        Generate a comprehensive summary using Gemini API
        
        Returns:
            str: Comprehensive summary of Bitcoin data
        """
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prepare prompt with data from CSV files
        prompt = "Provide a comprehensive analysis of today's Bitcoin market based on the following data in 1 sentences:\n\n"
        
        # List of CSV files to process
        csv_files = [
            'news_sentiment.csv',
            'fear_greed_index.csv',
            'bitcoin_price.csv'
        ]
        
        # Process each CSV file
        for filename in csv_files:
            filepath = os.path.join('bitcoin_sentiment_data', filename)
            
            try:
                # Read CSV file
                df = pd.read_csv(filepath)
                
                # Add data to prompt based on file type
                if filename == 'news_sentiment.csv':
                    prompt += "News Sentiment:\n"
                    for _, row in df.iterrows():
                        prompt += f"- Title: {row['title']}\n"
                        prompt += f"  Sentiment Polarity: {row['polarity']}\n"
                        prompt += f"  Subjectivity: {row['subjectivity']}\n"
                
                elif filename == 'fear_greed_index.csv':
                    prompt += "\nFear & Greed Index:\n"
                    for _, row in df.iterrows():
                        prompt += f"- Value: {row.get('value', 'N/A')}\n"
                        prompt += f"  Classification: {row.get('value_classification', 'N/A')}\n"
                
                elif filename == 'bitcoin_price.csv':
                    prompt += "\nBitcoin Price:\n"
                    for _, row in df.iterrows():
                        prompt += f"- Close Price: {row.get('close_price', 'N/A')}\n"
                        prompt += f"  Volume: {row.get('volume', 'N/A')}\n"
            
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        # Add context and request for analysis
        prompt += "\nProvide a comprehensive market analysis considering these data points. " \
                  "Discuss potential market trends, sentiment, and outlook. " \
                  "Include insights on price movement, news impact, and overall market sentiment. " \
                  "Format your response in clear, concise paragraphs."
        
        try:
            # Generate summary
            response = model.generate_content(prompt)
            
            # Save summary to file
            summary = response.text
            filename = f'bitcoin_market_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            filepath = os.path.join('bitcoin_summaries', filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"Summary saved to {filepath}")
            
            return summary
        
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Unable to generate summary due to an error."

    def run_comprehensive_analysis(self):
        """
        Run comprehensive Bitcoin data analysis
        
        Fetches data, saves to CSV, and generates summary
        """
        # Fetch and save data from various sources
        self.fetch_and_save_twitter_sentiment()
        self.fetch_and_save_news_sentiment()
        self.fetch_and_save_fear_and_greed_index()
        self.fetch_and_save_bitcoin_price()
        
        # Generate and return summary
        return self.generate_bitcoin_summary()

def main():
    # API Credentials (REPLACE WITH YOUR ACTUAL CREDENTIALS)
    TWITTER_API_KEY = 'VZI1qv5aLHaMP1uKYm0mVjxmi'
    TWITTER_API_SECRET = 'MD3aleOgOunXicSA5Uut29qaGTRiegm0QYLt4V0ZlnoAoIoFdr'
    TWITTER_ACCESS_TOKEN = '1473959633928679424-ekP6hB7U8wOep62w6c4Lw1ZgOhm7ty'
    TWITTER_ACCESS_TOKEN_SECRET = 'uxDT7U2yD92NnjCNd3icmcOihEWn6rovdQgKhte9iWpYt'
    GEMINI_API_KEY = 'AIzaSyBi7e2Adkd5OAYK2O1aPfxObMgtnHr7aX0'
    NEWS_API_KEY = 'ced5e2bcf3864a42a247c4ed476ba927'
    
    # Initialize and run comprehensive Bitcoin data analyzer
    analyzer = BitcoinDataAnalyzer(
        TWITTER_API_KEY, 
        TWITTER_API_SECRET, 
        TWITTER_ACCESS_TOKEN, 
        TWITTER_ACCESS_TOKEN_SECRET,
        GEMINI_API_KEY,
        NEWS_API_KEY
    )
    
    # Run comprehensive analysis
    summary = analyzer.run_comprehensive_analysis()
    
    # Print summary to console
    print("\n--- Bitcoin Market Summary ---")
    print(summary)

if __name__ == '__main__':
    main()

# Required Dependencies:
# pip install requests textblob tweepy yfinance pandas google-generativeai