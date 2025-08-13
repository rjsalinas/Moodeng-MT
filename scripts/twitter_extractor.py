#!/usr/bin/env python3
"""
Innovative Twitter Tweet Extractor using Apify
Extracts tweets from twitter_links.txt using Apify's Twitter scraper
Optimized for speed and reliability with fallback methods
"""

import os
import re
import json
import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import logging
from datetime import datetime
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
LINKS_FILE = Path("twitter_links.txt")
OUTPUT_DIR = Path("data") / "extracted_tweets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Environment variables
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "").strip()
TWITTER_BEARER = os.getenv("X_BEARER_TOKEN", "").strip()

# Apify Actor ID for Twitter scraper
APIFY_ACTOR_ID = "VsTreSuczsXhhRIqa"

# Configuration
BATCH_SIZE = 50  # Process links in batches
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
RATE_LIMIT_DELAY = 1  # seconds between requests


@dataclass
class ExtractedTweet:
    """Data structure for extracted tweet information"""
    tweet_id: str
    username: str
    text: str
    created_at: Optional[str]
    language: Optional[str]
    retweet_count: Optional[int]
    like_count: Optional[int]
    reply_count: Optional[int]
    quote_count: Optional[int]
    is_retweet: bool
    is_quote: bool
    has_media: bool
    hashtags: List[str]
    mentions: List[str]
    urls: List[str]
    extraction_method: str
    extraction_timestamp: str
    original_url: str


class TwitterLinkProcessor:
    """Processes Twitter links and extracts tweet IDs and usernames"""
    
    @staticmethod
    def extract_tweet_info(url: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract tweet ID and username from Twitter URL"""
        # Handle various Twitter URL formats
        patterns = [
            r'twitter\.com/([^/]+)/status/(\d+)',
            r'x\.com/([^/]+)/status/(\d+)',
            r'twitter\.com/i/web/status/(\d+)',
            r'x\.com/i/web/status/(\d+)',
            r'twitter\.com/i/status/(\d+)',
            r'x\.com/i/status/(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                if len(match.groups()) == 2:
                    username, tweet_id = match.groups()
                    # Filter out non-username parts
                    if username not in ['i', 'web']:
                        return username, tweet_id
                else:
                    tweet_id = match.group(1)
                    # For URLs without username, we'll extract it later
                    return None, tweet_id
        
        return None, None
    
    @staticmethod
    def load_links() -> List[str]:
        """Load Twitter links from file"""
        if not LINKS_FILE.exists():
            raise FileNotFoundError(f"Twitter links file not found: {LINKS_FILE}")
        
        with open(LINKS_FILE, 'r', encoding='utf-8') as f:
            links = [line.strip() for line in f if line.strip()]
        
        # Remove duplicates while preserving order
        unique_links = list(dict.fromkeys(links))
        logger.info(f"Loaded {len(unique_links)} unique Twitter links from {len(links)} total")
        
        return unique_links
    
    @staticmethod
    def batch_links(links: List[str], batch_size: int) -> List[List[str]]:
        """Split links into batches for processing"""
        return [links[i:i + batch_size] for i in range(0, len(links), batch_size)]


class ApifyExtractor:
    """Extracts tweets using Apify's Twitter scraper"""
    
    def __init__(self, token: str):
        self.token = token
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Apify client"""
        try:
            from apify_client import ApifyClient
            self.client = ApifyClient(self.token)
            logger.info("Apify client initialized successfully")
        except ImportError:
            logger.error("apify-client not installed. Install with: pip install apify-client")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Apify client: {e}")
            raise
    
    async def extract_tweets_batch(self, urls: List[str]) -> List[ExtractedTweet]:
        """Extract tweets from a batch of URLs using Apify"""
        if not self.client:
            raise RuntimeError("Apify client not initialized")
        
        logger.info(f"Processing batch of {len(urls)} URLs via Apify")
        
        try:
            # Prepare input for Apify actor
            run_input = {
                "startUrls": [{"url": url} for url in urls],
                "handles": [],  # We'll extract usernames from URLs
                "userQueries": [],
                "tweetsDesired": len(urls),  # One tweet per URL
                "profilesDesired": 0,
                "withReplies": False,
                "includeUserInfo": True,
                "storeUserIfNoTweets": False,
                "proxyConfig": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"],
                },
            }
            
            # Start the actor run
            logger.info("Starting Apify actor run...")
            run = self.client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
            
            # Wait for completion and get results
            dataset_id = run["defaultDatasetId"]
            logger.info(f"Apify run completed. Dataset ID: {dataset_id}")
            
            # Fetch results
            dataset = self.client.dataset(dataset_id)
            items = list(dataset.iterate_items())
            
            logger.info(f"Retrieved {len(items)} items from Apify")
            
            # Process results
            extracted_tweets = []
            for item in items:
                tweet = self._process_apify_item(item)
                if tweet:
                    extracted_tweets.append(tweet)
            
            logger.info(f"Successfully extracted {len(extracted_tweets)} tweets")
            return extracted_tweets
            
        except Exception as e:
            logger.error(f"Apify extraction failed: {e}")
            raise
    
    def _process_apify_item(self, item: Dict[str, Any]) -> Optional[ExtractedTweet]:
        """Process a single item from Apify's output"""
        try:
            # Extract basic tweet information
            tweet_id = str(item.get("id", ""))
            text = str(item.get("text", "") or item.get("fullText", ""))
            username = str(item.get("username", "") or item.get("authorUsername", ""))
            created_at = item.get("createdAt") or item.get("timestamp")
            language = item.get("lang", "")
            
            # Extract metrics
            metrics = item.get("publicMetrics", {}) or {}
            retweet_count = metrics.get("retweetCount", 0)
            like_count = metrics.get("likeCount", 0)
            reply_count = metrics.get("replyCount", 0)
            quote_count = metrics.get("quoteCount", 0)
            
            # Extract additional information
            is_retweet = bool(item.get("retweetedTweet"))
            is_quote = bool(item.get("quotedTweet"))
            has_media = bool(item.get("media") or item.get("images"))
            
            # Extract hashtags, mentions, and URLs
            hashtags = self._extract_hashtags(text)
            mentions = self._extract_mentions(text)
            urls = self._extract_urls(text)
            
            # Create ExtractedTweet object
            tweet = ExtractedTweet(
                tweet_id=tweet_id,
                username=username,
                text=text,
                created_at=created_at,
                language=language,
                retweet_count=retweet_count,
                like_count=like_count,
                reply_count=reply_count,
                quote_count=quote_count,
                is_retweet=is_retweet,
                is_quote=is_quote,
                has_media=has_media,
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                extraction_method="apify",
                extraction_timestamp=datetime.now().isoformat(),
                original_url=item.get("url", "")
            )
            
            return tweet
            
        except Exception as e:
            logger.warning(f"Failed to process Apify item: {e}")
            return None
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from tweet text"""
        return re.findall(r'#(\w+)', text)
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from tweet text"""
        return re.findall(r'@(\w+)', text)
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from tweet text"""
        return re.findall(r'https?://\S+', text)


class TwitterAPIExtractor:
    """Fallback extractor using Twitter API v2"""
    
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.session = None
    
    async def extract_tweets_batch(self, tweet_ids: List[str]) -> List[ExtractedTweet]:
        """Extract tweets using Twitter API v2"""
        if not self.bearer_token:
            logger.warning("Twitter API bearer token not provided")
            return []
        
        logger.info(f"Processing batch of {len(tweet_ids)} tweet IDs via Twitter API")
        
        # Twitter API allows max 100 IDs per request
        batches = [tweet_ids[i:i+100] for i in range(0, len(tweet_ids), 100)]
        all_tweets = []
        
        for batch in batches:
            try:
                batch_tweets = await self._fetch_tweets_batch(batch)
                all_tweets.extend(batch_tweets)
                await asyncio.sleep(RATE_LIMIT_DELAY)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch batch: {e}")
        
        logger.info(f"Successfully extracted {len(all_tweets)} tweets via Twitter API")
        return all_tweets
    
    async def _fetch_tweets_batch(self, tweet_ids: List[str]) -> List[ExtractedTweet]:
        """Fetch a batch of tweets from Twitter API"""
        url = "https://api.twitter.com/2/tweets"
        params = {
            "ids": ",".join(tweet_ids),
            "tweet.fields": "id,text,created_at,lang,public_metrics,referenced_tweets",
            "expansions": "author_id,referenced_tweets",
            "user.fields": "id,username"
        }
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        
        try:
            async with self.session.get(url, params=params, headers=headers, timeout=60) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tweets_data = data.get("data", [])
                    users = data.get("includes", {}).get("users", [])
                    
                    # Create user lookup
                    user_lookup = {str(u["id"]): u["username"] for u in users}
                    
                    extracted_tweets = []
                    for tweet in tweets_data:
                        extracted_tweet = self._process_api_tweet(tweet, user_lookup)
                        if extracted_tweet:
                            extracted_tweets.append(extracted_tweet)
                    
                    return extracted_tweets
                else:
                    logger.warning(f"Twitter API returned status {resp.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Twitter API request failed: {e}")
            return []
    
    def _process_api_tweet(self, tweet: Dict[str, Any], user_lookup: Dict[str, str]) -> Optional[ExtractedTweet]:
        """Process a single tweet from Twitter API"""
        try:
            tweet_id = str(tweet.get("id", ""))
            text = str(tweet.get("text", ""))
            author_id = str(tweet.get("author_id", ""))
            username = user_lookup.get(author_id, "")
            created_at = tweet.get("created_at")
            language = tweet.get("lang", "")
            
            # Extract metrics
            metrics = tweet.get("public_metrics", {})
            retweet_count = metrics.get("retweet_count", 0)
            like_count = metrics.get("like_count", 0)
            reply_count = metrics.get("reply_count", 0)
            quote_count = metrics.get("quote_count", 0)
            
            # Determine tweet type
            referenced_tweets = tweet.get("referenced_tweets", [])
            is_retweet = any(ref["type"] == "retweeted" for ref in referenced_tweets)
            is_quote = any(ref["type"] == "quoted" for ref in referenced_tweets)
            
            # Extract hashtags, mentions, and URLs
            hashtags = re.findall(r'#(\w+)', text)
            mentions = re.findall(r'@(\w+)', text)
            urls = re.findall(r'https?://\S+', text)
            
            # Create ExtractedTweet object
            extracted_tweet = ExtractedTweet(
                tweet_id=tweet_id,
                username=username,
                text=text,
                created_at=created_at,
                language=language,
                retweet_count=retweet_count,
                like_count=like_count,
                reply_count=reply_count,
                quote_count=quote_count,
                is_retweet=is_retweet,
                is_quote=is_quote,
                has_media=False,  # API doesn't provide media info in basic fields
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                extraction_method="twitter_api",
                extraction_timestamp=datetime.now().isoformat(),
                original_url=f"https://twitter.com/i/status/{tweet_id}"
            )
            
            return extracted_tweet
            
        except Exception as e:
            logger.warning(f"Failed to process API tweet: {e}")
            return None


class TweetExtractor:
    """Main tweet extraction orchestrator"""
    
    def __init__(self):
        self.apify_extractor = None
        self.twitter_api_extractor = None
        self.processor = TwitterLinkProcessor()
        self.extracted_tweets: List[ExtractedTweet] = []
        self.stats = {
            "total_links": 0,
            "successfully_extracted": 0,
            "apify_extracted": 0,
            "api_extracted": 0,
            "failed_extractions": 0,
            "start_time": None,
            "end_time": None
        }
        
        self._init_extractors()
    
    def _init_extractors(self):
        """Initialize available extractors"""
        # Initialize Apify extractor if token is available
        if APIFY_TOKEN:
            try:
                self.apify_extractor = ApifyExtractor(APIFY_TOKEN)
                logger.info("Apify extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Apify extractor: {e}")
                self.apify_extractor = None
        
        # Initialize Twitter API extractor if bearer token is available
        if TWITTER_BEARER:
            self.twitter_api_extractor = TwitterAPIExtractor(TWITTER_BEARER)
            logger.info("Twitter API extractor initialized")
        
        if not self.apify_extractor and not self.twitter_api_extractor:
            raise RuntimeError("No extractors available. Please provide APIFY_TOKEN or X_BEARER_TOKEN")
    
    async def extract_all_tweets(self) -> List[ExtractedTweet]:
        """Extract all tweets from twitter_links.txt"""
        self.stats["start_time"] = datetime.now()
        
        # Load and process links
        links = self.processor.load_links()
        self.stats["total_links"] = len(links)
        
        logger.info(f"Starting extraction of {len(links)} tweets...")
        
        # Process in batches
        batches = self.processor.batch_links(links, BATCH_SIZE)
        
        async with aiohttp.ClientSession() as session:
            if self.twitter_api_extractor:
                self.twitter_api_extractor.session = session
            
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} links)")
                
                try:
                    # Try Apify first (faster)
                    if self.apify_extractor:
                        batch_tweets = await self.apify_extractor.extract_tweets_batch(batch)
                        if batch_tweets:
                            self.extracted_tweets.extend(batch_tweets)
                            self.stats["apify_extracted"] += len(batch_tweets)
                            self.stats["successfully_extracted"] += len(batch_tweets)
                            logger.info(f"Batch {i+1}: Apify extracted {len(batch_tweets)} tweets")
                            continue
                    
                    # Fallback to Twitter API
                    if self.twitter_api_extractor:
                        # Extract tweet IDs from URLs
                        tweet_ids = []
                        for url in batch:
                            _, tweet_id = self.processor.extract_tweet_info(url)
                            if tweet_id:
                                tweet_ids.append(tweet_id)
                        
                        if tweet_ids:
                            batch_tweets = await self.twitter_api_extractor.extract_tweets_batch(tweet_ids)
                            if batch_tweets:
                                self.extracted_tweets.extend(batch_tweets)
                                self.stats["api_extracted"] += len(batch_tweets)
                                self.stats["successfully_extracted"] += len(batch_tweets)
                                logger.info(f"Batch {i+1}: Twitter API extracted {len(batch_tweets)} tweets")
                            else:
                                self.stats["failed_extractions"] += len(batch)
                        else:
                            self.stats["failed_extractions"] += len(batch)
                    else:
                        self.stats["failed_extractions"] += len(batch)
                
                except Exception as e:
                    logger.error(f"Batch {i+1} failed: {e}")
                    self.stats["failed_extractions"] += len(batch)
                
                # Rate limiting between batches
                if i < len(batches) - 1:
                    await asyncio.sleep(RATE_LIMIT_DELAY)
        
        self.stats["end_time"] = datetime.now()
        self._print_final_stats()
        
        return self.extracted_tweets
    
    def _print_final_stats(self):
        """Print final extraction statistics"""
        duration = self.stats["end_time"] - self.stats["start_time"]
        success_rate = (self.stats["successfully_extracted"] / self.stats["total_links"]) * 100
        
        logger.info("=" * 60)
        logger.info("EXTRACTION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total links processed: {self.stats['total_links']}")
        logger.info(f"Successfully extracted: {self.stats['successfully_extracted']}")
        logger.info(f"  - Apify extracted: {self.stats['apify_extracted']}")
        logger.info(f"  - Twitter API extracted: {self.stats['api_extracted']}")
        logger.info(f"Failed extractions: {self.stats['failed_extractions']}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Total duration: {duration}")
        logger.info("=" * 60)
    
    def save_results(self):
        """Save extracted tweets to various formats"""
        if not self.extracted_tweets:
            logger.warning("No tweets to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        csv_file = OUTPUT_DIR / f"extracted_tweets_{timestamp}.csv"
        self._save_to_csv(csv_file)
        
        # Save to JSON
        json_file = OUTPUT_DIR / f"extracted_tweets_{timestamp}.json"
        self._save_to_json(json_file)
        
        # Save statistics
        stats_file = OUTPUT_DIR / f"extraction_stats_{timestamp}.json"
        self._save_stats(stats_file)
        
        # Save tweet ID to username mapping
        mapping_file = OUTPUT_DIR / f"tweetid_to_username_{timestamp}.csv"
        self._save_mapping(mapping_file)
        
        logger.info(f"Results saved to {OUTPUT_DIR}")
    
    def _save_to_csv(self, filepath: Path):
        """Save tweets to CSV format"""
        rows = []
        for tweet in self.extracted_tweets:
            rows.append({
                "tweet_id": tweet.tweet_id,
                "username": tweet.username,
                "text": tweet.text,
                "created_at": tweet.created_at,
                "language": tweet.language,
                "retweet_count": tweet.retweet_count,
                "like_count": tweet.like_count,
                "reply_count": tweet.reply_count,
                "quote_count": tweet.quote_count,
                "is_retweet": tweet.is_retweet,
                "is_quote": tweet.is_quote,
                "has_media": tweet.has_media,
                "hashtags": "; ".join(tweet.hashtags),
                "mentions": "; ".join(tweet.mentions),
                "urls": "; ".join(tweet.urls),
                "extraction_method": tweet.extraction_method,
                "extraction_timestamp": tweet.extraction_timestamp,
                "original_url": tweet.original_url
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Saved {len(rows)} tweets to CSV: {filepath}")
    
    def _save_to_json(self, filepath: Path):
        """Save tweets to JSON format"""
        tweets_data = []
        for tweet in self.extracted_tweets:
            tweets_data.append({
                "tweet_id": tweet.tweet_id,
                "username": tweet.username,
                "text": tweet.text,
                "created_at": tweet.created_at,
                "language": tweet.language,
                "metrics": {
                    "retweet_count": tweet.retweet_count,
                    "like_count": tweet.like_count,
                    "reply_count": tweet.reply_count,
                    "quote_count": tweet.quote_count
                },
                "metadata": {
                    "is_retweet": tweet.is_retweet,
                    "is_quote": tweet.is_quote,
                    "has_media": tweet.has_media,
                    "hashtags": tweet.hashtags,
                    "mentions": tweet.mentions,
                    "urls": tweet.urls
                },
                "extraction": {
                    "method": tweet.extraction_method,
                    "timestamp": tweet.extraction_timestamp,
                    "original_url": tweet.original_url
                }
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tweets_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(tweets_data)} tweets to JSON: {filepath}")
    
    def _save_stats(self, filepath: Path):
        """Save extraction statistics"""
        stats_data = {
            "extraction_summary": self.stats,
            "extraction_config": {
                "batch_size": BATCH_SIZE,
                "max_retries": MAX_RETRIES,
                "retry_delay": RETRY_DELAY,
                "rate_limit_delay": RATE_LIMIT_DELAY
            },
            "extractors_available": {
                "apify": self.apify_extractor is not None,
                "twitter_api": self.twitter_api_extractor is not None
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, default=str)
        
        logger.info(f"Saved statistics to: {filepath}")
    
    def _save_mapping(self, filepath: Path):
        """Save tweet ID to username mapping"""
        mapping_data = []
        for tweet in self.extracted_tweets:
            if tweet.tweet_id and tweet.username:
                mapping_data.append({
                    "tweet_id": tweet.tweet_id,
                    "username": tweet.username,
                    "extraction_method": tweet.extraction_method
                })
        
        df = pd.DataFrame(mapping_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Saved {len(mapping_data)} mappings to: {filepath}")


async def main():
    """Main function"""
    print("Innovative Twitter Tweet Extractor")
    print("=" * 50)
    print("This tool extracts tweets from twitter_links.txt using Apify")
    print("with Twitter API fallback for maximum reliability and speed.")
    print("=" * 50)
    
    # Check if required files exist
    if not LINKS_FILE.exists():
        print(f"ERROR: {LINKS_FILE} not found!")
        return
    
    # Check if we have at least one extractor
    if not APIFY_TOKEN and not TWITTER_BEARER:
        print("ERROR: No API tokens provided!")
        print("Please set either APIFY_TOKEN or X_BEARER_TOKEN environment variable")
        return
    
    if APIFY_TOKEN:
        print("✓ Apify token found - will use fast Apify extraction")
    if TWITTER_BEARER:
        print("✓ Twitter API token found - will use as fallback")
    
    print(f"Found {LINKS_FILE}")
    print(f"Output will be saved to {OUTPUT_DIR}")
    print("=" * 50)
    
    try:
        # Initialize extractor
        extractor = TweetExtractor()
        
        # Extract all tweets
        extracted_tweets = await extractor.extract_all_tweets()
        
        # Save results
        extractor.save_results()
        
        print(f"\nExtraction completed successfully!")
        print(f"Extracted {len(extracted_tweets)} tweets")
        print(f"Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"\nExtraction failed: {e}")
        print("Check the logs for more details.")


if __name__ == "__main__":
    asyncio.run(main())
