"""
STEP 18 — VADER Sentiment Analysis on Customer Reviews
Adds sentiment_score column to olist_order_reviews in MySQL.

Run: python etl/vader_sentiment.py
"""

import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vader")

DB_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER','root')}:"
    f"{os.getenv('MYSQL_PASSWORD','password')}@"
    f"{os.getenv('MYSQL_HOST','localhost')}:"
    f"{os.getenv('MYSQL_PORT','3306')}/"
    f"{os.getenv('MYSQL_DATABASE','supply_chain_intelligence')}"
    "?charset=utf8mb4"
)

BATCH_SIZE = 5_000


def run_vader():
    engine = create_engine(DB_URL, echo=False)
    analyzer = SentimentIntensityAnalyzer()

    # Load reviews with text
    log.info("Loading reviews from MySQL...")
    df = pd.read_sql(
        """
        SELECT review_id,
               COALESCE(review_comment_title, '') AS title,
               COALESCE(review_comment_message, '') AS message,
               review_score
        FROM olist_order_reviews
        WHERE sentiment_score IS NULL
        """,
        engine,
    )
    log.info(f"  ↳ {len(df):,} reviews need sentiment scoring")

    if df.empty:
        log.info("All reviews already scored. Nothing to do.")
        return

    # Combine title + message for richer context
    df["text"] = (df["title"] + " " + df["message"]).str.strip()

    # Run VADER — compound score: -1 (most negative) to +1 (most positive)
    log.info("Running VADER sentiment analysis...")
    df["sentiment_score"] = df["text"].apply(
        lambda t: analyzer.polarity_scores(t)["compound"] if t.strip() else 0.0
    )

    # Distribution summary
    log.info("Sentiment distribution:")
    log.info(f"  Positive (>0.05):  {(df['sentiment_score'] >  0.05).sum():,}")
    log.info(f"  Neutral  (±0.05):  {(df['sentiment_score'].between(-0.05, 0.05)).sum():,}")
    log.info(f"  Negative (<-0.05): {(df['sentiment_score'] < -0.05).sum():,}")
    log.info(f"  Mean score: {df['sentiment_score'].mean():.3f}")

    # Write back to MySQL in batches
    log.info("Writing scores back to MySQL...")
    with engine.begin() as conn:
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i : i + BATCH_SIZE][["review_id", "sentiment_score"]]
            for _, row in batch.iterrows():
                conn.execute(
                    text(
                        "UPDATE olist_order_reviews "
                        "SET sentiment_score = :score "
                        "WHERE review_id = :rid"
                    ),
                    {"score": float(row["sentiment_score"]), "rid": row["review_id"]},
                )
            pct = min(100, int((i + BATCH_SIZE) / len(df) * 100))
            log.info(f"  ↳ Progress: {pct}% ({min(i + BATCH_SIZE, len(df)):,}/{len(df):,})")

    log.info("✓ VADER sentiment scores written to MySQL successfully!")

    # Verify
    with engine.connect() as conn:
        scored = conn.execute(
            text("SELECT COUNT(*) FROM olist_order_reviews WHERE sentiment_score IS NOT NULL")
        ).scalar()
    log.info(f"  Total reviews with sentiment_score: {scored:,}")


if __name__ == "__main__":
    run_vader()
