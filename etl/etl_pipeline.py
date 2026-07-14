"""
STEP 5 — Python ETL Pipeline
Loads all 9 Olist CSV files into MySQL with cleaning, validation & category translation.

Run: python etl/etl_pipeline.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etl")

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# ── DB connection ─────────────────────────────────────────────────────────────
DB_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER', 'root')}:"
    f"{os.getenv('MYSQL_PASSWORD', 'password')}@"
    f"{os.getenv('MYSQL_HOST', 'localhost')}:"
    f"{os.getenv('MYSQL_PORT', '3306')}/"
    f"{os.getenv('MYSQL_DATABASE', 'supply_chain_intelligence')}"
    "?charset=utf8mb4"
)

DATETIME_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",
]

# ── Table configs ─────────────────────────────────────────────────────────────
TABLES = [
    {
        "file":  "olist_customers_dataset.csv",
        "table": "olist_customers",
        "pk":    "customer_id",
    },
    {
        "file":  "olist_geolocation_dataset.csv",
        "table": "olist_geolocation",
        "pk":    None,   # no PK — duplicates allowed for zip codes
    },
    {
        "file":  "olist_sellers_dataset.csv",
        "table": "olist_sellers",
        "pk":    "seller_id",
    },
    {
        "file":  "product_category_name_translation.csv",
        "table": "product_category_translation",
        "pk":    "product_category_name",
    },
    {
        "file":  "olist_products_dataset.csv",
        "table": "olist_products",
        "pk":    "product_id",
    },
    {
        "file":  "olist_orders_dataset.csv",
        "table": "olist_orders",
        "pk":    "order_id",
    },
    {
        "file":  "olist_order_items_dataset.csv",
        "table": "olist_order_items",
        "pk":    None,   # composite PK handled by schema
    },
    {
        "file":  "olist_order_payments_dataset.csv",
        "table": "olist_order_payments",
        "pk":    None,
    },
    {
        "file":  "olist_order_reviews_dataset.csv",
        "table": "olist_order_reviews",
        "pk":    "review_id",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse any datetime columns present in this dataframe."""
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_products(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Merge English category names from translation table into products."""
    try:
        translation = pd.read_sql(
            "SELECT product_category_name, product_category_name_english "
            "FROM product_category_translation",
            engine,
        )
        df = df.merge(translation, on="product_category_name", how="left")
        log.info("  ↳ Category translation merged into products")
    except Exception:
        df["product_category_name_english"] = df.get("product_category_name")
    return df


def clean_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate zip+lat+lng rows to keep geolocation lean."""
    before = len(df)
    df = df.drop_duplicates(
        subset=["geolocation_zip_code", "geolocation_lat", "geolocation_lng"]
    )
    log.info(f"  ↳ Geolocation: dropped {before - len(df):,} duplicate rows")
    return df


def load_table(cfg: dict, engine) -> dict:
    """Read, clean, and load a single CSV table. Returns row counts."""
    fpath = RAW_DIR / cfg["file"]
    if not fpath.exists():
        log.error(f"  ✗ File not found: {fpath}")
        return {"table": cfg["table"], "loaded": 0, "status": "MISSING"}

    log.info(f"Loading → {cfg['table']}")
    df = pd.read_csv(fpath, low_memory=False)
    log.info(f"  ↳ Read {len(df):,} rows from {cfg['file']}")

    # Parse dates
    df = parse_dates(df)

    # Table-specific cleaning
    if cfg["table"] == "olist_products":
        df = clean_products(df, engine)
    if cfg["table"] == "olist_geolocation":
        df = clean_geolocation(df)

    # Drop rows with null primary key
    if cfg["pk"] and cfg["pk"] in df.columns:
        before = len(df)
        df = df.dropna(subset=[cfg["pk"]])
        dropped = before - len(df)
        if dropped:
            log.warning(f"  ↳ Dropped {dropped} rows with null {cfg['pk']}")

    # Remove duplicate PKs (keep first)
    if cfg["pk"] and cfg["pk"] in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[cfg["pk"]], keep="first")
        dupes = before - len(df)
        if dupes:
            log.warning(f"  ↳ Removed {dupes} duplicate {cfg['pk']} rows")

    # Write to MySQL
    
    df.to_sql(
    name=cfg["table"],
    con=engine,
    if_exists="append",
    index=False,
    chunksize=100,

     )

    log.info(f"  ✓ Loaded {len(df):,} rows into {cfg['table']}\n")
    return {"table": cfg["table"], "loaded": len(df), "status": "OK"}


def validate(engine):
    """Post-load validation: row counts + null checks on critical columns."""
    log.info("=" * 60)
    log.info("VALIDATION")
    log.info("=" * 60)

    checks = {
        "olist_orders":      ("order_id",   99_000),
        "olist_order_items": ("order_id",  110_000),
        "olist_customers":   ("customer_id", 95_000),
        "olist_sellers":     ("seller_id",    3_000),
        "olist_products":    ("product_id",  32_000),
    }

    all_passed = True
    with engine.connect() as conn:
        for table, (pk_col, min_rows) in checks.items():
            row = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            status = "✓" if row >= min_rows else "⚠"
            if row < min_rows:
                all_passed = False
            log.info(f"  {status}  {table:<35} {row:>8,} rows  (min expected: {min_rows:,})")

        # Null check on order timestamps
        nulls = conn.execute(
            text("SELECT COUNT(*) FROM olist_orders WHERE order_purchase_timestamp IS NULL")
        ).scalar()
        log.info(f"\n  Null order_purchase_timestamp: {nulls}")

    return all_passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("SUPPLY CHAIN ETL PIPELINE")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    engine = create_engine(DB_URL, echo=False)

    # Verify DB connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("✓ MySQL connection established\n")
    except Exception as e:
        log.error(f"✗ Cannot connect to MySQL: {e}")
        log.error("  Check your .env credentials and that MySQL is running.")
        sys.exit(1)

    results = []
    for cfg in TABLES:
        try:
            result = load_table(cfg, engine)
            results.append(result)
        except Exception as e:
            log.error(f"  ✗ Failed to load {cfg['table']}: {e}")
            results.append({"table": cfg["table"], "loaded": 0, "status": "ERROR"})

    # Summary
    log.info("=" * 60)
    log.info("LOAD SUMMARY")
    log.info("=" * 60)
    total = 0
    for r in results:
        icon = "✓" if r["status"] == "OK" else "✗"
        log.info(f"  {icon}  {r['table']:<40} {r['loaded']:>8,}  [{r['status']}]")
        total += r["loaded"]
    log.info(f"\n  Total rows loaded: {total:,}")

    # Validate
    passed = validate(engine)
    if passed:
        log.info("\n ETL complete — all validations passed!")
    else:
        log.warning("\n  ETL complete — some validation thresholds not met. Check row counts above.")


if __name__ == "__main__":
    main()
