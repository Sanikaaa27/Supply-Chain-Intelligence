"""
STEP 3 — Download Olist Dataset from Kaggle
Run: python etl/download_dataset.py
Requires: KAGGLE_USERNAME and KAGGLE_KEY in your .env file
"""

import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Set Kaggle credentials ────────────────────────────────────────────────────
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
os.environ["KAGGLE_KEY"]      = os.getenv("KAGGLE_KEY", "")

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]


def download_dataset():
    """Download and extract the Olist Brazilian E-Commerce dataset from Kaggle."""
    print("📦 Downloading Olist dataset from Kaggle...")

    try:
        import kaggle  # noqa: F401  (validates credentials on import)
    except Exception as e:
        print(f"❌ Kaggle auth failed: {e}")
        print("   Make sure KAGGLE_USERNAME and KAGGLE_KEY are set in your .env file.")
        return

    zip_path = RAW_DATA_DIR / "olist.zip"

    os.system(
        f"kaggle datasets download olistbr/brazilian-ecommerce "
        f"--path {RAW_DATA_DIR} --unzip"
    )

    # Verify all expected files are present
    print("\n✅ Verifying downloaded files:")
    all_ok = True
    for fname in EXPECTED_FILES:
        fpath = RAW_DATA_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1_048_576
            print(f"   ✓ {fname:<55} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ MISSING: {fname}")
            all_ok = False

    if all_ok:
        print("\n🎉 All 9 dataset files downloaded successfully!")
        print(f"   Location: {RAW_DATA_DIR.resolve()}")
    else:
        print("\n⚠️  Some files are missing. Re-run the script or download manually.")
        print("   Manual download: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")


if __name__ == "__main__":
    download_dataset()
