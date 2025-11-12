"""Configuration management for Sor Juana Downloader."""

import os
from pathlib import Path
from dbos import DBOSConfig

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "sor_juana.duckdb"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# DBOS Configuration
DBOS_CONFIG: DBOSConfig = {
    "name": "sor-juana-downloader",
    "system_database_url": os.environ.get("DBOS_SYSTEM_DATABASE_URL"),
}

# Download settings
WIKISOURCE_PAGE_LIMIT = 50
REQUEST_TIMEOUT = 30

# Deduplication settings
MINHASH_THRESHOLD = 0.8
MINHASH_NUM_PERM = 128
