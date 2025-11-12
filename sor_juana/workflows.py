"""DBOS workflows for orchestrating downloads and processing."""

from typing import List, Dict, Any
from dbos import DBOS, Queue
from .config import DBOS_CONFIG
from .database import CorpusDatabase
from .downloaders import GutenbergDownloader, WikisourceDownloader, BVMCDownloader
from .processing import TextDeduplicator

# Queue for concurrent downloads
queue = Queue("download_queue")


@DBOS.step()
def download_gutenberg_step() -> List[Dict[str, Any]]:
    """DBOS step for downloading from Gutenberg."""
    downloader = GutenbergDownloader()
    return downloader.download()


@DBOS.step()
def download_wikisource_step() -> List[Dict[str, Any]]:
    """DBOS step for downloading from Wikisource."""
    downloader = WikisourceDownloader()
    return downloader.download()


@DBOS.step()
def download_bvmc_step() -> List[Dict[str, Any]]:
    """DBOS step for downloading from BVMC."""
    downloader = BVMCDownloader()
    return downloader.download()


@DBOS.step()
def store_corpus_step(items: List[Dict[str, Any]]) -> int:
    """DBOS step for storing corpus in DuckDB."""
    with CorpusDatabase() as db:
        db.insert_many(items)
        return db.count()


@DBOS.step()
def deduplicate_corpus_step() -> int:
    """DBOS step for deduplicating corpus."""
    with CorpusDatabase() as db:
        texts = db.get_all_texts()
        deduplicator = TextDeduplicator()
        duplicate_ids = deduplicator.deduplicate_texts(texts)

        if duplicate_ids:
            db.delete_many(duplicate_ids)

        return len(duplicate_ids)


@DBOS.workflow()
def build_corpus_workflow() -> Dict[str, Any]:
    """
    Main DBOS workflow to build the Sor Juana corpus.

    Returns:
        Dictionary with workflow results
    """
    # Enqueue concurrent downloads
    gutenberg_handle = queue.enqueue(download_gutenberg_step)
    wikisource_handle = queue.enqueue(download_wikisource_step)
    bvmc_handle = queue.enqueue(download_bvmc_step)

    # Wait for completion
    gutenberg_data = gutenberg_handle.get_result()
    wikisource_data = wikisource_handle.get_result()
    bvmc_data = bvmc_handle.get_result()

    # Combine all items
    all_items = gutenberg_data + wikisource_data + bvmc_data

    # Store in database
    store_handle = queue.enqueue(store_corpus_step, all_items)
    total_stored = store_handle.get_result()

    # Deduplicate
    dedupe_handle = queue.enqueue(deduplicate_corpus_step)
    duplicates_removed = dedupe_handle.get_result()

    return {
        "total_downloaded": len(all_items),
        "total_stored": total_stored,
        "duplicates_removed": duplicates_removed,
        "final_count": total_stored - duplicates_removed,
    }


def init_dbos() -> DBOS:
    """Initialize and launch DBOS."""
    dbos = DBOS(config=DBOS_CONFIG)
    dbos.launch()
    return dbos


def shutdown_dbos(dbos: DBOS) -> None:
    """Shutdown DBOS."""
    dbos.shutdown()
