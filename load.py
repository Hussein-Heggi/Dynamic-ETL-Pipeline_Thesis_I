"""
load.py

Load phase of the Dynamic ETL Pipeline.
Loads enriched DataFrames into SQLite database.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_to_sqlite(
    dataframes: list[pd.DataFrame],
    db_path: str = "etl_data.db",
    table_names: list[str] | None = None,
    if_exists: str = "append",
) -> dict[str, Any]:
    """
    Load enriched DataFrames into SQLite database.

    Args:
        dataframes: List of DataFrames to load into the database
        db_path: Path to SQLite database file (default: 'etl_data.db')
        table_names: Optional list of table names. If None, uses index-based naming
                    (table_0, table_1, ...). Must match length of dataframes if provided.
        if_exists: What to do if table exists: 'append', 'replace', or 'fail'
                  (default: 'append')

    Returns:
        dict: Metadata about the load operation including:
            - status: Overall status ('success', 'partial_success', 'failure')
            - db_path: Path to the database file
            - tables_loaded: Number of tables successfully loaded
            - total_rows_loaded: Total number of rows loaded
            - results: Per-table results

    Example:
        >>> enriched_dfs, _ = transform_pipeline(dfs, keywords)
        >>> metadata = load_to_sqlite(
        ...     enriched_dfs,
        ...     db_path="my_data.db",
        ...     table_names=["stock_aapl", "stock_googl"]
        ... )
        >>> print(f"Loaded {metadata['tables_loaded']} tables")

    Raises:
        ValueError: If table_names length doesn't match dataframes length
    """
    # Validate inputs
    if not dataframes:
        logger.warning("No DataFrames provided to load_to_sqlite")
        return {
            "status": "no_data",
            "db_path": db_path,
            "tables_loaded": 0,
            "total_rows_loaded": 0,
            "results": [],
        }

    # Generate table names if not provided
    if table_names is None:
        table_names = [f"table_{i}" for i in range(len(dataframes))]
        logger.info(
            f"No table names provided, using index-based names: {table_names}"
        )
    else:
        if len(table_names) != len(dataframes):
            raise ValueError(
                f"Number of table names ({len(table_names)}) must match "
                f"number of DataFrames ({len(dataframes)})"
            )

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Load Phase")
    logger.info(f"Database: {db_path}")
    logger.info(f"Tables to load: {len(dataframes)}")
    logger.info(f"Mode: {if_exists}")
    logger.info(f"{'='*60}")

    metadata = {
        "status": "success",
        "db_path": str(Path(db_path).resolve()),
        "tables_loaded": 0,
        "total_rows_loaded": 0,
        "total_errors": 0,
        "results": [],
    }

    # Create database connection
    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        metadata["status"] = "failure"
        metadata["error"] = f"Database connection failed: {str(e)}"
        return metadata

    try:
        for idx, (df, table_name) in enumerate(zip(dataframes, table_names)):
            result = {
                "index": idx,
                "table_name": table_name,
                "rows": len(df),
                "columns": len(df.columns),
                "status": "pending",
            }

            try:
                logger.info(f"\n[Table {idx + 1}/{len(dataframes)}] Loading '{table_name}'")
                logger.info(f"  DataFrame shape: {df.shape}")

                # Check if DataFrame is empty
                if df.empty:
                    logger.warning(f"  Skipping empty DataFrame for table '{table_name}'")
                    result["status"] = "skipped_empty"
                    result["rows_loaded"] = 0
                    metadata["results"].append(result)
                    continue

                # Check if table exists
                table_exists = _check_table_exists(conn, table_name)

                if table_exists:
                    # Get current row count before append
                    current_rows = _get_table_row_count(conn, table_name)
                    logger.info(
                        f"  Table '{table_name}' exists with {current_rows} rows"
                    )

                    if if_exists == "append":
                        logger.info(f"  Appending {len(df)} rows to existing table")
                    elif if_exists == "replace":
                        logger.info(f"  Replacing table with {len(df)} rows")
                    else:  # fail
                        raise ValueError(
                            f"Table '{table_name}' already exists and if_exists='fail'"
                        )
                else:
                    logger.info(f"  Creating new table '{table_name}'")

                # Load DataFrame to SQLite
                df.to_sql(
                    name=table_name,
                    con=conn,
                    if_exists=if_exists,
                    index=False,
                    method="multi",  # Faster bulk inserts
                )

                # Verify load
                new_row_count = _get_table_row_count(conn, table_name)

                if table_exists and if_exists == "append":
                    rows_added = new_row_count - current_rows
                    logger.info(
                        f"  ✓ Successfully appended {rows_added} rows "
                        f"(total: {new_row_count} rows)"
                    )
                else:
                    logger.info(f"  ✓ Successfully loaded {new_row_count} rows")

                result["status"] = "success"
                result["rows_loaded"] = new_row_count
                result["table_existed"] = table_exists
                result["action"] = if_exists if table_exists else "create"

                metadata["tables_loaded"] += 1
                metadata["total_rows_loaded"] += len(df)

            except Exception as e:
                logger.error(f"  ✗ Failed to load table '{table_name}': {e}")
                result["status"] = "error"
                result["error"] = str(e)
                result["rows_loaded"] = 0
                metadata["total_errors"] += 1
                metadata["status"] = "partial_success"

            metadata["results"].append(result)

    except Exception as e:
        logger.error(f"Unexpected error during load phase: {e}", exc_info=True)
        metadata["status"] = "failure"
        metadata["error"] = f"Unexpected error: {str(e)}"

    finally:
        # Close connection
        conn.close()
        logger.info(f"\nDatabase connection closed")

    # Update overall status
    if metadata["total_errors"] > 0:
        if metadata["total_errors"] == len(dataframes):
            metadata["status"] = "failure"
        else:
            metadata["status"] = "partial_success"

    logger.info(f"\n{'='*60}")
    logger.info("Load Phase Complete")
    logger.info(f"Status: {metadata['status']}")
    logger.info(f"Tables loaded: {metadata['tables_loaded']}/{len(dataframes)}")
    logger.info(f"Total rows loaded: {metadata['total_rows_loaded']}")
    logger.info(f"Total errors: {metadata['total_errors']}")
    logger.info(f"{'='*60}\n")

    return metadata


def load_from_transform_output(
    enriched_dataframes: list[pd.DataFrame],
    transform_metadata: dict[str, Any],
    db_path: str = "etl_data.db",
    table_names: list[str] | None = None,
    if_exists: str = "append",
) -> dict[str, Any]:
    """
    Load DataFrames from transform phase output directly into SQLite.

    This is a convenience wrapper around load_to_sqlite that integrates
    with the transform phase output.

    Args:
        enriched_dataframes: Output from transform_pipeline()
        transform_metadata: Metadata from transform_pipeline()
        db_path: Path to SQLite database file
        table_names: Optional list of table names
        if_exists: What to do if table exists

    Returns:
        dict: Combined metadata from transform and load phases

    Example:
        >>> enriched_dfs, transform_meta = transform_pipeline(dfs, keywords)
        >>> combined_meta = load_from_transform_output(
        ...     enriched_dfs, transform_meta, db_path="my_data.db"
        ... )
    """
    logger.info("Loading data from transform phase output")

    # Load to database
    load_metadata = load_to_sqlite(
        enriched_dataframes,
        db_path=db_path,
        table_names=table_names,
        if_exists=if_exists,
    )

    # Combine metadata
    combined_metadata = {
        "transform": transform_metadata,
        "load": load_metadata,
        "overall_status": _combine_status(
            transform_metadata.get("overall_status", "unknown"),
            load_metadata.get("status", "unknown"),
        ),
    }

    return combined_metadata


def _check_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    return cursor.fetchone() is not None


def _get_table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    """Get the number of rows in a table."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]


def _combine_status(transform_status: str, load_status: str) -> str:
    """Combine transform and load statuses into overall status."""
    # Priority: failure > partial_success > success
    if "failure" in [transform_status, load_status]:
        return "failure"
    elif "partial" in [transform_status, load_status]:
        return "partial_success"
    elif transform_status == "success" and load_status == "success":
        return "success"
    else:
        return "unknown"


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    sample_df1 = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 50,
            "ts": pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC"),
            "close": np.random.uniform(150, 160, 50),
            "sma_20": np.random.uniform(150, 160, 50),
            "rsi_14": np.random.uniform(30, 70, 50),
        }
    )

    sample_df2 = pd.DataFrame(
        {
            "ticker": ["GOOGL"] * 50,
            "ts": pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC"),
            "close": np.random.uniform(140, 150, 50),
            "ema_50": np.random.uniform(140, 150, 50),
        }
    )

    # Load to SQLite
    metadata = load_to_sqlite(
        [sample_df1, sample_df2],
        db_path="example_etl.db",
        table_names=["stock_aapl", "stock_googl"],
        if_exists="replace",
    )

    print(f"\nLoad Status: {metadata['status']}")
    print(f"Tables loaded: {metadata['tables_loaded']}")
    print(f"Total rows: {metadata['total_rows_loaded']}")
