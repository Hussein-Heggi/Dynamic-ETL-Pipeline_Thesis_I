"""
transform.py

Complete transform phase orchestration for the Dynamic ETL Pipeline.
Handles data cleaning and feature enrichment for multiple DataFrames.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from data_cleaning import pipeline_clean
from enrichment import enrich_dataframe_from_keywords

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def transform_pipeline(
    dataframes: list[pd.DataFrame],
    keywords: list[str],
    registry_path: str = "registry.yaml",
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """
    Complete transform phase: clean and enrich multiple DataFrames.

    Args:
        dataframes: List of DataFrames to transform
        keywords: List of feature keywords to apply to all DataFrames.
                 The same features will be applied to each DataFrame.
        registry_path: Path to the feature registry YAML file

    Returns:
        tuple: (enriched_dataframes, pipeline_metadata)
            - enriched_dataframes: List of cleaned and enriched DataFrames
            - pipeline_metadata: Dictionary containing cleaning reports,
                                enrichment metadata, and error information

    Example:
        >>> df1 = pd.DataFrame(...)
        >>> df2 = pd.DataFrame(...)
        >>> keywords = ["20 day sma on close", "14 day rsi"]
        >>> enriched_dfs, metadata = transform_pipeline(
        ...     [df1, df2],
        ...     keywords
        ... )
        >>> print(f"Successfully transformed {len(enriched_dfs)} DataFrames")
    """
    # Validate inputs

    if not dataframes:
        logger.warning("Empty dataframes list provided to transform_pipeline")
        return [], {"status": "no_data", "dataframes_processed": 0}

    logger.info(f"Starting transform pipeline for {len(dataframes)} DataFrame(s)")

    enriched_dataframes = []
    pipeline_metadata = {
        "dataframes_processed": len(dataframes),
        "results": [],
        "overall_status": "success",
        "total_errors": 0,
    }

    for idx, df in enumerate(dataframes):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing DataFrame {idx + 1}/{len(dataframes)}")
        logger.info(f"{'=' * 60}")

        result = {
            "index": idx,
            "original_shape": df.shape,
            "keywords": keywords,
            "cleaning": {},
            "enrichment": {},
            "final_shape": None,
            "status": "pending",
            "errors": [],
        }

        try:
            # Phase 1: Data Cleaning
            logger.info(f"[DataFrame {idx + 1}] Phase 1: Data Cleaning")
            cleaned_df, cleaning_report = pipeline_clean(df)
            result["cleaning"] = cleaning_report
            result["cleaned_shape"] = cleaned_df.shape

            if cleaned_df.empty:
                logger.warning(
                    f"[DataFrame {idx + 1}] Cleaning resulted in empty DataFrame"
                )
                result["status"] = "empty_after_cleaning"
                result["errors"].append("DataFrame is empty after cleaning")
                enriched_dataframes.append(cleaned_df)
                pipeline_metadata["results"].append(result)
                pipeline_metadata["total_errors"] += 1
                continue

            logger.info(
                f"[DataFrame {idx + 1}] Cleaning complete: "
                f"{df.shape[0]} -> {cleaned_df.shape[0]} rows"
            )

            # Phase 2: Feature Enrichment
            logger.info(f"[DataFrame {idx + 1}] Phase 2: Feature Enrichment")
            logger.info(f"[DataFrame {idx + 1}] Keywords: {keywords}")

            if not keywords:
                logger.info(
                    f"[DataFrame {idx + 1}] No keywords provided, skipping enrichment"
                )
                enriched_df = cleaned_df
                result["enrichment"] = {
                    "status": "skipped",
                    "reason": "no_keywords",
                }
            else:
                enriched_df, enrichment_metadata = enrich_dataframe_from_keywords(
                    cleaned_df, keywords, registry_path
                )
                result["enrichment"] = enrichment_metadata

                if not enrichment_metadata.get("success", False):
                    logger.warning(
                        f"[DataFrame {idx + 1}] Enrichment failed or had errors"
                    )
                    if enrichment_metadata.get("errors"):
                        result["errors"].extend(enrichment_metadata["errors"])
                        pipeline_metadata["total_errors"] += len(
                            enrichment_metadata["errors"]
                        )
                else:
                    logger.info(
                        f"[DataFrame {idx + 1}] Enrichment complete: "
                        f"{cleaned_df.shape[1]} -> {enriched_df.shape[1]} columns"
                    )

            # Phase 3: Post-Enrichment Data Cleaning
            logger.info(f"[DataFrame {idx + 1}] Phase 3: Post-Enrichment Cleaning")
            final_df, post_cleaning_report = pipeline_clean(enriched_df)
            result["post_enrichment_cleaning"] = post_cleaning_report
            result["post_cleaned_shape"] = final_df.shape

            if final_df.empty:
                logger.warning(
                    f"[DataFrame {idx + 1}] Post-enrichment cleaning resulted in empty DataFrame"
                )
                result["status"] = "empty_after_post_cleaning"
                result["errors"].append("DataFrame is empty after post-enrichment cleaning")
                enriched_dataframes.append(final_df)
                pipeline_metadata["results"].append(result)
                pipeline_metadata["total_errors"] += 1
                continue

            logger.info(
                f"[DataFrame {idx + 1}] Post-enrichment cleaning complete: "
                f"{enriched_df.shape[0]} -> {final_df.shape[0]} rows"
            )

            result["final_shape"] = final_df.shape
            result["status"] = (
                "success"
                if enrichment_metadata.get("success", True)
                else "partial_success"
            )

            enriched_dataframes.append(final_df)

        except Exception as e:
            logger.error(
                f"[DataFrame {idx + 1}] Unexpected error during transform: {e}",
                exc_info=True,
            )
            result["status"] = "error"
            result["errors"].append(f"Unexpected error: {str(e)}")
            pipeline_metadata["total_errors"] += 1
            pipeline_metadata["overall_status"] = "partial_failure"

            # Append the original DataFrame if transformation failed
            enriched_dataframes.append(df.copy())

        finally:
            pipeline_metadata["results"].append(result)

    # Update overall status
    if pipeline_metadata["total_errors"] > 0:
        if pipeline_metadata["total_errors"] == len(dataframes):
            pipeline_metadata["overall_status"] = "failure"
        else:
            pipeline_metadata["overall_status"] = "partial_success"

    logger.info(f"\n{'=' * 60}")
    logger.info("Transform Pipeline Complete")
    logger.info(f"Status: {pipeline_metadata['overall_status']}")
    logger.info(f"DataFrames processed: {len(dataframes)}")
    logger.info(f"Total errors: {pipeline_metadata['total_errors']}")
    logger.info(f"{'=' * 60}\n")

    return enriched_dataframes, pipeline_metadata


def transform_pipeline_from_list(
    dataframes: list[pd.DataFrame],
    keywords: list[str],
    registry_path: str = "registry.yaml",
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """
    Alias for transform_pipeline for backwards compatibility.

    Args:
        dataframes: List of DataFrames to transform
        keywords: List of feature keywords to apply to all DataFrames
        registry_path: Path to the feature registry YAML file

    Returns:
        Same as transform_pipeline()

    Example:
        >>> enriched_dfs, metadata = transform_pipeline_from_list(
        ...     [df1, df2],
        ...     ["20 day sma on close", "14 day rsi"]
        ... )
    """
    return transform_pipeline(dataframes, keywords, registry_path)


def transform_single(
    df: pd.DataFrame,
    keywords: list[str],
    registry_path: str = "registry.yaml",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convenience function for transforming a single DataFrame.

    Args:
        df: DataFrame to transform
        keywords: List of feature keywords
        registry_path: Path to the feature registry YAML file

    Returns:
        tuple: (enriched_dataframe, metadata)

    Example:
        >>> df = pd.DataFrame(...)
        >>> keywords = ["20 day sma on close", "14 day rsi"]
        >>> enriched_df, metadata = transform_single(df, keywords)
    """
    enriched_dfs, pipeline_metadata = transform_pipeline(
        [df], keywords, registry_path
    )

    # Extract single result
    single_result = (
        pipeline_metadata["results"][0] if pipeline_metadata["results"] else {}
    )
    single_metadata = {
        "cleaning": single_result.get("cleaning", {}),
        "enrichment": single_result.get("enrichment", {}),
        "status": single_result.get("status", "unknown"),
        "errors": single_result.get("errors", []),
        "original_shape": single_result.get("original_shape"),
        "final_shape": single_result.get("final_shape"),
    }

    return enriched_dfs[0] if enriched_dfs else pd.DataFrame(), single_metadata


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    sample_df1 = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 100,
            "ts": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
            "open": np.random.uniform(150, 160, 100),
            "high": np.random.uniform(160, 170, 100),
            "low": np.random.uniform(140, 150, 100),
            "close": np.random.uniform(150, 160, 100),
            "volume": np.random.randint(1000000, 10000000, 100),
        }
    )

    sample_df2 = pd.DataFrame(
        {
            "ticker": ["GOOGL"] * 100,
            "ts": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
            "open": np.random.uniform(140, 150, 100),
            "high": np.random.uniform(150, 160, 100),
            "low": np.random.uniform(130, 140, 100),
            "close": np.random.uniform(140, 150, 100),
            "volume": np.random.randint(500000, 5000000, 100),
        }
    )

    # Define keywords to apply to all DataFrames
    keywords = ["20 day sma on close", "14 day rsi"]

    # Run transform pipeline
    enriched_dfs, metadata = transform_pipeline(
        [sample_df1, sample_df2], keywords
    )

    print(f"\nProcessed {len(enriched_dfs)} DataFrames")
    for i, df in enumerate(enriched_dfs):
        print(f"DataFrame {i + 1} shape: {df.shape}")
        print(f"Columns: {list(df.columns)}\n")
