"""Bounded CSV upload parsing for the public Streamlit application."""
from __future__ import annotations

from typing import Any

import pandas as pd


class CSVUploadError(ValueError):
    """A user-correctable CSV upload size, row-count, or format problem."""


def read_limited_csv(
    uploaded_file: Any,
    *,
    max_rows: int = 5000,
    max_bytes: int = 10 * 1024 * 1024,
) -> pd.DataFrame:
    """Read at most max_rows + 1 records, rejecting files above configured limits."""
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1")

    uploaded_size = getattr(uploaded_file, "size", None)
    if uploaded_size is not None and int(uploaded_size) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise CSVUploadError(f"This file exceeds the {limit_mb:.0f} MB upload limit.")

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    try:
        # One extra row is a sentinel that detects oversized row counts without
        # loading the entire uploaded file into memory.
        frame = pd.read_csv(uploaded_file, low_memory=False, nrows=max_rows + 1)
    except pd.errors.EmptyDataError as exc:
        raise CSVUploadError("The uploaded CSV is empty or has no header row.") from exc
    except UnicodeDecodeError as exc:
        raise CSVUploadError("The uploaded CSV must use UTF-8 text encoding.") from exc
    except pd.errors.ParserError as exc:
        raise CSVUploadError("The uploaded file could not be parsed as a valid CSV.") from exc

    if frame.empty:
        raise CSVUploadError("The uploaded CSV contains a header but no data rows.")
    if len(frame) > max_rows:
        raise CSVUploadError(
            f"This CSV contains more than {max_rows:,} data rows. Split it into smaller files and upload again."
        )
    return frame
