from __future__ import annotations

import io

import pytest

from src.phase9_uploads import CSVUploadError, read_limited_csv


class SizedBytesIO(io.BytesIO):
    @property
    def size(self) -> int:
        return len(self.getvalue())


def test_reads_small_csv() -> None:
    uploaded = SizedBytesIO(b"a,b\n1,2\n3,4\n")
    result = read_limited_csv(uploaded, max_rows=2, max_bytes=100)
    assert result.shape == (2, 2)
    assert list(result.columns) == ["a", "b"]


def test_rejects_csv_with_more_than_maximum_rows() -> None:
    uploaded = SizedBytesIO(b"a\n1\n2\n3\n")
    with pytest.raises(CSVUploadError, match="more than 2"):
        read_limited_csv(uploaded, max_rows=2, max_bytes=100)


def test_rejects_file_larger_than_byte_limit() -> None:
    uploaded = SizedBytesIO(b"a\n12345\n")
    with pytest.raises(CSVUploadError, match="upload limit"):
        read_limited_csv(uploaded, max_rows=10, max_bytes=4)


def test_rejects_empty_csv() -> None:
    uploaded = SizedBytesIO(b"")
    with pytest.raises(CSVUploadError, match="empty"):
        read_limited_csv(uploaded)


def test_rejects_header_only_csv() -> None:
    uploaded = SizedBytesIO(b"a,b\n")
    with pytest.raises(CSVUploadError, match="no data rows"):
        read_limited_csv(uploaded)
