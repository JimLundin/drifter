"""Tests for schema registration functionality."""

import json
from datetime import UTC, datetime, tzinfo
from pathlib import Path

import polars as pl
import pytest

from drifter import register


@pytest.fixture
def schema_dir(tmp_path: Path) -> Path:
    """Create a temporary schema directory."""
    return tmp_path


@pytest.fixture
def test_df() -> pl.DataFrame:
    """Create a test DataFrame."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        },
    )


@pytest.fixture
def updated_df() -> pl.DataFrame:
    """Create an updated test DataFrame with schema changes."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": ["25", "30", "35"],  # Changed type
            "email": [
                "alice@example.com",
                "bob@example.com",
                "charlie@example.com",
            ],  # Added
        },
    )


def test_initial_registration(
    test_df: pl.DataFrame,
    schema_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test initial schema registration."""
    # Mock datetime.now to return a fixed timestamp
    fixed_dt = datetime(2025, 1, 1, tzinfo=UTC)

    class MockDatetime:
        UTC = UTC

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return fixed_dt

    monkeypatch.setattr("drifter.datetime", MockDatetime)

    # Change to temp directory
    monkeypatch.chdir(schema_dir)

    changes = register(test_df, "users")

    # Verify changes
    assert changes.added == {"id", "name", "age"}
    assert not changes.removed
    assert not changes.changed

    # Verify schema file
    schema_file = schema_dir / ".drifter" / "users.json"
    assert schema_file.exists()
    history = json.loads(schema_file.read_text())
    assert len(history) == 1
    assert history[0]["timestamp"] == "2025-01-01T00:00:00+00:00"
    assert set(history[0]["schema"]["fields"]) == {"id", "name", "age"}


def test_schema_changes(
    test_df: pl.DataFrame,
    updated_df: pl.DataFrame,
    schema_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test schema changes detection."""
    # Mock datetime.now to return fixed timestamps
    timestamps = iter(
        [
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        ],
    )

    class MockDatetime:
        UTC = UTC

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return next(timestamps)

    monkeypatch.setattr("drifter.datetime", MockDatetime)

    # Change to temp directory
    monkeypatch.chdir(schema_dir)

    # Initial registration
    register(test_df, "users")

    # Update schema
    changes = register(updated_df, "users")

    # Verify changes
    assert changes.added == {"email"}
    assert not changes.removed
    assert changes.changed == {"age"}

    # Verify schema history
    schema_file = schema_dir / ".drifter" / "users.json"
    history = json.loads(schema_file.read_text())
    assert len(history) == 2
    assert history[0]["timestamp"] == "2025-01-01T00:00:00+00:00"
    assert history[1]["timestamp"] == "2025-01-02T00:00:00+00:00"
    assert set(history[1]["schema"]["fields"]) == {"id", "name", "age", "email"}


def test_no_changes(
    test_df: pl.DataFrame,
    schema_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test registering the same schema twice."""
    # Mock datetime.now to return a fixed timestamp
    fixed_dt = datetime(2025, 1, 1, tzinfo=UTC)

    class MockDatetime:
        UTC = UTC

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return fixed_dt

    monkeypatch.setattr("drifter.datetime", MockDatetime)

    # Change to temp directory
    monkeypatch.chdir(schema_dir)

    # Initial registration
    register(test_df, "users")

    # Register same schema again
    changes = register(test_df, "users")

    # Verify no changes
    assert not changes.added
    assert not changes.removed
    assert not changes.changed

    # Verify schema history (should not have changed)
    schema_file = schema_dir / ".drifter" / "users.json"
    history = json.loads(schema_file.read_text())
    assert len(history) == 1
    assert history[0]["timestamp"] == "2025-01-01T00:00:00+00:00"


def test_schema_removal(
    test_df: pl.DataFrame,
    schema_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test removing fields from schema."""
    # Mock datetime.now to return fixed timestamps
    timestamps = iter(
        [
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        ],
    )

    class MockDatetime:
        UTC = UTC

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return next(timestamps)

    monkeypatch.setattr("drifter.datetime", MockDatetime)

    # Change to temp directory
    monkeypatch.chdir(schema_dir)

    # Initial registration
    register(test_df, "users")

    # Remove a field
    df_with_removed = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            # age field removed
        },
    )
    changes = register(df_with_removed, "users")

    # Verify changes
    assert not changes.added
    assert changes.removed == {"age"}
    assert not changes.changed

    # Verify schema history
    schema_file = schema_dir / ".drifter" / "users.json"
    history = json.loads(schema_file.read_text())
    assert len(history) == 2
    assert history[0]["timestamp"] == "2025-01-01T00:00:00+00:00"
    assert history[1]["timestamp"] == "2025-01-02T00:00:00+00:00"
    assert set(history[1]["schema"]["fields"]) == {"id", "name"}


def test_corrupted_schema_file(
    test_df: pl.DataFrame,
    schema_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test handling of corrupted schema file."""
    # Mock datetime.now to return a fixed timestamp
    fixed_dt = datetime(2025, 1, 1, tzinfo=UTC)

    class MockDatetime:
        UTC = UTC

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return fixed_dt

    monkeypatch.setattr("drifter.datetime", MockDatetime)

    # Change to temp directory
    monkeypatch.chdir(schema_dir)

    # Create corrupted schema file
    schema_file = schema_dir / ".drifter" / "users.json"
    schema_file.parent.mkdir(exist_ok=True)
    schema_file.write_text("invalid json")

    # Should handle corrupted file and treat as initial registration
    changes = register(test_df, "users")

    # Verify changes
    assert changes.added == {"id", "name", "age"}
    assert not changes.removed
    assert not changes.changed

    # Verify schema file was recreated
    assert schema_file.exists()
    history = json.loads(schema_file.read_text())
    assert len(history) == 1
    assert history[0]["timestamp"] == "2025-01-01T00:00:00+00:00"
