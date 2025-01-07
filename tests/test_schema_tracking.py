"""Tests for schema tracking functionality."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from drifter import register, SchemaChanges


def test_initial_schema_registration():
    """Test registering a schema for the first time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        
        changes = register(df, "test", schema_dir=tmpdir)
        assert isinstance(changes, SchemaChanges)
        assert changes.has_changes()
        assert set(changes.added) == {"id", "name"}
        assert not changes.removed
        assert not changes.changed


def test_schema_changes_detection():
    """Test detecting various schema changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial schema
        df1 = pl.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "to_remove": [True, False]
        })
        register(df1, "test", schema_dir=tmpdir)
        
        # Modified schema
        df2 = pl.DataFrame({
            "id": ["1", "2"],  # Type change
            "name": ["Alice", "Bob"],
            "new_col": [1.0, 2.0]  # New column
        })
        changes = register(df2, "test", schema_dir=tmpdir)
        
        assert changes.has_changes()
        assert changes.added == ["new_col"]
        assert changes.removed == ["to_remove"]
        assert len(changes.changed) == 1
        assert changes.changed[0].name == "id"
        assert changes.changed[0].old_type == "Int64"
        assert changes.changed[0].new_type == "String"


def test_no_changes():
    """Test that no changes are detected when schema stays the same."""
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pl.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"]
        })
        
        # Register initial schema
        register(df, "test", schema_dir=tmpdir)
        
        # Register same schema again
        changes = register(df, "test", schema_dir=tmpdir)
        assert not changes.has_changes()
        assert not changes.added
        assert not changes.removed
        assert not changes.changed


def test_schema_history_persistence():
    """Test that schema history is correctly persisted to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Register initial schema
        df1 = pl.DataFrame({"id": [1]})
        register(df1, "test", schema_dir=tmpdir)
        
        # Check that history file exists
        history_file = Path(tmpdir) / "test_schema_history.json"
        assert history_file.exists()
        
        # Register changed schema
        df2 = pl.DataFrame({"id": [1], "name": ["test"]})
        changes = register(df2, "test", schema_dir=tmpdir)
        
        # Verify changes are detected using history from disk
        assert changes.has_changes()
        assert changes.added == ["name"]
