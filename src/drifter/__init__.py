"""Drifter package."""

import base64
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TypedDict

import polars as pl


@dataclass(frozen=True)
class ColumnChange:
    """Represents a change in a column's schema."""

    name: str
    old_type: pl.DataType | None = None
    new_type: pl.DataType | None = None


@dataclass(frozen=True)
class SchemaChange:
    """Represents changes between two schemas."""

    added: list[ColumnChange] = field(default_factory=list)
    removed: list[ColumnChange] = field(default_factory=list)
    changed: list[ColumnChange] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return True if there are any changes."""
        return bool(self.added or self.removed or self.changed)


class SchemaVersion(TypedDict):
    """Represents a version of a schema."""

    dataframe: str  # Base64 encoded serialized DataFrame
    timestamp: str  # ISO 8601 timestamp


def _load_history(file: Path) -> list[SchemaVersion]:
    """Load schema history from a file.

    Args:
        file: Path to the schema file.

    Returns:
        List of schema versions.

    """
    if not file.exists():
        return []

    try:
        return json.loads(file.read_text())
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def _save_history(file: Path, history: list[SchemaVersion]) -> None:
    """Save schema history to a file.

    Args:
        file: Path to save the schema file.
        history: List of schema versions to save.

    """
    file.parent.mkdir(exist_ok=True)
    file.write_text(json.dumps(history, indent=2))


def _compare_schemas(old_df: pl.DataFrame, new_df: pl.DataFrame) -> SchemaChange:
    """Compare two DataFrame schemas and return the differences.

    Args:
        old_df: The previous DataFrame.
        new_df: The new DataFrame.

    Returns:
        A SchemaChange object describing the differences between the schemas.

    """
    added = [
        ColumnChange(name=name, new_type=dtype)
        for name, dtype in new_df.schema.items()
        if name not in old_df.schema
    ]
    removed = [
        ColumnChange(name=name, old_type=dtype)
        for name, dtype in old_df.schema.items()
        if name not in new_df.schema
    ]
    changed = [
        ColumnChange(
            name=name,
            old_type=old_df.schema[name],
            new_type=new_df.schema[name],
        )
        for name in old_df.schema.keys() & new_df.schema.keys()
        if old_df.schema[name] != new_df.schema[name]
    ]

    return SchemaChange(added=added, removed=removed, changed=changed)


def register(dataframe: pl.DataFrame, source_id: str) -> SchemaChange:
    """Register a schema and track its changes.

    Args:
        dataframe: A Polars DataFrame.
        source_id: The source of the data.

    Returns:
        A SchemaChange object describing the changes if any were detected.

    """
    # Create empty DataFrame with same schema to minimize storage
    empty_df = dataframe.clone()
    schema_file = Path(".drifter") / f"{source_id}.json"
    history = _load_history(schema_file)

    # For initial registration, all columns are considered added
    if not history:
        changes = SchemaChange(
            added=[
                ColumnChange(name=name, new_type=dtype)
                for name, dtype in empty_df.schema.items()
            ],
        )
    else:
        # Compare with latest version
        latest = history[-1]
        old_df = pl.DataFrame.deserialize(
            BytesIO(base64.b64decode(latest["dataframe"])),
        )
        changes = _compare_schemas(old_df, empty_df)

    # Save new version if there are changes
    if changes:
        history.append(
            {
                "dataframe": base64.b64encode(empty_df.serialize()).decode("ascii"),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        _save_history(schema_file, history)

    return changes


__all__ = ["SchemaChange", "register"]
