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

    dataframe: str
    timestamp: str


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
    new_dataframe = dataframe.clone()
    # Get schema directory path
    schema_dir = Path(".drifter")
    schema_dir.mkdir(exist_ok=True)
    schema_file = schema_dir / f"{source_id}.json"

    # Load or initialize history
    history: list[SchemaVersion] = []
    if schema_file.exists():
        try:
            with schema_file.open() as f:
                history = json.load(f)
        except (json.JSONDecodeError, KeyError, TypeError):
            # If file is corrupted, start fresh
            history = []

    # Handle schema changes
    if not history:
        changes = SchemaChange(
            added=[
                ColumnChange(name=name, new_type=dtype)
                for name, dtype in new_dataframe.schema.items()
            ],
        )
    else:
        # Deserialize the latest version's DataFrame
        latest_version = history[-1]
        old_df_bytes = base64.b64decode(latest_version["dataframe"])
        old_dataframe = pl.DataFrame.deserialize(BytesIO(old_df_bytes))
        changes = _compare_schemas(old_dataframe, new_dataframe)

    # Add new version to history if there are changes
    if changes:
        # Serialize the new DataFrame
        serialized_df = new_dataframe.serialize()

        history.append(
            {
                "dataframe": base64.b64encode(serialized_df).decode("ascii"),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        # Save schema history
        with schema_file.open("w") as f:
            json.dump(history, f, indent=2)

    return changes


__all__ = ["SchemaChange", "register"]
