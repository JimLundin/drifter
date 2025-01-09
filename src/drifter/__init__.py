"""Drifter package."""

import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import polars as pl

from drifter.core import Schema
from drifter.polars import from_polars_schema


@dataclass
class SchemaChange:
    """Represents changes between two schemas."""

    added: set[str] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)
    changed: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        """Return True if there are any changes."""
        return bool(self.added or self.removed or self.changed)


class SchemaVersion(TypedDict):
    """Represents a version of a schema."""

    schema: Mapping[str, Any]
    timestamp: str


def register(df: pl.DataFrame, table_name: str) -> SchemaChange:
    """Register a schema and track its changes.

    Args:
        df: A Polars DataFrame.
        table_name: The name of the table.

    Returns:
        A SchemaChange object describing the changes if any were detected.

    """
    # Convert DataFrame schema to our format
    fields = from_polars_schema(df)
    new_schema = Schema(fields=fields)

    # Get schema directory path
    schema_dir = Path(".drifter")
    schema_dir.mkdir(exist_ok=True)
    schema_file = schema_dir / f"{table_name}.json"

    # Load or initialize history
    history: list[SchemaVersion] = []
    if schema_file.exists():
        with schema_file.open() as f, suppress(json.JSONDecodeError):
            history = json.load(f)

    # Handle schema changes
    if not history:
        changes = SchemaChange(added=set(fields))
    else:
        old_schema = Schema.from_dict(history[-1]["schema"])
        changes = _compare_schemas(old_schema, new_schema)

    # Add new version to history
    if changes:
        history.append({
            "schema": new_schema.to_dict(),
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Save schema history
        with schema_file.open("w") as f:
            json.dump(history, f, indent=2)

    return changes


def _compare_schemas(old: Schema, new: Schema) -> SchemaChange:
    """Compare two schemas and return any changes.

    Args:
        old: The old schema.
        new: The new schema.

    Returns:
        A SchemaChange object describing the changes if any were detected.

    """
    return SchemaChange(
        added=set(new.fields) - set(old.fields),
        removed=set(old.fields) - set(new.fields),
        changed={
            k for k, v in new.fields.items() if k in old.fields and v != old.fields[k]
        },
    )


__all__ = ["SchemaChange", "register"]
