"""Schema definition and change detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import polars as pl

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


class DType(StrEnum):
    """Base data types supported by the schema."""

    STRING = auto()
    INT = auto()
    UINT = auto()
    FLOAT = auto()
    BOOL = auto()
    DATE = auto()
    DATETIME = auto()
    LIST = auto()
    STRUCT = auto()


class BaseField(TypedDict):
    """Base schema field with common attributes."""

    name: str
    dtype: DType
    type_hash: int  # Hash of the original Polars type


class PrimitiveField(BaseField):
    """Field for primitive types (no additional attributes)."""


class NumericField(BaseField):
    """Field for numeric types with bit width."""

    bit_width: int


class ListField(BaseField):
    """Field for list types with inner type."""

    inner: Field  # Forward reference


class ArrayField(BaseField):
    """Field for array types with inner type."""

    inner: Field
    length: int


class StructField(BaseField):
    """Field for struct types with field definitions."""

    fields: dict[str, Field]  # Forward reference


# Union type for all possible field types
Field = PrimitiveField | NumericField | ListField | StructField


class Schema(TypedDict):
    """The complete schema definition."""

    version: str
    fields: dict[str, Field]


@dataclass
class ColumnChange:
    """Represents a change in a column's schema."""

    field: str
    old: Field
    new: Field


@dataclass
class SchemaChanges:
    """Contains all detected schema changes between versions."""

    added: dict[str, Field]  # Added field names
    removed: dict[str, Field]  # Removed field names
    changed: dict[str, ColumnChange]  # Changed field types

    def has_changes(self) -> bool:
        """Check if there are any schema changes."""
        return bool(self.added or self.removed or self.changed)

    def __bool__(self) -> bool:
        """Boolean context - Check if there are any schema changes."""
        return self.has_changes()


def _get_base_dtype(dtype: PolarsDataType) -> tuple[DType, int | None]:
    """Get the base dtype and bit width from a Polars type."""
    # Map base types to our DType enum
    base_type_map = {
        pl.Utf8: (DType.STRING, None),
        pl.String: (DType.STRING, None),
        pl.Boolean: (DType.BOOL, None),
        pl.Date: (DType.DATE, None),
        pl.Datetime: (DType.DATETIME, None),
        # Integer types
        pl.Int8: (DType.INT, 8),
        pl.Int16: (DType.INT, 16),
        pl.Int32: (DType.INT, 32),
        pl.Int64: (DType.INT, 64),
        pl.UInt8: (DType.UINT, 8),
        pl.UInt16: (DType.UINT, 16),
        pl.UInt32: (DType.UINT, 32),
        pl.UInt64: (DType.UINT, 64),
        # Float types
        pl.Float32: (DType.FLOAT, 32),
        pl.Float64: (DType.FLOAT, 64),
    }

    return base_type_map.get(dtype.base_type(), (DType.STRING, None))


def _polars_type_to_field(name: str, dtype: PolarsDataType) -> Field:
    """Convert a Polars type to our schema type system."""
    # Handle lists
    if isinstance(dtype, pl.List):
        return ListField(
            name=name,
            dtype=DType.LIST,
            inner=_polars_type_to_field("", dtype.inner),
            type_hash=hash(dtype),
        )

    # Handle arrays
    if isinstance(dtype, pl.Array):
        return ArrayField(
            name=name,
            dtype=DType.LIST,
            inner=_polars_type_to_field("", dtype.inner),
            length=dtype.width,
            type_hash=hash(dtype),
        )

    # Handle structs
    if isinstance(dtype, pl.Struct):
        return StructField(
            name=name,
            dtype=DType.STRUCT,
            fields={
                field.name: _polars_type_to_field(field.name, field.dtype)
                for field in dtype.fields
            },
            type_hash=hash(dtype),
        )

    # Get base type and bit width
    base_type, bit_width = _get_base_dtype(dtype)

    # Create appropriate field type
    if bit_width is not None:
        return NumericField(
            name=name,
            dtype=base_type,
            bit_width=bit_width,
            type_hash=hash(dtype),
        )

    return PrimitiveField(
        name=name,
        dtype=base_type,
        type_hash=hash(dtype),
    )


def _are_fields_equal(field1: Field, field2: Field) -> bool:
    """Compare two fields for equality using type hashes."""
    return field1["type_hash"] == field2["type_hash"]


def _create_schema(df: pl.DataFrame) -> Schema:
    """Create a schema from a DataFrame."""
    return {
        "version": "1.0",
        "fields": {
            name: _polars_type_to_field(name, dtype)
            for name, dtype in zip(df.columns, df.dtypes, strict=True)
        },
    }


def register(
    df: pl.DataFrame,
    name: str,
    schema_dir: str | Path = ".drifter",
) -> SchemaChanges:
    """Register a DataFrame's schema and detect any changes.

    Args:
        df: DataFrame to register schema for
        name: Unique identifier for this DataFrame's schema
        schema_dir: Directory to store schema history (default: .drifter)

    Returns:
        SchemaChanges object containing any detected changes
    """
    schema_dir = Path(schema_dir)
    schema_dir.mkdir(exist_ok=True)
    schema_file = schema_dir / f"{name}_schema_history.json"

    # Create current schema
    current_schema = _create_schema(df)

    # Load history
    if schema_file.exists():
        with schema_file.open() as f:
            history: list[Schema] = json.load(f)

        # Detect changes from previous schema
        prev_schema = history[-1]

        # Find added and removed fields
        added = {
            name: current_schema["fields"][name]
            for name in set(current_schema["fields"]) - set(prev_schema["fields"])
        }
        removed = {
            name: prev_schema["fields"][name]
            for name in set(prev_schema["fields"]) - set(current_schema["fields"])
        }

        # Find changed fields
        changed = {
            name: ColumnChange(
                field=name,
                old=prev_schema["fields"][name],
                new=current_schema["fields"][name],
            )
            for name in set(current_schema["fields"]) & set(prev_schema["fields"])
            if not _are_fields_equal(
                current_schema["fields"][name], prev_schema["fields"][name]
            )
        }

        changes = SchemaChanges(added, removed, changed)
    else:
        # First registration
        history = []
        changes = SchemaChanges(
            added=current_schema["fields"],
            removed={},
            changed={},
        )

    # Update history
    history.append(current_schema)
    with schema_file.open("w") as f:
        json.dump(history, f, indent=2)

    return changes
