"""Tests for schema functionality."""

import tempfile
from pathlib import Path

import polars as pl

from drifter.schema import (
    DType,
    _are_fields_equal,
    _polars_type_to_field,
    register,
)


def test_primitive_type_conversion() -> None:
    """Test conversion of primitive Polars types."""
    cases = [
        (pl.String(), DType.STRING),
        (pl.Utf8(), DType.STRING),
        (pl.Boolean(), DType.BOOL),
        (pl.Binary(), DType.BINARY),
        (pl.Null(), DType.NULL),
        (pl.Object(), DType.OBJECT),
        (pl.Unknown(), DType.UNKNOWN),
    ]
    for dtype, expected_type in cases:
        field = _polars_type_to_field("test", dtype)
        assert field["dtype"] == expected_type
        assert field["name"] == "test"
        assert isinstance(field["type_hash"], int)


def test_categorical_type_conversion() -> None:
    """Test conversion of categorical types."""
    cat_type = pl.Categorical()
    field = _polars_type_to_field("test", cat_type)
    assert field["dtype"] == DType.CATEGORICAL
    assert field["name"] == "test"
    assert isinstance(field["type_hash"], int)
    assert field["ordering"] == cat_type.ordering


def test_numeric_type_conversion() -> None:
    """Test conversion of numeric Polars types."""
    cases = [
        (pl.Int8(), DType.INT, 8),
        (pl.Int16(), DType.INT, 16),
        (pl.Int32(), DType.INT, 32),
        (pl.Int64(), DType.INT, 64),
        (pl.Int128(), DType.INT, 128),
        (pl.UInt8(), DType.UINT, 8),
        (pl.UInt16(), DType.UINT, 16),
        (pl.UInt32(), DType.UINT, 32),
        (pl.UInt64(), DType.UINT, 64),
        (pl.Float32(), DType.FLOAT, 32),
        (pl.Float64(), DType.FLOAT, 64),
    ]
    for dtype, expected_type, expected_width in cases:
        field = _polars_type_to_field("test", dtype)
        assert field["dtype"] == expected_type
        assert field["bit_width"] == expected_width
        assert field["name"] == "test"
        assert isinstance(field["type_hash"], int)


def test_temporal_type_conversion() -> None:
    """Test conversion of temporal types."""
    # Test date type
    date_field = _polars_type_to_field("test", pl.Date())
    assert date_field["dtype"] == DType.DATE
    assert "time_unit" not in date_field

    # Test time type
    time_field = _polars_type_to_field("test", pl.Time())
    assert time_field["dtype"] == DType.TIME
    assert "time_unit" not in time_field

    # Test datetime type
    datetime_field = _polars_type_to_field("test", pl.Datetime())
    assert datetime_field["dtype"] == DType.DATETIME
    assert datetime_field["time_unit"] == datetime_field["time_unit"]
    assert datetime_field["timezone"] is None

    # Test duration type
    duration_field = _polars_type_to_field("test", pl.Duration())
    assert duration_field["dtype"] == DType.DURATION
    assert duration_field["time_unit"] == duration_field["time_unit"]


def test_list_type_conversion() -> None:
    """Test conversion of list types."""
    # Test list of primitives
    list_type = pl.List(pl.Int32())
    field = _polars_type_to_field("test", list_type)
    assert field["dtype"] == DType.LIST
    assert field["name"] == "test"
    assert isinstance(field["type_hash"], int)
    assert field["inner"]["dtype"] == DType.INT
    assert field["inner"]["bit_width"] == 32

    # Test nested lists
    nested_list = pl.List(pl.List(pl.String()))
    field = _polars_type_to_field("test", nested_list)
    assert field["dtype"] == DType.LIST
    assert field["inner"]["dtype"] == DType.LIST
    assert field["inner"]["inner"]["dtype"] == DType.STRING


def test_array_type_conversion() -> None:
    """Test conversion of array types."""
    array_type = pl.Array(pl.Int32(), 5)
    field = _polars_type_to_field("test", array_type)
    assert field["dtype"] == DType.ARRAY
    assert field["name"] == "test"
    assert isinstance(field["type_hash"], int)
    assert field["inner"]["dtype"] == DType.INT
    assert field["inner"]["bit_width"] == 32
    assert field["size"] == 5


def test_struct_type_conversion() -> None:
    """Test conversion of struct types."""
    struct_type = pl.Struct(
        [
            pl.Field("a", pl.Int32()),
            pl.Field("b", pl.String()),
            pl.Field("c", pl.List(pl.Float64())),
        ],
    )
    field = _polars_type_to_field("test", struct_type)
    assert field["dtype"] == DType.STRUCT
    assert field["name"] == "test"
    assert isinstance(field["type_hash"], int)
    assert field["fields"]["a"]["dtype"] == DType.INT
    assert field["fields"]["a"]["bit_width"] == 32
    assert field["fields"]["b"]["dtype"] == DType.STRING
    assert field["fields"]["c"]["dtype"] == DType.LIST
    assert field["fields"]["c"]["inner"]["dtype"] == DType.FLOAT
    assert field["fields"]["c"]["inner"]["bit_width"] == 64


def test_field_equality() -> None:
    """Test field equality comparison."""
    # Same types should be equal
    field1 = _polars_type_to_field("test", pl.Int32())
    field2 = _polars_type_to_field("test", pl.Int32())
    assert _are_fields_equal(field1, field2)

    # Different types should not be equal
    field3 = _polars_type_to_field("test", pl.Int64())
    assert not _are_fields_equal(field1, field3)

    # Complex types
    list1 = _polars_type_to_field("test", pl.List(pl.Int32()))
    list2 = _polars_type_to_field("test", pl.List(pl.Int32()))
    list3 = _polars_type_to_field("test", pl.List(pl.Int64()))
    assert _are_fields_equal(list1, list2)
    assert not _are_fields_equal(list1, list3)


def test_register_new_schema() -> None:
    """Test registering a new schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "str_col": ["a", "b", "c"],
                "list_col": [[1], [2], [3]],
            },
        )
        changes = register(test_df, "test", tmpdir)

        # First registration should show all fields as added
        assert len(changes.added) == 3
        assert not changes.removed
        assert not changes.changed

        # Check that history file was created
        history_file = Path(tmpdir) / "test_schema_history.json"
        assert history_file.exists()


def test_register_schema_changes() -> None:
    """Test detecting schema changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial schema
        df1 = pl.DataFrame(
            {
                "unchanged": [1, 2, 3],
                "will_change": ["a", "b", "c"],
                "will_remove": [1.0, 2.0, 3.0],
            },
        )
        changes = register(df1, "test", tmpdir)
        assert changes  # Should have changes on first registration
        assert len(changes.added) == 3  # All fields should be added
        assert not changes.removed
        assert not changes.changed

        # No changes
        changes = register(df1, "test", tmpdir)
        assert not changes  # Should have no changes for same schema

        # Schema changes
        df2 = pl.DataFrame(
            {
                "unchanged": [1, 2, 3],
                "will_change": [1, 2, 3],  # Changed type from String to Int64
                "will_add": ["x", "y", "z"],  # New field
            },
        )
        changes = register(df2, "test", tmpdir)
        assert changes  # Should have changes
        assert len(changes.added) == 1  # One field added
        assert len(changes.removed) == 1  # One field removed
        assert len(changes.changed) == 1  # One field changed type
        assert "will_add" in changes.added
        assert "will_remove" in changes.removed
        assert "will_change" in changes.changed


def test_register_complex_changes() -> None:
    """Test schema changes with complex types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial schema with complex types
        df1 = pl.DataFrame(
            {
                "list_col": [[1], [2], [3]],
                "struct_col": [{"a": 1}, {"a": 2}, {"a": 3}],
            },
        )
        changes = register(df1, "test", tmpdir)
        assert changes  # Should have changes on first registration
        assert len(changes.added) == 2  # Both fields should be added
        assert not changes.removed
        assert not changes.changed

        # Change list inner type and struct field type
        df2 = pl.DataFrame(
            {
                "list_col": [["1"], ["2"], ["3"]],  # Changed inner type to String
                "struct_col": [{"a": "1"}, {"a": "2"}, {"a": "3"}],  # Changed to String
            },
        )
        changes = register(df2, "test", tmpdir)
        assert changes  # Should have changes
        assert not changes.added  # No fields added
        assert not changes.removed  # No fields removed
        assert len(changes.changed) == 2  # Both fields changed type
