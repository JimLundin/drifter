# Drifter

A lightweight schema evolution tracking library for Polars DataFrames.

## Overview

Drifter helps you track schema changes in your DataFrames across different runs of your data pipeline. It automatically detects:
- Added columns
- Removed columns
- Type changes

## Installation

```bash
pip install drifter
```

## Usage

```python
import polars as pl
from drifter import register

# Create a DataFrame
df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"]
})

# Register the schema
if changes := register(df, "users"):
    print("Schema changes detected!")
    if changes.added_columns:
        print(f"New columns: {changes.added_columns}")
    if changes.removed_columns:
        print(f"Removed columns: {changes.removed_columns}")
    if changes.type_changes:
        print("Type changes:")
        for change in changes.type_changes:
            print(f"  {change.name}: {change.old_type} -> {change.new_type}")

# Later, with schema changes...
df_updated = pl.DataFrame({
    "id": ["1", "2", "3"],  # Type changed from Int64 to String
    "name": ["Alice", "Bob", "Charlie"],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com"]  # New column
})

if changes := register(df_updated, "users"):
    print("Schema changes detected!")
    # Handle changes...
```

## How It Works

Drifter stores schema history in JSON files under a `.drifter` directory (configurable). Each DataFrame's schema history is stored in a separate file based on the name provided to `register()`.

The schema history tracks:
- Column names
- Column types
- Column nullability

When you register a DataFrame, Drifter compares its schema against the most recent schema in the history and returns any detected changes.

## Advanced Usage

You can create your own `SchemaRegistry` instance if you want to customize the storage location:

```python
from drifter import SchemaRegistry

registry = SchemaRegistry(schema_dir="path/to/schema/storage")
changes = registry.register(df, "users")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.