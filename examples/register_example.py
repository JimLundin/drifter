"""Example demonstrating schema evolution tracking with drifter."""

import polars as pl

from drifter import register


def main() -> None:
    """Run the example."""
    # First run - initial schema registration
    test_df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        },
    )

    if changes := register(test_df, "users"):
        print("Initial schema registration:")  # noqa: T201
        print(f"Added fields: {changes.added}")  # noqa: T201
    else:
        print("No changes detected.")  # noqa: T201

    # Second run - with schema changes
    test_df_updated = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": ["25", "30", "35"],  # Changed type from Int64 to Utf8
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
        },
    )

    if changes := register(test_df_updated, "users"):
        print("\nSchema changes detected:")  # noqa: T201
        if changes.added:
            print(f"Added fields: {changes.added}")  # noqa: T201
        if changes.removed:
            print(f"Removed fields: {changes.removed}")  # noqa: T201
        if changes.changed:
            print(f"Changed fields: {changes.changed}")  # noqa: T201
    else:
        print("No changes detected.")  # noqa: T201


if __name__ == "__main__":
    main()
