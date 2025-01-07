"""Example demonstrating schema evolution tracking with drifter."""

import polars as pl

from drifter import register


def main() -> None:
    """Run the example."""
    # First run - initial schema registration
    test_df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
    })

    print("Registering initial schema...")
    if changes := register(test_df, "users"):
        print("Changes detected:", changes)
    else:
        print("No changes detected.")

    # Second run - with schema changes
    test_df_updated = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": ["25", "30", "35"],  # Changed type from Int64 to Utf8
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
    })

    print("Registering updated schema...")
    if changes := register(test_df_updated, "users"):
        print("Changes detected:", changes)
    else:
        print("No changes detected.")

if __name__ == "__main__":
    main()