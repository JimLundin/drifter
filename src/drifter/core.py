"""Core logic for the drifter library."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Self


@dataclass(frozen=True)
class Field:
    """Represents a column schema with type information and metadata."""

    nullable: bool
    type: str = field(init=False, default="Field")


@dataclass(frozen=True)
class IntegerField(Field):
    """Represents an integer column schema with type information and metadata."""

    bits: int
    signed: bool
    type: Literal["Integer"] = field(init=False, default="Integer")


@dataclass(frozen=True)
class FloatField(Field):
    """Represents a floating-point column schema with type information and metadata."""

    bits: int
    type: Literal["Float"] = field(init=False, default="Float")


@dataclass(frozen=True)
class StringField(Field):
    """Represents a string column schema with type information and metadata."""

    type: Literal["String"] = field(init=False, default="String")


@dataclass(frozen=True)
class BinaryField(Field):
    """Represents a binary column schema with type information and metadata."""

    type: Literal["Binary"] = field(init=False, default="Binary")


@dataclass(frozen=True)
class BooleanField(Field):
    """Represents a boolean column schema with type information and metadata."""

    type: Literal["Boolean"] = field(init=False, default="Boolean")


@dataclass(frozen=True)
class DateField(Field):
    """Represents a date column schema with type information and metadata."""

    type: Literal["Date"] = field(init=False, default="Date")


@dataclass(frozen=True)
class TimestampField(Field):
    """Represents a timestamp column schema with type information and metadata."""

    type: Literal["Timestamp"] = field(init=False, default="Timestamp")


@dataclass(frozen=True)
class TimeField(Field):
    """Represents a time column schema with type information and metadata."""

    type: Literal["Time"] = field(init=False, default="Time")


@dataclass(frozen=True)
class DatetimeField(Field):
    """Represents a datetime column schema with type information and metadata."""

    type: Literal["Datetime"] = field(init=False, default="Datetime")


@dataclass(frozen=True)
class ListField(Field):
    """Represents a list column schema with type information and metadata."""

    inner: Field
    type: Literal["List"] = field(init=False, default="List")


@dataclass(frozen=True)
class StructField(Field):
    """Represents a struct column schema with type information and metadata."""

    fields: dict[str, Field]
    type: Literal["Struct"] = field(init=False, default="Struct")


FIELD_CLASSES = {
    "Integer": IntegerField,
    "Float": FloatField,
    "String": StringField,
    "Boolean": BooleanField,
    "Binary": BinaryField,
    "Date": DateField,
    "Timestamp": TimestampField,
    "Time": TimeField,
    "Datetime": DatetimeField,
    "List": ListField,
    "Struct": StructField,
}


def field_from_dict(data: Mapping[str, Any]) -> Field:
    """Create a Field instance from a dictionary representation."""
    field_type = data["type"]

    if field_type == "List":
        return ListField(
            nullable=data["nullable"],
            inner=field_from_dict(data["inner"]),
        )
    if field_type == "Struct":
        return StructField(
            nullable=data["nullable"],
            fields={
                name: field_from_dict(field_data)
                for name, field_data in data["fields"].items()
            },
        )
    field_cls = FIELD_CLASSES.get(field_type)
    if field_cls is None:
        msg = f"Unknown field type: {field_type}"
        raise ValueError(msg)

    return field_cls(**{k: v for k, v in data.items() if k != "type"})


@dataclass(frozen=True)
class Schema:
    """Represents a complete schema definition."""

    fields: Mapping[str, Field]

    def to_dict(self) -> Mapping[str, Any]:
        """Serialize the Schema to a dictionary."""
        return {"fields": {name: asdict(field) for name, field in self.fields.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Create a Schema from a dictionary representation."""
        fields: Mapping[str, Field] = {}
        for name, field_data in data["fields"].items():
            fields[name] = field_from_dict(field_data)

        return cls(fields)
