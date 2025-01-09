"""Core logic for the drifter library."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Self


@dataclass(frozen=True)
class Field:
    """Represents a column schema with type information and metadata."""

    nullable: bool
    type: str = field(init=False, default="Field")


# Numeric Types


@dataclass(frozen=True)
class FloatField(Field):
    """Represents a floating-point number."""

    bits: Literal[32, 64]
    type: Literal["Float"] = field(init=False, default="Float")


@dataclass(frozen=True)
class IntegerField(Field):
    """Represents an integer number."""

    bits: Literal[8, 16, 32, 64]
    signed: bool
    type: Literal["Integer"] = field(init=False, default="Integer")


# Temporal Types


@dataclass(frozen=True)
class DateField(Field):
    """Represents a date."""

    type: Literal["Date"] = field(init=False, default="Date")


@dataclass(frozen=True)
class DatetimeField(Field):
    """Represents a datetime."""

    type: Literal["Datetime"] = field(init=False, default="Datetime")


@dataclass(frozen=True)
class DurationField(Field):
    """Represents a duration."""

    type: Literal["Duration"] = field(init=False, default="Duration")


@dataclass(frozen=True)
class TimeField(Field):
    """Represents a time."""

    type: Literal["Time"] = field(init=False, default="Time")


# Nested Types


@dataclass(frozen=True)
class ArrayField(Field):
    """Represents a fixed-length array."""

    inner: Field
    shape: list[int] | None = None
    width: int | None = None
    type: Literal["Array"] = field(init=False, default="Array")


@dataclass(frozen=True)
class ListField(Field):
    """Represents a variable-length list."""

    inner: Field
    type: Literal["List"] = field(init=False, default="List")


@dataclass(frozen=True)
class StructField(Field):
    """Represents a struct."""

    fields: Mapping[str, Field]
    type: Literal["Struct"] = field(init=False, default="Struct")


# String Types


@dataclass(frozen=True)
class StringField(Field):
    """Represents a UTF-8 string."""

    type: Literal["String"] = field(init=False, default="String")


@dataclass(frozen=True)
class CategoricalField(Field):
    """Represents a categorical value."""

    ordered: bool = False
    type: Literal["Categorical"] = field(init=False, default="Categorical")


@dataclass(frozen=True)
class EnumField(Field):
    """Represents an enum."""

    variants: list[str]
    type: Literal["Enum"] = field(init=False, default="Enum")


# Other Types


@dataclass(frozen=True)
class BinaryField(Field):
    """Represents a binary blob."""

    type: Literal["Binary"] = field(init=False, default="Binary")


@dataclass(frozen=True)
class BooleanField(Field):
    """Represents a boolean value."""

    type: Literal["Boolean"] = field(init=False, default="Boolean")


@dataclass(frozen=True)
class NullField(Field):
    """Represents a null value."""

    type: Literal["Null"] = field(init=False, default="Null")


FIELD_CLASSES = {
    "Array": ArrayField,
    "Binary": BinaryField,
    "Boolean": BooleanField,
    "Categorical": CategoricalField,
    "Date": DateField,
    "Datetime": DatetimeField,
    "Duration": DurationField,
    "Enum": EnumField,
    "Float": FloatField,
    "Integer": IntegerField,
    "List": ListField,
    "Null": NullField,
    "String": StringField,
    "Struct": StructField,
    "Time": TimeField,
}


def field_from_dict(data: Mapping[str, Any]) -> Field:
    """Create a Field instance from a dictionary representation."""
    field_type = data["type"]

    if field_type == "List":
        return ListField(
            nullable=data["nullable"],
            inner=field_from_dict(data["inner"]),
        )
    if field_type == "Array":
        return ArrayField(
            nullable=data["nullable"],
            inner=field_from_dict(data["inner"]),
            shape=data.get("shape"),
            width=data.get("width"),
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
