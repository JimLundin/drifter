"""Core logic for the drifter library."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Self


@dataclass(frozen=True)
class Field:
    """Base field type."""

    nullable: bool
    type: str = field(init=False, default="field")


# Numeric Types


@dataclass(frozen=True)
class FloatField(Field):
    """Represents a floating-point number."""

    bits: Literal[32, 64]
    type: str = field(init=False, default="float")


@dataclass(frozen=True)
class IntegerField(Field):
    """Represents an integer number."""

    bits: Literal[8, 16, 32, 64]
    signed: bool
    type: str = field(init=False, default="integer")


# Temporal Types


@dataclass(frozen=True)
class DateField(Field):
    """Represents a date."""

    type: str = field(init=False, default="date")


@dataclass(frozen=True)
class DatetimeField(Field):
    """Represents a datetime."""

    type: str = field(init=False, default="datetime")


@dataclass(frozen=True)
class DurationField(Field):
    """Represents a duration."""

    type: str = field(init=False, default="duration")


@dataclass(frozen=True)
class TimeField(Field):
    """Represents a time."""

    type: str = field(init=False, default="time")


# Collection Types


@dataclass(frozen=True)
class SequenceField(Field):
    """Represents a sequence of values.

    If shape is None, this represents a variable-length sequence (like a List).
    If shape is provided, this represents a fixed-length sequence (like an Array)
    with the given dimensions.
    """

    inner: Field
    size: int | None
    type: str = field(init=False, default="sequence")


@dataclass(frozen=True)
class StructField(Field):
    """Represents a struct."""

    fields: Mapping[str, Field]
    type: str = field(init=False, default="struct")


# String Types


@dataclass(frozen=True)
class StringField(Field):
    """Represents a UTF-8 string."""

    type: str = field(init=False, default="string")


@dataclass(frozen=True)
class CategoricalField(Field):
    """Represents a categorical value."""

    ordered: bool = False
    type: str = field(init=False, default="categorical")


@dataclass(frozen=True)
class EnumField(Field):
    """Represents an enum."""

    variants: list[str]
    type: str = field(init=False, default="enum")


# Other Types


@dataclass(frozen=True)
class BinaryField(Field):
    """Represents a binary blob."""

    type: str = field(init=False, default="binary")


@dataclass(frozen=True)
class BooleanField(Field):
    """Represents a boolean value."""

    type: str = field(init=False, default="boolean")


@dataclass(frozen=True)
class NullField(Field):
    """Represents a null value."""

    type: str = field(init=False, default="null")


FIELD_CLASSES = {
    "binary": BinaryField,
    "boolean": BooleanField,
    "categorical": CategoricalField,
    "date": DateField,
    "datetime": DatetimeField,
    "duration": DurationField,
    "enum": EnumField,
    "float": FloatField,
    "integer": IntegerField,
    "sequence": SequenceField,
    "string": StringField,
    "struct": StructField,
    "time": TimeField,
    "null": NullField,
}


def field_from_dict(data: Mapping[str, Any]) -> Field:
    """Create a Field instance from a dictionary representation."""
    field_type = data["type"]

    if field_type == "struct":
        return StructField(
            nullable=data["nullable"],
            fields={
                name: field_from_dict(field_data)
                for name, field_data in data["fields"].items()
            },
        )
    if field_type == "sequence":
        return SequenceField(
            nullable=data["nullable"],
            inner=field_from_dict(data["inner"]),
            size=data.get("size"),
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
