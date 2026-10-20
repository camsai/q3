"""q3.io.interpolation_settings.interpolation_settings — :class:`InterpolationSettings`, one model of the ``control.interpolation`` block."""

from __future__ import annotations

from typing import Any, Literal, Self

from mat3ra.code.entity import InMemoryEntityPydantic
from pydantic import ConfigDict


class InterpolationSettings(InMemoryEntityPydantic):
    """What both steps share: the method, and the way a block is written."""

    model_config = ConfigDict(extra="forbid")

    #: The method the block is for: the one the engine has.
    method: Literal["wannier"] = "wannier"

    def to_block(self) -> dict[str, Any]:
        """The block as the manifest carries it: every key that was set, nothing else.

        An empty list or mapping is no request either -- ``export_formats: []``
        says only what its absence says -- so it is left out too.
        """
        return {
            key: value
            for key, value in self.model_dump(exclude_none=True).items()
            if value not in ([], {})
        }

    def updated(self, **changes: Any) -> Self:
        """These settings with *changes* applied, validated again as a whole.

        ``None`` unsets a key. Unlike ``model_copy(update=...)`` this refuses a
        key the block does not have and a value it cannot hold, which is what
        keeps a misspelt threshold from evaluating the model it was meant to
        change.
        """
        return type(self).model_validate({**self.model_dump(), **changes})
