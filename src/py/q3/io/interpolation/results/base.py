"""q3.io.interpolation.results.base — the hand-written base of the generated interpolation results model.

What the generated model cannot say for itself (``schemas/manifest.yaml``
attaches this base by schema ``$id``): reading and writing the file, as every
results file does, and making the ``quality`` block a driver annotates.
"""

from __future__ import annotations

import typing

from q3.io.results.base import ResultsDocument


class InterpolationResultsDocument(ResultsDocument):
    """The results file of ``q3_utility interpolate``, which the Wannier workflow annotates."""

    def quality_block(self):
        """``quality``, the block a driver annotates; made where absent."""
        if getattr(self, "quality", None) is None:
            annotation = type(self).model_fields["quality"].annotation
            (block,) = [kind for kind in typing.get_args(annotation) if kind is not type(None)]
            self.quality = block()
        return self.quality
