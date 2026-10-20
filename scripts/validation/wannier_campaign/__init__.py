"""The acceptance campaign of the automatic Wannier protocol.

"Is it automated?" is answered with a table: a fixed, seeded roster of structures
goes through ``run_automatic_wannier_workflow`` unattended, and every row records
the mesh, the rung that fired, the band distance, the spread variance, the decay
ratio and the status. The roster is built from the eight elements Q3 ships
datasets for (the magnetic three wait for a spin-polarized capture), so the table
says whether the machinery is in the published regime on the chemistry available
here, not what a two-hundred-material study would.

The package is organised by what each part is about:

- ``settings``        every fixed choice: datasets, cutoffs, cells, the SCF, the seed;
- ``utils``           helpers with no campaign state;
- ``base_structure``  one cell of the roster before its variants, ``BaseStructure``;
- ``structures``      the base cells and the variants each one gets;
- ``case``            one structure of the roster, ``CampaignCase``;
- ``roster``          the roster itself, and how ``--only`` selects from it;
- ``manifest``        the material and SCF manifest the engine reads per case;
- ``row``             one row of the table, ``CampaignRow``;
- ``table``           the Markdown table with its caveat;
- ``runner``          every case through the workflow, resumable.

``scripts/validation/run_wannier_campaign.py`` is the command line over it. The
package imports ``q3`` like any other caller; which ``q3`` that is belongs to
whoever runs it, and the command line puts this checkout's first.
"""

from __future__ import annotations

from . import runner
from .base_structure import BaseStructure
from .case import CampaignCase
from .manifest import write_case
from .roster import campaign_roster, select_cases
from .row import CampaignRow
from .runner import run_wannier_campaign
from .settings import (
    CUTOFF_OVERRIDES_RY,
    DATASET_FILES,
    DEFAULT_DATA_DIRECTORY,
    DEFAULT_THRESHOLD_ATTEMPTS,
    FAMILIES,
    RATTLE_FRACTION,
    REPOSITORY_ROOT,
)
from .table import campaign_table

__all__ = [
    "CUTOFF_OVERRIDES_RY",
    "DATASET_FILES",
    "DEFAULT_DATA_DIRECTORY",
    "DEFAULT_THRESHOLD_ATTEMPTS",
    "FAMILIES",
    "RATTLE_FRACTION",
    "REPOSITORY_ROOT",
    "BaseStructure",
    "CampaignCase",
    "CampaignRow",
    "campaign_roster",
    "campaign_table",
    "run_wannier_campaign",
    "runner",
    "select_cases",
    "write_case",
]
