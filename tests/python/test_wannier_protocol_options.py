"""tests/python/test_wannier_protocol_options.py — what a caller may choose, refused before anything runs.

The protocol's first expensive step is an SCF. A choice the engine would refuse
an hour later -- a density-of-states mesh of two numbers, an export format it
does not write -- is refused when the options are built, in the same words.
"""

import pytest
from q3.workflows.wannier import AcceptGate, ProtocolOptions
from q3.workflows.wannier.settings import ACCEPT_ETA_EV, ACCEPT_ETA_MAXIMUM_EV


def test_the_defaults_are_the_protocols_own():
    options = ProtocolOptions()
    assert options.grid is None and options.dos_mesh is None
    assert options.export_formats == ()
    assert options.gate == AcceptGate(
        eta_ev=ACCEPT_ETA_EV,
        eta_maximum_ev=ACCEPT_ETA_MAXIMUM_EV,
        energy_offset_ev=options.energy_offset_ev,
        smearing_ev=options.smearing_ev,
    )


def test_a_mesh_is_three_positive_integers_whichever_mesh_it_is():
    assert ProtocolOptions(grid=[4, 4, 4], dos_mesh=(24, 24, 24)).grid == (4, 4, 4)
    with pytest.raises(ValueError, match="grid must be three positive integers"):
        ProtocolOptions(grid=(4, 0, 4))
    with pytest.raises(ValueError, match="dos_mesh must be three positive integers"):
        ProtocolOptions(dos_mesh=(24, 24))


def test_an_export_format_the_engine_does_not_write_is_refused():
    assert ProtocolOptions(export_formats=("wannier90",)).export_formats == ("wannier90",)
    with pytest.raises(ValueError, match="wannier90"):
        ProtocolOptions(export_formats=("wannier91",))


def test_the_options_are_a_value():
    options = ProtocolOptions(maximum_mesh_refinements=0)
    with pytest.raises(ValueError, match="frozen"):
        options.maximum_mesh_refinements = 2
