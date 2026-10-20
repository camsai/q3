"""q3.flavors — Python wrappers for Q3 calculation flavors."""

from q3.flavors.non_self_consistent_field import run_nscf
from q3.flavors.self_consistent_field import run_scf

__all__ = ["run_scf", "run_nscf"]
