#pragma once

/**
 * MeanFieldRecord — everything a mean-field exporter needs, as plain data.
 *
 * This is the seam that turns Q3's in-process exporters into converters. The
 * BerkeleyGW writers used to read a live
 * `NormConservingPlanewaveKohnShamSCFState`, which tied them to a running SCF:
 * getting a file wrong meant re-running the calculation, and a second output
 * format meant another engine flavor and another `control.*` key. They now read
 * this record instead, and the record can be built two ways:
 *
 *   - `build_mean_field_record()` (from_scf_state.hpp) — from a live SCF state,
 *     for the in-process export; and
 *   - `read_mean_field_record()` (checkpoint.hpp) — from `<prefix>.save/`, for
 *     the standalone converter.
 *
 * Same writer code either way, so the two paths cannot drift. This mirrors how
 * Quantum ESPRESSO splits `pw.x` from `pw2bgw.x`, and is what makes a second
 * format (Yambo, HDF5 `WFN.h5`) a converter over this record rather than more
 * engine surface area.
 *
 * Everything here is plain data with one deliberate exception: wavefunction
 * coefficients come through MeanFieldBandSource. Materializing every band would
 * change the in-process path's peak memory from one band to all of them, which
 * is a real regression on large systems for no benefit -- the live source hands
 * back a pointer into the quantum system exactly as before, and the file-backed
 * source reads one k-point at a time.
 *
 * Units follow `Crystal`'s own conventions rather than a new normalized set:
 * lattice vectors and Cartesian atom positions in units of `alat`, reciprocal
 * lattice vectors in units of `tpiba` = 2*pi/alat, `alat`/`cell_volume` in
 * bohr and bohr^3, energies in Rydberg, k-points and fractional translations in
 * crystal coordinates, Miller indices as integers. Re-scaling to bohr here
 * would only force every writer to undo it -- BerkeleyGW wants exactly these
 * units. Format-specific conversions that are genuinely format-specific (BGW's
 * Ry->eV for `vxc.dat`, its `sum(kw) = 1` renormalization) stay in the writers.
 */

#include <array>
#include <memory>
#include <vector>

#include "io/mean_field_record/projector_source.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/symmetry/operation.hpp"

namespace q3 {

/**
 * Per-band plane-wave coefficients, fetched on demand.
 *
 * `coefficients()` returns a pointer to `plane_waves_per_kpoint[kpoint_index]`
 * contiguous values, valid until the next call for a different k-point (the
 * file-backed implementation caches one k-point). Callers write them out
 * immediately, which is all any exporter does.
 */
class MeanFieldBandSource {
  public:
    virtual ~MeanFieldBandSource() = default;

    /**
     * `kpoint_state_index` is the (k-point, spin) state index, matching the
     * eigenvalue/occupation row convention below -- not the physical k-point
     * index. They coincide only for a spin-unpolarized run.
     */
    [[nodiscard]] virtual const Complex* coefficients(int kpoint_state_index,
                                                      int band_index) const = 0;
};

struct [[nodiscard]] MeanFieldRecord {
    // ── Cell ─────────────────────────────────────────────────────────
    /// QE `ibrav` convention; 0 means explicit/free lattice vectors.
    int bravais_lattice_index = 0;
    Real lattice_parameter = 0.0;             // alat, bohr
    Real two_pi_over_lattice_parameter = 0.0; // tpiba, 1/bohr
    Real cell_volume = 0.0;                   // bohr^3
    Matrix3x3 lattice_vectors;                // rows are a1/a2/a3, in alat units
    Matrix3x3 reciprocal_lattice_vectors;     // rows are b1/b2/b3, in tpiba units

    // ── Atoms ────────────────────────────────────────────────────────
    /// Cartesian positions in alat units, parallel to `atomic_numbers`.
    std::vector<Vector3D> atom_positions_cartesian;
    std::vector<int> atomic_numbers;

    // ── Symmetry ─────────────────────────────────────────────────────
    /**
     * Already resolved, so a converter never needs spglib. `reciprocal_rotation`
     * is `(rotation^-1)^T`, which is what BerkeleyGW's `tnp` record is built
     * from -- see berkeleygw/header.hpp.
     */
    std::vector<SymmetryOperation> symmetry_operations;

    // ── Basis ────────────────────────────────────────────────────────
    /// Charge-density ("dense") FFT grid dimensions.
    std::array<int, 3> fft_grid_dimensions = {0, 0, 0};
    /**
     * Miller indices of the charge-density G set, in the engine's own order.
     * Only the integer indices are kept: every mean-field format writes those,
     * and none of them needs the Cartesian coordinates or |G|^2 that the
     * engine's GVector also carries. Keeping this order stable matters --
     * per-k-point G sets below index into it.
     */
    std::vector<IntegerVector3D> charge_density_miller_indices;
    bool is_gamma_only = false;
    /// Plane-wave kinetic-energy cutoffs, Rydberg (BGW's `ecutrho`/`ecutwfc`).
    Real charge_density_energy_cutoff = 0.0;
    Real wavefunction_energy_cutoff = 0.0;

    // ── k-points ─────────────────────────────────────────────────────
    /**
     * Monkhorst-Pack grid dimensions and 0/1 shift flags as requested.
     * Explicit k-point lists retain zero dimensions and shifts.
     */
    std::array<int, 3> kpoint_grid_dimensions = {0, 0, 0};
    std::array<int, 3> kpoint_grid_shifts = {0, 0, 0};
    /// Crystal (fractional) coordinates, one per physical k-point.
    std::vector<Vector3D> kpoint_coordinates;
    /// Engine-convention weights; formats renormalize as they require.
    std::vector<Real> kpoint_weights;
    std::vector<int> plane_waves_per_kpoint;
    /// Per physical k-point, indices into `charge_density_miller_indices`.
    std::vector<std::vector<int>> kpoint_g_vector_indices;

    // ── Bands ────────────────────────────────────────────────────────
    int number_of_bands = 0;
    int number_of_spin_components = 1;
    /// Indexed [kpoint_state_index][band_index]; Rydberg.
    std::vector<std::vector<Real>> eigenvalues_rydberg;
    /// Indexed [kpoint_state_index][band_index]; max 1 per (k, spin) state.
    std::vector<std::vector<Real>> occupations;
    std::shared_ptr<const MeanFieldBandSource> band_source;

    // ── Fields ───────────────────────────────────────────────────────
    /**
     * Charge density on the charge-density G set, one entry per spin channel,
     * each parallel to `charge_density_miller_indices`.
     *
     * G-space rather than real-space on purpose. The engine holds rho(r) and
     * the writers used to call `PlanewaveBasis::rho_r_to_g()` themselves --
     * a *computation*, which a converter reading files has no basis to
     * perform. Doing the transform once while the record is built keeps the
     * record purely descriptive, and matches
     * `exchange_correlation_potential_g`, which is G-space already. Not scaled
     * by the cell volume: that is BerkeleyGW's convention, applied by its
     * writer.
     */
    std::vector<std::vector<Complex>> charge_density_g;
    /**
     * Exchange-correlation potential on the charge-density G set, Rydberg,
     * parallel to `charge_density_miller_indices`. Empty when the producing run
     * did not have one; `vxc.dat` export refuses rather than inventing values.
     */
    std::vector<Complex> exchange_correlation_potential_g;
    /**
     * The local Kohn-Sham potential the Hamiltonian applied -- ionic local plus
     * Hartree plus exchange-correlation -- one entry per spin channel, Rydberg,
     * parallel to `charge_density_miller_indices`, G = 0 the cell average.
     * Empty when the producing run did not have one, which includes every
     * record read back from `mean-field.q3mf`; `VSC` export then refuses.
     */
    std::vector<std::vector<Complex>> self_consistent_potential_g;

    // ── Nonlocal pseudopotential ─────────────────────────────────────
    /**
     * Filled only when the projectors were requested
     * (`control.mean_field_export_kleinman_bylander_projectors`), and only for
     * norm-conserving datasets: then V_NL = Σ_atom |β⟩ D ⟨β| is the whole
     * nonlocal part of H. `projector_source` stays null otherwise, and the
     * `VKB` file is not written.
     */
    /// 0-based species index of each atom, parallel to `atomic_numbers`.
    std::vector<int> atom_species_indices;
    /// Projector channels -- one per (β, m) -- of each species.
    std::vector<int> projectors_per_species;
    /**
     * D_ij in Rydberg, indexed [spin][atom], each block
     * `projectors_per_species[species]` squared values with the first index
     * fastest -- the channel order of `projector_source`'s rows for that atom.
     */
    std::vector<std::vector<std::vector<Real>>> nonlocal_coefficients;
    std::shared_ptr<const MeanFieldProjectorSource> projector_source;

    [[nodiscard]] int number_of_kpoints() const {
        return static_cast<int>(kpoint_coordinates.size());
    }

    [[nodiscard]] int number_of_atoms() const {
        return static_cast<int>(atomic_numbers.size());
    }

    [[nodiscard]] int number_of_g_vectors() const {
        return static_cast<int>(charge_density_miller_indices.size());
    }

    /// (k-point, spin) state count -- the row count of eigenvalues/occupations.
    [[nodiscard]] int number_of_kpoint_states() const {
        return number_of_kpoints() * number_of_spin_components;
    }
};

} // namespace q3
