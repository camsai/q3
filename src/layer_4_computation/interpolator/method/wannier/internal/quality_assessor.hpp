#pragma once

/**
 * WannierQualityAssessor — measures a WannierQualityReport: the statistics of
 * the per-function spreads, and the centres against the atoms and the bonds
 * of the structure the model belongs to.
 *
 * The structure is fixed for the life of the assessor, so what depends on it
 * alone -- which pairs of atoms are bonded, and where their midpoints are --
 * is worked out once, when it is built. A bond is a pair of atoms, over
 * periodic images, within BOND_TOLERANCE of the shortest interatomic
 * distance; the image range of two covers a cell whose shortest bond crosses
 * a boundary twice, which a doubled cell with a small basis can do.
 *
 * Two outlier flags sit beside the variance of the spreads: a function more
 * than OUTLIER_RATIO times the median, and a function above
 * LARGE_SPREAD_ANGSTROM_SQUARED, which no s or p Wannier function of a
 * well-formed model reaches.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/quality_report.hpp"

namespace q3 {

class WannierQualityAssessor {
  public:
    /// A function this many times the median spread is flagged.
    static constexpr Real OUTLIER_RATIO = 5.0;
    /// A function above this spread is flagged whatever the others do.
    static constexpr Real LARGE_SPREAD_ANGSTROM_SQUARED = 10.0;
    /// Two atoms this close to the shortest interatomic distance count as bonded.
    static constexpr Real BOND_TOLERANCE = 1.2;

    /**
     * `real_lattice_bohr` has rows a1, a2, a3 in the Cartesian frame the
     * centres are in, and `atom_positions_crystal` are fractional. No atoms
     * is a structure the centres cannot be held against -- a record imported
     * from files may arrive without one -- and skips those checks rather
     * than refusing.
     */
    WannierQualityAssessor(const Matrix3x3& real_lattice_bohr,
                           std::vector<Vector3D> atom_positions_crystal)
        : cartesian_from_crystal_(real_lattice_bohr.transpose())
        , crystal_from_cartesian_(real_lattice_bohr.inverse().transpose())
        , atoms_crystal_(std::move(atom_positions_crystal)) {
        find_bonds();
    }

    /// The spreads and centres of `construction`'s final gauge, and the decay of its H(R).
    [[nodiscard]] WannierQualityReport assess(const WannierConstructionReport& construction) const {
        WannierQualityReport report;
        report.decay = construction.hamiltonian_decay;
        report.spreads_bohr_squared = construction.spread.spreads_bohr_squared;
        summarize_spreads(report);
        if (!construction.spread.centers_bohr.empty() && !atoms_crystal_.empty()) {
            locate_centers(report, construction.spread.centers_bohr);
        }
        return report;
    }

  private:
    static void summarize_spreads(WannierQualityReport& report) {
        const std::vector<Real>& spreads = report.spreads_bohr_squared;
        if (spreads.empty()) {
            return;
        }
        std::vector<Real> sorted = spreads;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t middle = sorted.size() / 2;
        report.spread_median_bohr_squared =
            sorted.size() % 2 == 1 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
        report.spread_maximum_bohr_squared = sorted.back();

        Real mean = 0.0;
        for (const Real spread : spreads) {
            mean += spread;
        }
        mean /= static_cast<Real>(spreads.size());
        Real variance = 0.0;
        for (const Real spread : spreads) {
            variance += (spread - mean) * (spread - mean);
        }
        report.spread_variance_bohr_squared = variance / static_cast<Real>(spreads.size());

        const Real large = LARGE_SPREAD_ANGSTROM_SQUARED /
                           (constants::BOHR_RADIUS_ANGSTROM * constants::BOHR_RADIUS_ANGSTROM);
        for (std::size_t index = 0; index < spreads.size(); index++) {
            if (report.spread_median_bohr_squared > 0.0 &&
                spreads[index] > OUTLIER_RATIO * report.spread_median_bohr_squared) {
                report.spread_outliers.push_back(static_cast<int>(index));
            }
            if (spreads[index] > large) {
                report.spreads_above_ten_angstrom_squared.push_back(static_cast<int>(index));
            }
        }
    }

    void locate_centers(WannierQualityReport& report,
                        const std::vector<Vector3D>& centers_bohr) const {
        report.shortest_bond_bohr = shortest_bond_bohr_;
        for (const Vector3D& center : centers_bohr) {
            const Vector3D center_crystal = crystal_from_cartesian_ * center;
            report.nearest_atom_distance_bohr.push_back(
                nearest_distance(center_crystal, atoms_crystal_));
            report.nearest_bond_midpoint_distance_bohr.push_back(
                bond_midpoints_crystal_.empty()
                    ? 0.0
                    : nearest_distance(center_crystal, bond_midpoints_crystal_));
        }
    }

    /// The shortest minimum-image distance from `point_crystal` to any of `sites_crystal`.
    [[nodiscard]] Real nearest_distance(const Vector3D& point_crystal,
                                        const std::vector<Vector3D>& sites_crystal) const {
        Real nearest = -1.0;
        for (const Vector3D& site : sites_crystal) {
            const Real distance = minimum_image_distance(point_crystal - site);
            if (nearest < 0.0 || distance < nearest) {
                nearest = distance;
            }
        }
        return nearest;
    }

    /// The shortest |f . A| over the periodic images of a fractional offset.
    [[nodiscard]] Real minimum_image_distance(Vector3D offset_crystal) const {
        for (int axis = 0; axis < 3; axis++) {
            offset_crystal[axis] -= std::round(offset_crystal[axis]);
        }
        Real shortest = -1.0;
        for_each_image(1, [&](const IntegerVector3D& image) {
            const Real distance =
                (cartesian_from_crystal_ * (offset_crystal + translation(image))).norm();
            if (shortest < 0.0 || distance < shortest) {
                shortest = distance;
            }
        });
        return shortest;
    }

    /**
     * Every pair of atoms over the periodic images, the shortest of them,
     * and the midpoints of the pairs within BOND_TOLERANCE of it. An atom
     * paired with its own zero image is the atom, not a bond.
     */
    void find_bonds() {
        struct Pair {
            Vector3D midpoint_crystal;
            Real length;
        };
        std::vector<Pair> pairs;
        Real shortest = -1.0;
        for (std::size_t first = 0; first < atoms_crystal_.size(); first++) {
            for (std::size_t second = first; second < atoms_crystal_.size(); second++) {
                for_each_image(BOND_IMAGE_RANGE, [&](const IntegerVector3D& image) {
                    if (first == second && image == IntegerVector3D{0, 0, 0}) {
                        return;
                    }
                    const Vector3D separation =
                        atoms_crystal_[second] + translation(image) - atoms_crystal_[first];
                    const Real length = (cartesian_from_crystal_ * separation).norm();
                    pairs.push_back({atoms_crystal_[first] + separation * 0.5, length});
                    if (shortest < 0.0 || length < shortest) {
                        shortest = length;
                    }
                });
            }
        }
        shortest_bond_bohr_ = shortest > 0.0 ? shortest : 0.0;
        for (const Pair& pair : pairs) {
            if (pair.length <= BOND_TOLERANCE * shortest) {
                bond_midpoints_crystal_.push_back(pair.midpoint_crystal);
            }
        }
    }

    /// Calls `visit` with every lattice translation whose components lie within +-`range` cells.
    template <typename Visitor> static void for_each_image(int range, Visitor visit) {
        for (int first = -range; first <= range; first++) {
            for (int second = -range; second <= range; second++) {
                for (int third = -range; third <= range; third++) {
                    visit(IntegerVector3D{first, second, third});
                }
            }
        }
    }

    [[nodiscard]] static Vector3D translation(const IntegerVector3D& image) {
        return Vector3D{
            static_cast<Real>(image.x), static_cast<Real>(image.y), static_cast<Real>(image.z)};
    }

    static constexpr int BOND_IMAGE_RANGE = 2;

    Matrix3x3 cartesian_from_crystal_;
    Matrix3x3 crystal_from_cartesian_;
    std::vector<Vector3D> atoms_crystal_;
    std::vector<Vector3D> bond_midpoints_crystal_;
    Real shortest_bond_bohr_ = 0.0;
};

} // namespace q3
