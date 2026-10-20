#pragma once

/**
 * FFTransformable — base class for planewave fields with dual real/reciprocal space storage.
 *
 * Provides typed storage for both the real-space and reciprocal-space
 * representations of a Fourier-transformable field, plus pure virtual
 * transform methods that subclasses implement using their specific FFT basis.
 *
 * Template parameters:
 *   RealRepresentationType        — e.g. std::vector<std::vector<Real>> for spin-resolved grids
 *   ReciprocalRepresentationType  — e.g. std::vector<Complex> for G-space coefficients
 */

namespace q3 {

template <typename RealRepresentationType, typename ReciprocalRepresentationType>
class FFTransformable {
  public:
    virtual ~FFTransformable() = default;

    // ─── Representation access ───────────────────────────────────────────────
    [[nodiscard]] const RealRepresentationType& real_space_representation() const {
        return real_space_data_;
    }
    RealRepresentationType& real_space_representation() {
        return real_space_data_;
    }
    [[nodiscard]] const ReciprocalRepresentationType& reciprocal_space_representation() const {
        return reciprocal_space_data_;
    }
    ReciprocalRepresentationType& reciprocal_space_representation() {
        return reciprocal_space_data_;
    }

    // ─── Transforms ─────────────────────────────────────────────────────────
    virtual void to_reciprocal_space() = 0;
    virtual void to_real_space() = 0;

  protected:
    RealRepresentationType real_space_data_;
    ReciprocalRepresentationType reciprocal_space_data_;
};

} // namespace q3
