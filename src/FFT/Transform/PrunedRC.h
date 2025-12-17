#ifndef IPPL_FFT_TRANSFORM_PRUNEDRC_HPP
#define IPPL_FFT_TRANSFORM_PRUNEDRC_HPP

#include <array>
#include <mpi.h>

#include "Utility/IpplTimings.h"
#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/Backend/Backend.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"

namespace ippl {
    //=========================================================================
    // Pruned Real-to-Complex Transform
    //=========================================================================

    template <typename RealField>
    class FFT<PrunedRCTransform, RealField> {
    public:
        static constexpr unsigned Dim = RealField::dim;

        using T         = typename RealField::value_type;
        using Complex_t = Kokkos::complex<T>;
        using MemSpace  = typename RealField::memory_space;
        using ExecSpace = typename RealField::execution_space;
        using Layout_t  = FieldLayout<Dim>;

        using ComplexField = Field<Complex_t, Dim, typename RealField::Mesh_t,
                                   typename RealField::Centering_t, ExecSpace>::uniform_type;

        using Backend_t     = fft::HeffteR2C<T, Dim, MemSpace>;
        using TempReal_t    = Kokkos::View<T***, Kokkos::LayoutLeft, MemSpace>;
        using TempComplex_t = Kokkos::View<Complex_t***, Kokkos::LayoutLeft, MemSpace>;

        FFT(const Layout_t& layoutReal, const Layout_t& layoutComplexFull,
            const Layout_t& layoutComplexPruned, const PruningParams<Dim>& pruning,
            const ParameterList& params)
            : pruning_(pruning) {
            static_assert(Dim == 2 || Dim == 3, "heFFTe only supports 2D and 3D");

            std::array<long long, 3> lowReal, highReal, lowComplexFull, highComplexFull;
            fft::domainToBounds<Dim>(layoutReal.getLocalNDIndex(), lowReal, highReal);
            fft::domainToBounds<Dim>(layoutComplexFull.getLocalNDIndex(), lowComplexFull,
                                     highComplexFull);

            heffte::box3d<long long> inbox{lowReal, highReal};
            heffte::box3d<long long> outbox{lowComplexFull, highComplexFull};

            int r2c_dir = params.get<int>("r2c_direction", 0);
            backend_ = std::make_unique<Backend_t>(inbox, outbox, r2c_dir, Comm->getCommunicator(),
                                                   params);

            // Store full complex dimensions for zero-padding
            const auto& lDomReal = layoutReal.getLocalNDIndex();
            fullComplexDims_[0]  = lDomReal[0].length() / 2 + 1;
            fullComplexDims_[1]  = lDomReal[1].length();
            fullComplexDims_[2]  = lDomReal[2].length();
        }

        void transform(TransformDirection direction, RealField& f, ComplexField& g);

    private:
        PruningParams<Dim> pruning_;
        std::unique_ptr<Backend_t> backend_;
        std::array<std::size_t, Dim> fullComplexDims_;

        TempReal_t tempReal_;
        TempComplex_t tempComplexFull_;
    };

    //-------------------------------------------------------------------------
    // Pruned R2C Implementation
    //-------------------------------------------------------------------------

    template <typename RealField>
    void FFT<PrunedRCTransform, RealField>::transform(TransformDirection direction, RealField& f,
                                                      ComplexField& g) {
        auto fview    = f.getView();
        auto gview    = g.getView();
        const int ngf = f.getNghost();
        const int ngg = g.getNghost();

        const auto& lDomReal   = f.getLayout().getLocalNDIndex();
        const auto& lDomPruned = g.getLayout().getLocalNDIndex();

        const std::size_t N0 = lDomReal[0].length();
        const std::size_t N1 = lDomReal[1].length();
        const std::size_t N2 = lDomReal[2].length();

        const std::size_t K0 = pruning_.n_modes[0];
        const std::size_t K1 = pruning_.n_modes[1];
        const std::size_t K2 = pruning_.n_modes[2];

        // Ensure temp buffers
        if (tempReal_.size() != f.getOwned().size()) {
            tempReal_ = detail::shrinkView("pruned_r2c_real", fview, ngf);
        }

        std::size_t fullComplexSize =
            fullComplexDims_[0] * fullComplexDims_[1] * fullComplexDims_[2];
        if (tempComplexFull_.size() != fullComplexSize) {
            tempComplexFull_ = TempComplex_t("pruned_r2c_complex_full", fullComplexDims_[0],
                                             fullComplexDims_[1], fullComplexDims_[2]);
        }

        if (direction == FORWARD) {
            // Copy real field to temp (no ghosts)
            auto& real_out = tempReal_;
            Kokkos::parallel_for(
                "copy_real_pruned_r2c",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {ngf, ngf, ngf}, {int(fview.extent(0)) - ngf, int(fview.extent(1)) - ngf,
                                      int(fview.extent(2)) - ngf}),
                KOKKOS_LAMBDA(int i, int j, int k) {
                    real_out(i - ngf, j - ngf, k - ngf) = fview(i, j, k);
                });

            // R2C FFT: real -> full complex
            backend_->forward(tempReal_.data(), tempComplexFull_.data());

            // Extract pruned modes from full complex to output
            // R2C layout: dimension 0 has [0, N/2] (only positive freqs)
            // dimensions 1,2 have [0, ..., N/2-1, -N/2, ..., -1]
            auto& fullcomplex_temp = tempComplexFull_;
            Kokkos::parallel_for(
                "extract_pruned_modes_r2c",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {lDomPruned[0].first(), lDomPruned[1].first(), lDomPruned[2].first()},
                    {lDomPruned[0].last() + 1, lDomPruned[1].last() + 1, lDomPruned[2].last() + 1}),
                KOKKOS_LAMBDA(int64_t gi, int64_t gj, int64_t gk) {
                    // Map pruned index to full grid index
                    // Dimension 0 (R2C): only positive freqs, direct mapping
                    // Dimensions 1,2: positive [0, K/2) -> [0, K/2), negative [K/2, K) -> [N-K/2,
                    // N)
                    int64_t fi = gi;  // R2C dimension - direct mapping
                    int64_t fj = (gj < int64_t(K1 / 2)) ? gj : (N1 - K1 + gj);
                    int64_t fk = (gk < int64_t(K2 / 2)) ? gk : (N2 - K2 + gk);

                    int li_full = fi;
                    int lj_full = fj;
                    int lk_full = fk;

                    int li_pruned = gi - lDomPruned[0].first() + ngg;
                    int lj_pruned = gj - lDomPruned[1].first() + ngg;
                    int lk_pruned = gk - lDomPruned[2].first() + ngg;

                    gview(li_pruned, lj_pruned, lk_pruned) =
                        fullcomplex_temp(li_full, lj_full, lk_full);
                });

        } else {  // BACKWARD
            // Zero-initialize full complex temp
            Kokkos::deep_copy(tempComplexFull_, Complex_t(0, 0));
            auto& fullcomplex_temp = tempComplexFull_;

            // Copy pruned modes to correct positions in full complex
            Kokkos::parallel_for(
                "zeropad_pruned_modes_r2c",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {lDomPruned[0].first(), lDomPruned[1].first(), lDomPruned[2].first()},
                    {lDomPruned[0].last() + 1, lDomPruned[1].last() + 1, lDomPruned[2].last() + 1}),
                KOKKOS_LAMBDA(int64_t gi, int64_t gj, int64_t gk) {
                    // Map pruned index to full grid index
                    int64_t fi = gi;  // R2C dimension - direct mapping
                    int64_t fj = (gj < int64_t(K1 / 2)) ? gj : (N1 - K1 + gj);
                    int64_t fk = (gk < int64_t(K2 / 2)) ? gk : (N2 - K2 + gk);

                    int li_full = fi;
                    int lj_full = fj;
                    int lk_full = fk;

                    int li_pruned = gi - lDomPruned[0].first() + ngg;
                    int lj_pruned = gj - lDomPruned[1].first() + ngg;
                    int lk_pruned = gk - lDomPruned[2].first() + ngg;

                    fullcomplex_temp(li_full, lj_full, lk_full) =
                        gview(li_pruned, lj_pruned, lk_pruned);
                });

            // C2R IFFT: full complex -> real
            backend_->backward(tempComplexFull_.data(), tempReal_.data());

            auto& realIn = tempReal_;
            // Copy real result to output (with ghosts)
            Kokkos::parallel_for(
                "copy_real_from_temp_pruned",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {ngf, ngf, ngf}, {int(fview.extent(0)) - ngf, int(fview.extent(1)) - ngf,
                                      int(fview.extent(2)) - ngf}),
                KOKKOS_LAMBDA(int i, int j, int k) {
                    fview(i, j, k) = realIn(i - ngf, j - ngf, k - ngf);
                });
        }
    }

}  // namespace ippl

#endif