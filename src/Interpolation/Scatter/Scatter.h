#ifndef IPPL_SCATTER_H
#define IPPL_SCATTER_H

#include <unordered_set>

#include "Utility/Tuning.h"

#include "Interpolation/Binning.h"
#include "Interpolation/Scatter/AtomicScatter.h"
#include "Interpolation/Scatter/GridParallelScatter.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"
#include "Interpolation/Scatter/ScatterConfig.h"
#include "Interpolation/Scatter/TileSizeCache.h"
#include "Interpolation/Scatter/TiledScatter.h"
#include "Interpolation/WidthDispatcher.h"
#include "Particle/ParticleAttrib.h"

namespace ippl {

    namespace Interpolation::detail {

        struct UnsortedPolicy {
            static constexpr bool use_sorting      = false;
            static constexpr bool requires_binning = false;
        };

        struct SortedPolicy {
            static constexpr bool use_sorting      = true;
            static constexpr bool requires_binning = true;
        };

        template <typename Kernel, typename FieldType, typename PositionsType, typename ValuesType>
        struct DeduceScatterTypes {
            using FieldTr = ippl::detail::FieldTraits<std::decay_t<FieldType>>;
            using PosTr   = ippl::detail::AttribTraits<std::decay_t<PositionsType>>;
            using ValTr   = ippl::detail::AttribTraits<std::decay_t<ValuesType>>;
            using VecTr   = ippl::detail::VectorTraits<typename PosTr::value_type>;

            using type = ScatterTypes<FieldTr::dim, typename VecTr::real_type, std::decay_t<Kernel>,
                                      typename FieldTr::view_type, typename PosTr::view_type,
                                      typename ValTr::view_type>;
        };

        template <typename Kernel, typename FieldType, typename PositionsType, typename ValuesType>
        using DeducedScatterTypes =
            typename DeduceScatterTypes<Kernel, FieldType, PositionsType, ValuesType>::type;

        // ─────────────────────────────────────────────────────────────────────────
        // Tuner registry - one tuner per (method, Dim, RealType, is_complex) combo
        // ─────────────────────────────────────────────────────────────────────────
        template <template <int, class, class> class Impl, unsigned Dim, typename RealType,
                  bool IsComplex>
        TileSizeTuner<Dim, Vector<int, Dim>>& get_scatter_tuner() {
            static TileSizeTuner<Dim, Vector<int, Dim>> tuner;
            return tuner;
        }

    }  // namespace Interpolation::detail

    // ─────────────────────────────────────────────────────────────────────────────
    // Helper trait: detect Kokkos::complex<T> at compile time.
    // Used to derive is_complex from the field's value_type in operator().
    // ─────────────────────────────────────────────────────────────────────────────
    namespace detail {
        template <typename T>
        struct is_kokkos_complex : std::false_type {};
        template <typename T>
        struct is_kokkos_complex<Kokkos::complex<T>> : std::true_type {};
    }  // namespace detail

    template <typename Kernel, unsigned Dim>
    class Scatter {
    public:
        Scatter(const Kernel& kernel, const Interpolation::ScatterConfig<Dim>& config = {})
            : kernel_m(kernel)
            , config_m(config) {}

        template <typename ValueT, typename FieldT, class Mesh, class Centering, class... ViewArgs,
                  typename ParticleT, class... PosProps, class... ValProps>
        void operator()(Field<FieldT, Dim, Mesh, Centering, ViewArgs...>& field,
                        const ParticleAttrib<Vector<ParticleT, Dim>, PosProps...>& positions,
                        const ParticleAttrib<ValueT, ValProps...>& values) {
            using Types =
                Interpolation::detail::DeducedScatterTypes<Kernel, decltype(field),
                                                           decltype(positions), decltype(values)>;

            // ── Determine field value type complexity ──────────────────────────
            // FieldT is the grid value type.  Particle values may differ (e.g. real
            // particles scattering to a complex grid), so we base is_complex on FieldT.
            constexpr bool is_complex_field =
                ippl::detail::is_kokkos_complex<FieldT>::value;

            // ── Auto-select best method from benchmark cache ───────────────────
            // If a tile-size cache is loaded (from BenchmarkTileSweep CSV output),
            // pick the method with the highest recorded throughput for this kernel
            // width and value type.  Users who need a specific method should either
            // set enable_tuning=true or use an empty/absent cache file.
            //
            // The cache-resolved method is stored in config_m for this call so that
            // resolve_config() (called inside dispatch) uses the same method key for
            // its own cache tile/team lookup.  A log message is printed once per
            // (method, width, is_complex) triple.
            {
                auto& cache = Interpolation::TileSizeCache::instance();
                if (!config_m.enable_tuning && cache.loaded()) {
                    if (auto best = cache.get_best(kernel_m.width(), is_complex_field)) {
                        // Print info once per (original_method, best_method, width, is_complex)
                        static std::unordered_set<std::size_t> reported;
                        const std::size_t key =
                            (static_cast<std::size_t>(config_m.method) * 3 +
                             static_cast<std::size_t>(best->method)) * 10000 +
                            static_cast<std::size_t>(kernel_m.width()) * 2 +
                            static_cast<std::size_t>(is_complex_field);
                        if (reported.find(key) == reported.end()) {
                            reported.insert(key);
                            if (best->method != config_m.method) {
                                std::cout << "[Scatter] Auto-selecting method "
                                          << method_name(best->method)
                                          << " (cached throughput=" << std::fixed
                                          << std::setprecision(1)
                                          << best->entry.throughput_Mpts_s
                                          << " Mpts/s) for w=" << kernel_m.width()
                                          << " is_complex=" << is_complex_field
                                          << " (overrides "
                                          << method_name(config_m.method) << ")\n";
                            }
                        }
                        // Override the method.  resolve_config() will subsequently
                        // look up tile/team_size/oversubscription for this method.
                        config_m.method = best->method;
                    }
                }
            }

            const auto method = config_m.method;

            const bool run_atomic =
                (method == Interpolation::ScatterMethod::Atomic)
                || (method == Interpolation::ScatterMethod::OutputFocusedZBatch);

            if (run_atomic) {
                if (config_m.sort) {
                    dispatch<Interpolation::detail::AtomicScatter, Types,
                             Interpolation::detail::SortedPolicy>(field, positions, values);
                } else {
                    dispatch<Interpolation::detail::AtomicScatter, Types,
                             Interpolation::detail::UnsortedPolicy>(field, positions, values);
                }
                return;
            }

            if (method == Interpolation::ScatterMethod::Tiled) {
                dispatch<Interpolation::detail::TiledScatter, Types,
                         Interpolation::detail::SortedPolicy>(field, positions, values);
                return;
            }

            if (method == Interpolation::ScatterMethod::OutputFocused) {
                dispatch<Interpolation::detail::GridParallelScatter, Types,
                         Interpolation::detail::SortedPolicy>(field, positions, values);
            }
        }

    private:
        // ------------------------------------------------------------------
        // Human-readable method name for log messages
        // ------------------------------------------------------------------
        static const char* method_name(Interpolation::ScatterMethod m) {
            switch (m) {
                case Interpolation::ScatterMethod::Tiled:         return "Tiled";
                case Interpolation::ScatterMethod::OutputFocused: return "OutputFocused";
                case Interpolation::ScatterMethod::Atomic:        return "Atomic";
                default:                                          return "Unknown";
            }
        }

        // ------------------------------------------------------------------
        // get_device_shmem
        // ------------------------------------------------------------------
        static size_t get_device_shmem() {
            static size_t cached = 0;
            if (cached > 0)
                return cached;
#if defined(KOKKOS_ENABLE_CUDA)
            {
                int dev = 0, b = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&b, cudaDevAttrMaxSharedMemoryPerBlock, dev);
                cached = static_cast<size_t>(std::max(b, 0));
            }
#elif defined(KOKKOS_ENABLE_HIP)
            {
                int dev = 0;
                hipGetDevice(&dev);
                hipDeviceProp_t prop;
                hipGetDeviceProperties(&prop, dev);
                cached = prop.sharedMemPerBlock;
            }
#else
            cached = static_cast<size_t>(1) << 30;
#endif
            return cached;
        }

        // ------------------------------------------------------------------
        // resolve_config
        //
        // Returns a copy of config_m with all benchmarked-optimal parameters
        // applied.  Priority order (highest to lowest):
        //
        //   1. ScatterConfig::enable_tuning == true  → unchanged (tuner handles it).
        //
        //   2. TileSizeCache has an entry for (config_m.method, width, is_complex)
        //      → apply tile sizes, team_size, oversubscription_factor, z_batches.
        //        A value of -1 in the cache entry means "not recorded".
        //
        //   3. Fallback: config_m unchanged (ScatterConfig defaults).
        //
        // Note: config_m.method has already been overridden by get_best() in
        // operator() before dispatch() is called, so the cache lookup here uses
        // the already-resolved effective method.
        // ------------------------------------------------------------------
        template <template <int, class, class> class Impl, int W, class Types, class Policy,
                  bool IsComplex>
        Interpolation::ScatterConfig<Dim> resolve_config() const {
            if (config_m.enable_tuning)
                return config_m;

            auto& cache  = Interpolation::TileSizeCache::instance();
            auto  cached = cache.get(config_m.method, W, IsComplex);

            if (cached.has_value()) {
                const auto& e = cached.value();

                // One-time info message per (method, width, is_complex) triple.
                if (cache.loaded()) {
                    static std::unordered_set<std::size_t> reported;
                    std::size_t key =
                        (static_cast<std::size_t>(config_m.method) * 100 + W) * 2
                        + static_cast<std::size_t>(IsComplex);
                    if (reported.find(key) == reported.end()) {
                        reported.insert(key);
                        std::cout << "[Scatter] Applying cached config from " << cache.source()
                                  << ": method=" << method_name(config_m.method)
                                  << " width=" << W
                                  << " is_complex=" << IsComplex
                                  << " tile=(" << e.tile[0];
                        for (unsigned d = 1; d < Dim; ++d)
                            std::cout << "," << e.tile[d < 3 ? d : 2];
                        std::cout << ")";
                        if (e.team_size > 0)
                            std::cout << " team_size=" << e.team_size;
                        if (e.oversubscription_factor > 0)
                            std::cout << " oversubscription=" << e.oversubscription_factor;
                        if (e.z_batches > 0)
                            std::cout << " z_batches=" << e.z_batches;
                        std::cout << " throughput=" << std::fixed << std::setprecision(1)
                                  << e.throughput_Mpts_s << " Mpts/s\n";
                    }
                }

                auto resolved = config_m;

                Vector<int, Dim> tile;
                for (unsigned d = 0; d < Dim; ++d)
                    tile[d] = e.tile[d < 3 ? d : 2];
                resolved.set_tile_size(tile);

                if (e.team_size > 0)
                    resolved.team_size = e.team_size;
                if (e.oversubscription_factor > 0)
                    resolved.oversubscription_factor = e.oversubscription_factor;
                if (e.z_batches > 0)
                    resolved.z_batches = e.z_batches;

                return resolved;
            }

            return config_m;
        }

        // ------------------------------------------------------------------
        // clamp_tile_to_shmem
        // ------------------------------------------------------------------
        template <template <int, class, class> class Impl, int W, class Types, class Policy,
                  bool IsComplex>
        static void clamp_tile_to_shmem(Interpolation::ScatterConfig<Dim>& cfg) {
            if constexpr (!Impl<W, Types, Policy>::requires_binning)
                return;

            using execution_space = typename Types::execution_space;
            using team_policy     = Kokkos::TeamPolicy<execution_space>;

            const size_t avail = team_policy(1, cfg.team_size).scratch_size_max(0);

            Vector<int, Dim> tile = cfg.get_tile_size();

            for (int itr = 0; itr < 64; ++itr) {
                const size_t req =
                    Impl<W, Types, Policy>::template compute_scratch_size<IsComplex>(
                        tile, cfg.team_size, cfg.z_batches);
                if (req <= avail)
                    break;

                unsigned maxd = 0;
                for (unsigned d = 1; d < Dim; ++d)
                    if (tile[d] > tile[maxd])
                        maxd = d;

                if (tile[maxd] <= 1)
                    break;

                --tile[maxd];
            }

            cfg.set_tile_size(tile);
        }

        template <template <int, class, class> class Impl, class Types, class Policy, class Field,
                  class Positions, class Values>
        void dispatch(Field& field, const Positions& positions, const Values& values) {
            using memory_space = typename Types::memory_space;
            using RealType     = typename Types::RealType;
            using grid_value_t = typename Field::value_type;

            constexpr bool is_complex = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;

            const int width          = kernel_m.width();
            const size_t n_particles = positions.getParticleCount();

            Interpolation::WidthDispatcher<1, 14>::dispatch(width, [&]<int W>() {
                // ── Step 1: Resolve full config from cache ─────────────────────
                auto tuned_config = resolve_config<Impl, W, Types, Policy, is_complex>();

                if constexpr (Impl<W, Types, Policy>::requires_binning) {
                    if (config_m.enable_tuning) {
                        Vector<int, Dim> tuned_tile =
                            get_tuned_tile_size<Impl, W, Types, Policy, is_complex>(
                                field, tuned_config.get_tile_size());
                        tuned_config.set_tile_size(tuned_tile);
                    }
                }

                // ── Step 1b: Safety – clamp tile to shared-memory budget ───────
                clamp_tile_to_shmem<Impl, W, Types, Policy, is_complex>(tuned_config);

                const Vector<int, Dim> tile_size = tuned_config.get_tile_size();

                // ── Step 2: Binning ────────────────────────────────────────────
                Interpolation::detail::BinningResult<Dim, memory_space> binning;
                if constexpr (Impl<W, Types, Policy>::requires_binning) {
                    binning = performBinning<Types>(positions, field, tile_size);
                } else if (config_m.do_binning()) {
                    binning = performBinning<Types>(positions, field, tile_size);
                }

                // ── Step 3: Run functor ────────────────────────────────────────
                auto args = Impl<W, Types, Policy>::Arguments::create(
                    field, positions, values, kernel_m, tuned_config, binning);

                Impl<W, Types, Policy> functor{std::move(args)};

                field = 0.0;
                functor.run(n_particles);
                Kokkos::fence();

                // ── Step 4: End tuning context ─────────────────────────────────
                if constexpr (Impl<W, Types, Policy>::requires_binning) {
                    if (config_m.enable_tuning) {
                        auto& tuner = Interpolation::detail::get_scatter_tuner<Impl, Dim, RealType,
                                                                               is_complex>();
                        tuner.end();
                    }
                }

                field.accumulateHalo();
            });
        }

        template <template <int, class, class> class Impl, int W, class Types, class Policy,
                  bool IsComplex, class Field>
        Vector<int, Dim> get_tuned_tile_size(const Field& /*field*/,
                                             const Vector<int, Dim>& default_tile) {
            using RealType        = typename Types::RealType;
            using execution_space = typename Types::execution_space;
            using team_policy     = Kokkos::TeamPolicy<execution_space>;

            auto& tuner =
                Interpolation::detail::get_scatter_tuner<Impl, Dim, RealType, IsComplex>();

            if (!tuner.is_initialized()) {
                const size_t max_scratch = team_policy(1, config_m.team_size).scratch_size_max(0);

                std::vector<int> candidates = {1, 2, 3, 4, 8, 16, 32};

                auto scratch_calc = [&](const Vector<int, Dim>& tile) {
                    return Impl<W, Types, Policy>::template compute_scratch_size<IsComplex>(
                        tile, config_m.team_size, config_m.z_batches);
                };

                tuner.initialize("Scatter_" + std::string(typeid(Impl<W, Types, Policy>).name()),
                                 candidates, max_scratch, scratch_calc, default_tile);
            }

            return tuner.begin();
        }

        template <typename Types, typename Positions, typename Field>
        auto performBinning(const Positions& positions, const Field& field,
                            const Vector<int, Dim>& tile_size) {
            using memory_space = typename Types::memory_space;

            auto [permute, bin_offsets, num_tiles] = Interpolation::detail::bin_particles(
                positions, field.getLayout(), field.get_mesh(), tile_size, kernel_m.width());

            return Interpolation::detail::BinningResult<Dim, memory_space>{permute, bin_offsets,
                                                                           num_tiles};
        }

        Kernel kernel_m;
        Interpolation::ScatterConfig<Dim> config_m;
    };

}  // namespace ippl

#endif  // IPPL_SCATTER_H