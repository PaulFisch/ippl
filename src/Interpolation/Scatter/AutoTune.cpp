// ============================================================================
// Width-2 scatter auto-tuner — see AutoTune.h for the user-visible contract.
// ============================================================================

#include "Ippl.h"

#include "Interpolation/Scatter/AutoTune.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Field/Field.h"
#include "FieldLayout/FieldLayout.h"
#include "Interpolation/Gather/GatherConfig.h"
#include "Interpolation/Kernels.h"
#include "Interpolation/Scatter/ScatterConfig.h"
#include "Meshes/UniformCartesian.h"
#include "Particle/ParticleAttrib.h"
#include "Particle/ParticleBase.h"
#include "Particle/ParticleSpatialLayout.h"

#include "Interpolation/Gather/Gather.h"
#include "Interpolation/Scatter/Scatter.h"

namespace ippl::Interpolation::AutoTune {

    namespace {

        struct Sample {
            ScatterMethod method;
            std::string   value_type;  // "real" | "complex"
            int           kernel_width;
            int           tile_x, tile_y, tile_z;
            int           team_size;
            int           osub;
            int           z_batches;
            double        throughput_Mpts_s;
            double        time_ms;
        };

        struct GatherSample {
            GatherMethod method;
            int          tile_x, tile_y, tile_z;
            double       throughput_Mpts_s;
        };

        const char* gather_method_name(GatherMethod m) {
            switch (m) {
                case GatherMethod::Atomic:     return "Atomic";
                case GatherMethod::AtomicSort: return "AtomicSort";
            }
            return "Atomic";
        }

        void write_gather_csv(const std::string& path, const GatherSample& s) {
            std::ofstream out(path);
            if (!out.is_open()) return;
            out << "method,kernel_width,tile_x,tile_y,tile_z,throughput_Mpts_s\n"
                << std::fixed << std::setprecision(2)
                << gather_method_name(s.method) << ",2,"
                << s.tile_x << "," << s.tile_y << "," << s.tile_z << ","
                << s.throughput_Mpts_s << "\n";
        }

        const char* method_name(ScatterMethod m) {
            switch (m) {
                case ScatterMethod::Atomic:              return "Atomic";
                case ScatterMethod::Tiled:               return "Tiled";
                case ScatterMethod::OutputFocused:       return "OutputFocused";
                case ScatterMethod::OutputFocusedZBatch: return "OutputFocusedZBatch";
            }
            return "Atomic";
        }

        void write_csv(const std::string& path, const std::vector<Sample>& samples) {
            const std::string header =
                "method,value_type,kernel_width,rho,best_tile_x,best_tile_y,best_tile_z,"
                "best_team_size,best_oversubscription_factor,best_z_batches,"
                "throughput_Mpts_s,time_ms,kernel_evaluations,preflight_rejections";
            std::ofstream out(path);
            if (!out.is_open()) return;
            out << header << "\n" << std::fixed;
            for (const auto& s : samples) {
                out << method_name(s.method) << "," << s.value_type << "," << s.kernel_width
                    << ",0.0," << s.tile_x << "," << s.tile_y << "," << s.tile_z << ","
                    << s.team_size << "," << s.osub << "," << s.z_batches << ","
                    << std::setprecision(2) << s.throughput_Mpts_s << ","
                    << std::setprecision(4) << s.time_ms << ",0,0\n";
            }
        }

        // Trivial CSV for backends with nothing to tune: Atomic only, team=1.
        void write_trivial(const std::string& path) {
            std::vector<Sample> samples;
            for (int w : {1, 2}) {
                for (const char* vt : {"real", "complex"}) {
                    samples.push_back({ScatterMethod::Atomic, vt, w, 1, 1, 1, 1, 1, 1, 0.0, 0.0});
                }
            }
            write_csv(path, samples);
        }

        void write_trivial_gather(const std::string& path) {
            write_gather_csv(path, GatherSample{GatherMethod::Atomic, 1, 1, 1, 0.0});
        }

        template <typename ExecSpace>
        double time_config(ScatterMethod method, const Vector<int, 3>& tile_size,
                           int team_size, int osub, int z_batches) {
            using value_t   = double;
            using mesh_t    = ippl::UniformCartesian<value_t, 3>;
            using center_t  = typename mesh_t::DefaultCentering;
            using field_t   = ippl::Field<value_t, 3, mesh_t, center_t, ExecSpace>;
            using flayout_t = ippl::FieldLayout<3>;
            using playout_t = ippl::ParticleSpatialLayout<value_t, 3, mesh_t, ExecSpace>;

            constexpr unsigned N         = 32;
            constexpr size_t   nParticle = 100000;

            ippl::Index Ix(N), Iy(N), Iz(N);
            ippl::NDIndex<3> dom(Ix, Iy, Iz);
            std::array<bool, 3> isParallel = {true, true, true};
            ippl::Vector<value_t, 3> hx{1.0 / N, 1.0 / N, 1.0 / N};
            ippl::Vector<value_t, 3> origin{0, 0, 0};

            flayout_t layout(MPI_COMM_WORLD, dom, isParallel);
            mesh_t    mesh(dom, hx, origin);
            field_t   field(mesh, layout);
            field = 0.0;

            playout_t playout(layout, mesh);

            struct Bunch : ippl::ParticleBase<playout_t> {
                explicit Bunch(playout_t& pl) : ippl::ParticleBase<playout_t>(pl) {
                    this->addAttribute(Q);
                }
                ippl::ParticleAttrib<value_t, ExecSpace> Q;
            };
            Bunch bunch(playout);
            const size_t nLoc = nParticle / std::max(1, ippl::Comm->size());
            bunch.create(nLoc);

            auto R_host = bunch.R.getHostMirror();
            std::mt19937_64 eng(42 + ippl::Comm->rank());
            std::uniform_real_distribution<value_t> u(0.01, 0.99);
            for (size_t i = 0; i < nLoc; ++i) {
                R_host(i)[0] = u(eng);
                R_host(i)[1] = u(eng);
                R_host(i)[2] = u(eng);
            }
            Kokkos::deep_copy(bunch.R.getView(), R_host);
            bunch.Q = 1.0;
            bunch.update();

            ScatterConfig<3> cfg = ScatterConfig<3>::template get_default<ExecSpace>();
            cfg.method           = method;
            cfg.set_tile_size(tile_size);
            if (team_size > 0) cfg.team_size = team_size;
            if (osub > 0)      cfg.oversubscription_factor = osub;
            if (z_batches > 0) cfg.z_batches = z_batches;
            cfg.lock_method   = true;
            cfg.enable_tuning = false;

            ippl::Interpolation::LinearKernel<value_t> cic;

            field = 0.0;
            bunch.Q.scatter_kernel(field, bunch.R, cic, cfg);
            Kokkos::fence();

            constexpr int runs = 3;
            double best_ms     = 1e18;
            for (int r = 0; r < runs; ++r) {
                field = 0.0;
                Kokkos::fence();
                auto t0 = std::chrono::steady_clock::now();
                bunch.Q.scatter_kernel(field, bunch.R, cic, cfg);
                Kokkos::fence();
                auto t1 = std::chrono::steady_clock::now();
                const double ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best_ms) best_ms = ms;
            }
            return (best_ms > 0) ? double(nLoc) / 1e6 / (best_ms * 1e-3) : 0.0;
        }

        template <typename ExecSpace>
        double time_gather(GatherMethod method, const Vector<int, 3>& tile_size) {
            using value_t   = double;
            using mesh_t    = ippl::UniformCartesian<value_t, 3>;
            using center_t  = typename mesh_t::DefaultCentering;
            using field_t   = ippl::Field<value_t, 3, mesh_t, center_t, ExecSpace>;
            using flayout_t = ippl::FieldLayout<3>;
            using playout_t = ippl::ParticleSpatialLayout<value_t, 3, mesh_t, ExecSpace>;

            constexpr unsigned N         = 32;
            constexpr size_t   nParticle = 100000;

            ippl::Index Ix(N), Iy(N), Iz(N);
            ippl::NDIndex<3> dom(Ix, Iy, Iz);
            std::array<bool, 3> isParallel = {true, true, true};
            ippl::Vector<value_t, 3> hx{1.0 / N, 1.0 / N, 1.0 / N};
            ippl::Vector<value_t, 3> origin{0, 0, 0};

            flayout_t layout(MPI_COMM_WORLD, dom, isParallel);
            mesh_t    mesh(dom, hx, origin);
            field_t   field(mesh, layout);
            field = 1.0;

            playout_t playout(layout, mesh);
            struct Bunch : ippl::ParticleBase<playout_t> {
                explicit Bunch(playout_t& pl) : ippl::ParticleBase<playout_t>(pl) {
                    this->addAttribute(Q);
                }
                ippl::ParticleAttrib<value_t, ExecSpace> Q;
            };
            Bunch bunch(playout);
            const size_t nLoc = nParticle / std::max(1, ippl::Comm->size());
            bunch.create(nLoc);

            auto R_host = bunch.R.getHostMirror();
            std::mt19937_64 eng(42 + ippl::Comm->rank());
            std::uniform_real_distribution<value_t> u(0.01, 0.99);
            for (size_t i = 0; i < nLoc; ++i) {
                R_host(i)[0] = u(eng);
                R_host(i)[1] = u(eng);
                R_host(i)[2] = u(eng);
            }
            Kokkos::deep_copy(bunch.R.getView(), R_host);
            bunch.Q = 0.0;
            bunch.update();

            GatherConfig<3> cfg = GatherConfig<3>::template get_default<ExecSpace>();
            cfg.method = method;
            cfg.set_tile_size({tile_size[0], tile_size[1], tile_size[2]});

            ippl::Interpolation::LinearKernel<value_t> cic;
            ippl::Gather<decltype(cic), 3> gather_op(cic, cfg);

            gather_op(field, bunch.R, bunch.Q);
            Kokkos::fence();

            constexpr int runs = 3;
            double best_ms     = 1e18;
            for (int r = 0; r < runs; ++r) {
                Kokkos::fence();
                auto t0 = std::chrono::steady_clock::now();
                gather_op(field, bunch.R, bunch.Q);
                Kokkos::fence();
                auto t1 = std::chrono::steady_clock::now();
                const double ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best_ms) best_ms = ms;
            }
            return (best_ms > 0) ? double(nLoc) / 1e6 / (best_ms * 1e-3) : 0.0;
        }

        template <typename ExecSpace>
        GatherSample sweep_gather() {
            GatherSample best{GatherMethod::Atomic, 1, 1, 1, 0.0};

            const double atomic_tp = time_gather<ExecSpace>(GatherMethod::Atomic, {1, 1, 1});
            if (atomic_tp > best.throughput_Mpts_s) {
                best = GatherSample{GatherMethod::Atomic, 1, 1, 1, atomic_tp};
            }

            for (const auto& t : std::vector<Vector<int, 3>>{{4, 4, 4}, {8, 8, 8}}) {
                const double tp = time_gather<ExecSpace>(GatherMethod::AtomicSort, t);
                if (tp > best.throughput_Mpts_s) {
                    best = GatherSample{GatherMethod::AtomicSort, t[0], t[1], t[2], tp};
                }
            }
            return best;
        }

        template <typename ExecSpace>
        std::vector<Sample> sweep() {
            std::vector<Sample> out;

            const bool host_backend =
#ifdef KOKKOS_ENABLE_OPENMP
                std::is_same_v<ExecSpace, Kokkos::OpenMP>
#else
                false
#endif
                ;
            const int default_team = host_backend ? 1 : 32;

            // Atomic — tile is irrelevant to the dispatcher; fix to (1,1,1).
            {
                Vector<int, 3> tile{1, 1, 1};
                const double tp =
                    time_config<ExecSpace>(ScatterMethod::Atomic, tile, default_team, 1, 1);
                out.push_back(Sample{ScatterMethod::Atomic, "real", 2,
                                     1, 1, 1, default_team, 1, 1, tp, 0.0});
            }

            // Tiled — small candidate set over (tile, team, osub).
            {
                Sample best{ScatterMethod::Tiled, "real", 2, 4, 4, 4,
                            host_backend ? 1 : 64, 1, 1, 0.0, 0.0};
                for (const auto& t :
                     std::vector<Vector<int, 3>>{{2, 2, 2}, {4, 4, 4}, {8, 8, 8}}) {
                    const double tp = time_config<ExecSpace>(ScatterMethod::Tiled, t,
                                                             host_backend ? 1 : 64, 1, 1);
                    if (tp > best.throughput_Mpts_s) {
                        best = Sample{ScatterMethod::Tiled, "real", 2, t[0], t[1], t[2],
                                      host_backend ? 1 : 64, 1, 1, tp, 0.0};
                    }
                }
                out.push_back(best);
            }

            // OutputFocused — tile + z_batches.
            {
                Sample best{ScatterMethod::OutputFocused, "real", 2, 2, 2, 2,
                            host_backend ? 1 : 128, 1, 1, 0.0, 0.0};
                for (const auto& t : std::vector<Vector<int, 3>>{{2, 2, 2}, {4, 4, 4}}) {
                    for (int zb : {1, 4}) {
                        const double tp = time_config<ExecSpace>(
                            ScatterMethod::OutputFocused, t, host_backend ? 1 : 128, 1, zb);
                        if (tp > best.throughput_Mpts_s) {
                            best = Sample{ScatterMethod::OutputFocused, "real", 2,
                                          t[0], t[1], t[2],
                                          host_backend ? 1 : 128, 1, zb, tp, 0.0};
                        }
                    }
                }
                out.push_back(best);
            }

            // Mirror real → complex.
            const size_t base = out.size();
            for (size_t i = 0; i < base; ++i) {
                Sample s     = out[i];
                s.value_type = "complex";
                out.push_back(s);
            }
            return out;
        }

    }  // namespace

    bool runOnFirstUse(const std::string& output_path) {
        if (const char* dis = std::getenv("IPPL_AUTO_TUNE")) {
            if (std::string(dis) == "0") {
                return std::filesystem::exists(output_path);
            }
        }

        const std::string gather_path =
            std::filesystem::path(output_path).replace_filename("gather_sweep_optimal.csv").string();

        const bool scatter_done = std::filesystem::exists(output_path);
        const bool gather_done  = std::filesystem::exists(gather_path);
        if (scatter_done && gather_done) return true;

        const bool is_rank_zero = (ippl::Comm == nullptr) || (ippl::Comm->rank() == 0);

        if (is_rank_zero) {
#ifdef KOKKOS_ENABLE_CUDA
            if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>) {
                if (!scatter_done) write_csv(output_path, sweep<Kokkos::Cuda>());
                if (!gather_done)  write_gather_csv(gather_path, sweep_gather<Kokkos::Cuda>());
            } else
#endif
#ifdef KOKKOS_ENABLE_OPENMP
            if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>) {
                if (!scatter_done) write_csv(output_path, sweep<Kokkos::OpenMP>());
                if (!gather_done)  write_gather_csv(gather_path, sweep_gather<Kokkos::OpenMP>());
            } else
#endif
            {
                if (!scatter_done) write_trivial(output_path);
                if (!gather_done)  write_trivial_gather(gather_path);
            }
        }
        if (ippl::Comm != nullptr) {
            ippl::Comm->barrier();
        }
        return std::filesystem::exists(output_path);
    }

}  // namespace ippl::Interpolation::AutoTune
