// MWE v11: Two TeamThreadRange with DIFFERENT range sizes
// The scatter uses range (pstart, pend) which is offset per tile
// The flush uses range (0, hist_total)
//
// This tests if the issue is related to thread scheduling/assignment
// when doing multiple TeamThreadRange in the same team

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        using exec_space = Kokkos::DefaultExecutionSpace;
        using complex_t = Kokkos::complex<double>;
        using scratch_space = typename exec_space::scratch_memory_space;
        using scratch_view = Kokkos::View<double*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        std::cout << "Execution space: " << exec_space::name() << std::endl;
        std::cout << "v11: Two TeamThreadRange with different ranges" << std::endl;

        constexpr int num_teams = 125;
        constexpr int team_size = 16;

        constexpr int tile_x = 4, tile_y = 4, tile_z = 4;
        constexpr int W = 1;
        constexpr int hx = tile_x + W;
        constexpr int hy = tile_y + W;
        constexpr int hz = tile_z + W;
        constexpr int hist_total = hx * hy * hz;  // 125
        constexpr int half_left = (W - 1) / 2;

        constexpr int nghost = 1;
        constexpr int nx = 20, ny = 20, nz = 20;

        // Key difference: use OFFSET ranges like the real scatter does
        constexpr int items_per_tile = 50;  // Different from hist_total!

        Kokkos::View<int*, exec_space> offsets("offsets", num_teams + 1);
        auto h_offsets = Kokkos::create_mirror_view(offsets);
        for (int t = 0; t <= num_teams; ++t) {
            h_offsets(t) = t * items_per_tile;
        }
        Kokkos::deep_copy(offsets, h_offsets);

        Kokkos::View<complex_t***, Kokkos::LayoutLeft, exec_space> grid("grid",
            nx + 2*nghost, ny + 2*nghost, nz + 2*nghost);
        Kokkos::deep_copy(grid, complex_t(0.0, 0.0));

        const size_t scratch_size = 2 * hist_total * sizeof(double);

        std::cout << "items_per_tile=" << items_per_tile << ", hist_total=" << hist_total << std::endl;
        std::cout << "Launching..." << std::endl;
        std::cout << std::flush;

        auto start = std::chrono::high_resolution_clock::now();

        Kokkos::parallel_for(
            "v11_different_ranges",
            Kokkos::TeamPolicy<exec_space>(num_teams, team_size)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<exec_space>::member_type& team) {
                const int tile_id = team.league_rank();

                const int tile_ix = tile_id % 5;
                const int tile_iy = (tile_id / 5) % 5;
                const int tile_iz = tile_id / 25;

                const int tile_x0 = tile_ix * tile_x;
                const int tile_y0 = tile_iy * tile_y;
                const int tile_z0 = tile_iz * tile_z;

                scratch_view hist_r(team.team_scratch(0), hist_total);
                scratch_view hist_i(team.team_scratch(0), hist_total);

                // Initialize
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hist_total), [&](int idx) {
                    hist_r(idx) = 0.0;
                    hist_i(idx) = 0.0;
                });

                team.team_barrier();

                // First parallel_for: OFFSET range (pstart, pend)
                // This is what the scatter phase does
                const int pstart = offsets(tile_id);
                const int pend = offsets(tile_id + 1);

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, pstart, pend), [&](int ip) {
                    // Simulate scatter: write to histogram based on "particle" index
                    int local_ip = ip - pstart;
                    int hist_idx = local_ip % hist_total;
                    Kokkos::atomic_add(&hist_r(hist_idx), 1.0);
                    Kokkos::atomic_add(&hist_i(hist_idx), 0.5);
                });

                team.team_barrier();

                // Second parallel_for: range (0, hist_total)
                // Different range than above!
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hist_total), [&](int idx) {
                    const int hx_i = idx % hx;
                    const int hy_i = (idx / hx) % hy;
                    const int hz_i = idx / (hx * hy);

                    const int local_x = tile_x0 + hx_i - half_left;
                    const int local_y = tile_y0 + hy_i - half_left;
                    const int local_z = tile_z0 + hz_i - half_left;

                    if (local_x < -nghost || local_x >= nx + nghost ||
                        local_y < -nghost || local_y >= ny + nghost ||
                        local_z < -nghost || local_z >= nz + nghost) {
                        return;
                    }

                    const int gx = local_x + nghost;
                    const int gy = local_y + nghost;
                    const int gz = local_z + nghost;

                    const double r = hist_r(idx);
                    const double i = hist_i(idx);
                    if (r != 0.0 || i != 0.0) {
                        Kokkos::atomic_add(&grid(gx, gy, gz), complex_t(r, i));
                    }
                });
            });

        Kokkos::fence();

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Completed in " << ms << " ms" << std::endl;
    }
    Kokkos::finalize();
    return 0;
}