/*
Copyright (c) 2016, Blue Brain Project
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <vector>

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/mpi/nrnmpi.h"
#include "coreneuron/mpi/nrnmpi_impl.h"

namespace coreneuron {

/** display global mechanism count */
void write_mech_report() {
    /// mechanim count across all gids, local to rank
    const auto n_memb_func = corenrn.get_memb_funcs().size();
    std::vector<unsigned long long> local_mech_count(n_memb_func, 0);

    /// each gid record goes on separate row, only check non-empty threads
    for (size_t i = 0; i < nrn_nthread; i++) {
        const auto& nt = nrn_threads[i];
        for (auto* tml = nt.tml; tml; tml = tml->next) {
            const int type = tml->index;
            const auto& ml = tml->ml;
            local_mech_count[type] += ml->nodecount;
        }
    }

    std::vector<unsigned long long> total_mech_count(n_memb_func);

#if NRNMPI
    /// get global sum of all mechanism instances
    MPI_Allreduce(&local_mech_count[0],
                  &total_mech_count[0],
                  local_mech_count.size(),
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);

#else
    total_mech_count = local_mech_count;
#endif
    /// print global stats to stdout
    if (nrnmpi_myid == 0) {
        printf("\n================ MECHANISMS COUNT BY TYPE ==================\n");
        printf("%4s %20s %10s\n", "Id", "Name", "Count");
        for (size_t i = 0; i < total_mech_count.size(); i++) {
            printf("%4lu %20s %10lld\n", i, nrn_get_mechname(i), total_mech_count[i]);
        }
        printf("=============================================================\n");
    }
}

}  // namespace coreneuron
