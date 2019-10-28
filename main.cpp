#include "MPFADSolver.h"

using namespace std;
using namespace moab;

int main (int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPFADSolver* solver = new MPFADSolver();

    solver->load_file("test_case_linear_anisotropic_part.h5m");
    solver->run();
    printf("Writing file\n");
    solver->write_file("test_case_linear_anisotropic_part_result.h5m");
    printf("Done\n");

    delete solver;

    MPI_Finalize();

    return 0;
}
