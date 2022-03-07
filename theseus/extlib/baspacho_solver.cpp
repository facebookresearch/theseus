
#include <torch/extension.h>

#include "baspacho/baspacho/Solver.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace testing;
using namespace std;

class SymbolicDecomposition {
   public:
    SymbolicDecomposition(int x, double y) {
        auto colBlocks = randomCols(115, 0.03, 57);
        colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
        vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 2, 5, 47);
        solver = createSolver({}, paramSize, ss);
        cout << "created: " << x << ", " << y << endl;
    }

    BaSpaCho::SolverPtr solver;
};

PYBIND11_MODULE(baspacho_solver, m) {
    m.doc() = "Python bindings for BaSpaCho solver";
    py::class_<SymbolicDecomposition>(m, "SymbolicDecomposition",
                                      "Symbolic decomposition")
        .def(py::init<int, double>(),
             "Initialization, it computes the fill-reducing permutation,\n"
             "performs the symbolic factorization, preparing the data "
             "structures",
             py::arg("x"), py::arg("y"));
};