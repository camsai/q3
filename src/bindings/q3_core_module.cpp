/**
 * q3_core — nanobind Python extension module.
 *
 * Exposes run_scf() and run_nscf() as Python callables that invoke the Q3
 * DFT engine in-process (Mode B) rather than via subprocess (Mode A).
 *
 * Usage from Python::
 *
 *     import q3_core
 *
 *     # Returns a JSON string (same schema as results.json)
 *     result_json = q3_core.run_scf("path/to/input.yml")
 *     result_json = q3_core.run_nscf("path/to/nscf_input.yml")
 *
 * Thread safety
 * -------------
 * The module releases the GIL during C++ computation (``nb::call_guard``
 * is NOT used here because Q3 is not thread-safe; GIL is held throughout
 * the C++ call to prevent concurrent invocations).
 */

#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings/q3_core_api.hpp"
#include "io/addon_registration.hpp"

namespace nb = nanobind;

NB_MODULE(q3_core, module) {
    q3::addons::register_all();
    module.doc() = R"doc(
        q3_core — In-process Q3 DFT engine (Mode B).

        Functions
        ---------
        run_scf(input_path, write_output_files=True) -> str
            Run a self-consistent field calculation.  Returns a JSON string.

        run_nscf(input_path, write_output_files=True) -> str
            Run a non-self-consistent field calculation.  Returns a JSON string.

        Notes
        -----
        Call reset_counters() between successive calls if profiling metrics
        are needed per-invocation (run_scf/run_nscf reset automatically).
    )doc";

    module.def(
        "run_scf",
        [](const std::string& input_path, bool write_output_files) -> std::string {
            return q3::bindings::run_scf(input_path, write_output_files);
        },
        nb::arg("input_path"),
        nb::arg("write_output_files") = true,
        R"doc(
            Run a self-consistent field calculation in-process.

            Parameters
            ----------
            input_path : str
                Path to a Q3 YAML input file (calculation = scf).
            write_output_files : bool, optional
                If True (default), write results.json and checkpoint files.

            Returns
            -------
            str
                JSON string with the results (same schema as results.json).

            Raises
            ------
            RuntimeError
                If SCF fails or the input is invalid.
        )doc");

    module.def(
        "run_nscf",
        [](const std::string& input_path, bool write_output_files) -> std::string {
            return q3::bindings::run_nscf(input_path, write_output_files);
        },
        nb::arg("input_path"),
        nb::arg("write_output_files") = true,
        R"doc(
            Run a non-self-consistent field calculation in-process.

            Parameters
            ----------
            input_path : str
                Path to a Q3 YAML input file (calculation = nscf) with
                kpoints_list specified.
            write_output_files : bool, optional
                If True (default), write results.json.

            Returns
            -------
            str
                JSON string with eigenvalues at each requested k-point.

            Raises
            ------
            RuntimeError
                If a valid SCF checkpoint is not found.
        )doc");

    module.def(
        "reset_counters",
        []() { q3::ProfilingCounters::instance().reset(); },
        R"doc(
            Reset all internal profiling counters.

            run_scf() and run_nscf() call this automatically before each
            invocation.  Call manually only if you need to reset counters
            mid-run (unusual).
        )doc");

    nb::class_<q3::bindings::SCFSolver>(module, "SCFSolver")
        .def(nb::init<const std::string&>(), nb::arg("input_path"))
        .def("solve",
             &q3::bindings::SCFSolver::solve,
             nb::arg("coordinates"),
             nb::arg("write_output_files") = true,
             R"doc(
                 Solve the SCF problem for the given atomic coordinates.

                 Parameters
                 ----------
                 coordinates : list of list of float
                     Atomic coordinates (crystal/fractional) for all atoms.
                 write_output_files : bool, optional
                     If True (default), write results.json and checkpoint files.

                 Returns
                 -------
                 str
                     JSON string with the results.
             )doc");
}
