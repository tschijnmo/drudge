/** Implementation of canonpy.
 */

#include <canonpy.h>

#include <Python.h>

//
// Python module definition
// ========================
//

/** Docstring for the canonpy module.
 */

static const char* canonpy_docstring = R"CANONPYDOC(
Canonpy, simple wrapper over libcanon for Python.

)CANONPYDOC";

/** Methods in the canonpy module.
 */

static PyMethodDef canonpy_methods[] = { { NULL, NULL, 0, NULL } };

/** Executes the initialization of the canonpy module.
 */

static int canopy_exec(PyObject* m)
{
    // TODO: add module attributes, mostly types here.

    return 0;
}

/** Slots for for canonpy module definition.
 */

static struct PyModuleDef_Slot canonpy_slots[] = {
    { Py_mod_exec, canonpy_exec }, { 0, NULL },
};

/** Canonpy module definition.
 */

// clang-format off

static struct PyModuleDef canonpy_module = {
    PyModuleDef_HEAD_INIT,
    "drudge.canonpy",
    canonpy_docstring,
    0, // m-size
    canonpy_methods,
    canonpy_slots,
    NULL, // Transverse
    NULL, // Clear
    NULL  // Free
};

// clang-format on

/** The published canonpy function.
 */

PyMODINIT_FUNC PyInit_canonpy(void)
{
    return PyModuleDef_Init(&canonpy_module);
}
