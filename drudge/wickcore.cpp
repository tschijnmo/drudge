/** Implementation of core wick expansion.
 *
 * Functions in this module are basically direct translation of the initial
 * Python version.
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include <Python.h>

/** Vectors of vector indices.
 */

using Vecs = std::vector<size_t>;

/** Contractions.
 *
 * Different from the Python version, here for each pivot vector, we just store
 * a list of vectors giving non-zero contractions with it.
 */

using Contrs = std::vector<Vecs>;

//
// Internal functions
// ==================
//

/** Add Wick schemes recursively.
 *
 * This is the core recursive function.  Zero will be returned on success.
 */

static int add_wick(PyObject* schemes, std::vector<bool>& avail, size_t pivot,
    Vecs& contred, const Vecs& vec_order, const Contrs& contrs)
{
    size_t n_vecs = avail.size();
    bool contr_all = vec_order.empty();

    // Find the actual pivot, which has to be available.
    //
    // pivot = next(i for i in range(pivot, n_vecs - 1) if avail[i])
    for (; pivot < n_vecs && !avail[pivot]; ++pivot) {
    }

    if (pivot == n_vecs) {
        // When everything is decided.

        if (!contr_all || std::none_of(avail.begin(), avail.end(),
                              [](bool i) { return i; })) {

            PyObject* scheme = PyTuple_New(2);
            if (scheme == NULL) {
                return 1;
            }

            PyObject* perm = PyList_New(n_vecs);
            if (perm == NULL) {
                Py_DECREF(scheme);
                return 1;
            }
            PyTuple_SET_ITEM(scheme, 0, perm); // Ownership of perm is stolen.

            size_t n_contred = contred.size();
            PyObject* n_contred_py = PyLong_FromSize_t(n_contred);
            if (n_contred_py == NULL) {
                Py_DECREF(scheme);
                return 1;
            }
            PyTuple_SET_ITEM(scheme, 1, n_contred_py);

            size_t i = 0; // Next index to write vector to.
            for (; i < n_contred; ++i) {
                PyObject* curr_vec = PyLong_FromSize_t(contred[i]);
                if (curr_vec == NULL) {
                    Py_DECREF(scheme);
                    return 1;
                }
                PyList_SET_ITEM(perm, i, curr_vec);
            }

            // This loop will be can skipped when all vectors are contracted.
            if (!contr_all) {
                for (auto j = vec_order.begin(); j != vec_order.end(); ++j) {
                    size_t vec_idx = *j;
                    if (avail[vec_idx]) {
                        PyObject* curr_vec = PyLong_FromSize_t(vec_idx);
                        if (curr_vec == NULL) {
                            Py_DECREF(scheme);
                            return 1;
                        }
                        PyList_SET_ITEM(perm, i++, curr_vec);
                    }
                };
            }

            assert(i == n_vecs);

            auto stat = PyList_Append(schemes, scheme);
            if (stat != 0) {
                return 1;
            }

            Py_DECREF(scheme);
        }

        return 0;
    }

    const auto& pivot_contrs = contrs[pivot];
    if (contr_all && pivot_contrs.empty()) {
        return 0;
    }

    if (!contr_all) {
        add_wick(schemes, avail, pivot + 1, contred, vec_order, contrs);
    }

    avail[pivot] = false;
    contred.push_back(pivot);
    std::for_each(pivot_contrs.begin(), pivot_contrs.end(), [&](auto vec_idx) {
        if (avail[vec_idx]) {
            avail[vec_idx] = false;
            contred.push_back(vec_idx);

            add_wick(schemes, avail, pivot + 1, contred, vec_order, contrs);

            avail[vec_idx] = true;
            contred.pop_back();
        }
    });
    avail[pivot] = true;
    contred.pop_back();

    return 0;
}

//
// Public functions
// ================
//

/** Docstring for the wickcore module.
 */

static const char* compose_wick_docstring
    = R"__doc__(Compose all Wick expansion schemes.

All Wick expansion schemes from the given vector order and contractions will be
returned.  This function has exactly the same interface and semantics as the
corresponding Python function.

)__doc__";

/** Generate all Wick composition schemes.
 */

static PyObject* compose_wick_func(
    PyObject* self, PyObject* args, PyObject* keywds)
{

    //
    // Parse input arguments.
    //

    PyObject* vec_order_arg;
    PyObject* contrs_arg;

    static char* kwlist[] = { "vec_order", "contrs", NULL };

    auto arg_stat = PyArg_ParseTupleAndKeywords(
        args, keywds, "OO", kwlist, &vec_order_arg, &contrs_arg);
    if (!arg_stat) {
        return NULL;
    }

    // Check contraction first, since it always has the correct number of
    // vectors.

    if (PySequence_Check(contrs_arg) != 1) {
        PyErr_SetString(
            PyExc_TypeError, "Invalid contractions, expecting sequence");
        return NULL;
    }
    size_t n_vecs = PySequence_Size(contrs_arg);

    bool contr_all = false;
    if (vec_order_arg == Py_None) {
        contr_all = true;
    } else if (PySequence_Check(vec_order_arg)) {
        if (PySequence_Size(vec_order_arg) != static_cast<Py_ssize_t>(n_vecs)) {
            PyErr_SetString(PyExc_ValueError,
                "Invalid vector order and contractions, inconsistent size");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
            "Invalid vector order, expecting None or sequence");
        return NULL;
    }

    if (n_vecs < 2) {
        PyErr_SetString(PyExc_ValueError,
            "Invalid vectors, need at least two vectors to contract");
        return NULL;
    }

    //
    // Translate input parameters
    //

    Vecs vec_order{};

    if (!contr_all) {
        vec_order.reserve(n_vecs);
        for (size_t i = 0; i < n_vecs; ++i) {
            PyObject* entry = PySequence_GetItem(vec_order_arg, i);
            if (!PyLong_Check(entry)) {
                PyErr_SetString(PyExc_TypeError,
                    "Invalid vector order entry, expecting integer");
                Py_DECREF(entry);
                return NULL;
            }

            size_t vec_idx = PyLong_AsSize_t(entry);
            Py_DECREF(entry);
            if (PyErr_Occurred()) {
                return NULL;
            }

            vec_order.push_back(vec_idx);
        }
    }

    Contrs contrs(n_vecs);

    for (size_t i = 0; i < n_vecs; ++i) {
        PyObject* entry = PySequence_GetItem(contrs_arg, i);
        if (!PyDict_Check(entry)) {
            // We only support dictionary, rather than general mapping.
            PyErr_SetString(
                PyExc_TypeError, "Invalid contraction, expecting dict");
            Py_DECREF(entry);
            return NULL;
        }

        PyObject* key;
        PyObject* value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(entry, &pos, &key, &value)) {
            if (!PyLong_Check(key)) {
                PyErr_SetString(PyExc_TypeError,
                    "Invalid key in contraction, expecting integer");
                return NULL;
            }

            size_t vec_idx = PyLong_AsSize_t(key);
            if (PyErr_Occurred()) {
                return NULL;
            }

            contrs[i].push_back(vec_idx);
        }

        Py_DECREF(entry);
    }

    //
    // Decision tree status
    //

    std::vector<bool> avail(n_vecs, true);

    Vecs contred{};

    //
    // Prepare the output
    //

    PyObject* schemes = PyList_New(0);
    if (schemes == NULL) {
        return NULL;
    }

    //
    // Run the core recursion.
    //
    auto stat = add_wick(schemes, avail, 0, contred, vec_order, contrs);

    if (stat == 0) {
        return schemes;
    } else {
        return NULL;
    }
}

//
// Python module definition
// ========================
//

/** Docstring for the wickcore module.
 */

static const char* wickcore_docstring = R"__doc__(Core Wick expansion utilities.

This module contains core functions to get all possible compositions of Wick
contractions.

)__doc__";

/** Methods in the wickcore module.
 */

static PyMethodDef wickcore_methods[]
    = { { "compose_wick", (PyCFunction)compose_wick_func,
            METH_VARARGS | METH_KEYWORDS, compose_wick_docstring },
        { NULL, NULL, 0, NULL } };

/** Executes the initialization of the wickcore module.
 *
 * Nothing much is done here, kept for future extensibility.
 */

static int wickcore_exec(PyObject* m) { return 0; }

/** Slots for for wickcore module definition.
 */

static struct PyModuleDef_Slot wickcore_slots[] = {
    { Py_mod_exec, (void*)wickcore_exec }, { 0, NULL },
};

/** Wick core module definition.
 */

// clang-format off

static struct PyModuleDef wickcore_module = {
    PyModuleDef_HEAD_INIT,
    "drudge.wickcore",
    wickcore_docstring,
    0, // m-size
    wickcore_methods,
    wickcore_slots,
    NULL, // Transverse
    NULL, // Clear
    NULL  // Free
};

// clang-format on

/** The published wickcore function.
 */

PyMODINIT_FUNC PyInit_wickcore(void)
{
    return PyModuleDef_Init(&wickcore_module);
}
