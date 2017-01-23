/** Implementation of canonpy.
 */

#include <canonpy.h>

#include <Python.h>

#include <algorithm>
#include <string>

#include <libcanon/perm.h>

using libcanon::Simple_perm;

//
// Perm class
// ==========
//
// Internal functions
// ------------------
//
// These functions are not directly set to Python types but are used
// internally.  They are also used by other parts of this extension.
//

/** Builds a tuple object for a permutation.
 *
 * This function creates a pair, where the first field is a list of integers
 * for the preimage array, and the second field is the accompanied action,
 * which is also encoded as an integer.
 */

static PyObject* build_perm_to_tuple(const Simple_perm& perm);

/** Builds a permutation from its construction arguments.
 *
 * An iterable of positive integers for the pre-image array needs to be given
 * as the first argument.  The accompanied action can be optionally given as
 * another integral argument, or by the keyword ``acc``.
 *
 * If the arguments are not valid, a Perm of size zero is going to be returned.
 *
 * This function is designed to be compatible with the result from the function
 * `build_perm_to_tuple`.  However, more general input format is accepted.
 */

static Simple_perm make_perm_from_args(PyObject* args, PyObject* kwargs);

//
// Interface functions
// -------------------
//

const static char* perm_getnewargs_doc
    = "Get the arguments for new to construct the Perm.";

static PyObject* perm_getnewargs(Perm_object* self)
{
    // Here we directly use the tuple format of a perm.

    return build_perm_to_tuple(self->perm);
}

/** Deallocates a perm instance.
 */

static void perm_dealloc(Perm_object* self)
{
    perm.perm.~Simple_perm();

    // For subclassing.
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Forms the string representation of a Perm object.
 */

static PyObject* perm_repr(Perm_object* self)
{
    const Simple_perm& perm = self->perm;

    std::wstring repr("Perm(");

    size_t size = perm.size();

    if (size > 0) {
        for (size_t i = 0; i < size; ++i) {
            if (i == 0) {
                repr.append('[');
            } else {
                repr.append(", ");
            }
            repr.append(std::to_wstring(perm >> i));
        }
        repr.append(']');

        // Add the accompanied action only when we need.
        char acc = perm.acc();
        if (acc != 0) {
            repr.append(", ");
            repr.append(std::to_wstring(acc));
        }
    }

    // This is used for empty or non-empty permutation.
    repr.append(')');

    return PyUnicode_FromUnicode(repr.data(), repr.size());
}

/** Creates a new Perm object.
 *
 * The actual work is delegated to the Python/Perm interface function.
 */

static PyObject* perm_new(PyTypeObject* type, PyObject* args, PyObject* kwargs)
{
    Perm_object* self;

    // Pay attention to subclassing.
    self = (Perm_object*)type->tp_alloc(type, 0);

    if (!self)
        return NULL;

    Simple_perm perm = get_perm_from_args(args, kwargs);

    if (perm.size() > 0) {
        new (&self->perm) Simple_perm(std::move(perm));
        return (PyObject*)self;
    } else {
        return NULL;
    }
}

//
// Class definition
// ----------------
//

/** Methods for Perm objects.
 */

static PyMethodDef perm_methods[] = {
    { "__getnewargs__", (PyCFunction)perm_getnewargs, METH_NOARGS,
        perm_getnewargs_doc },
    { NULL, NULL } /* sentinel */
};

/** Sequence operations for Perm objects.
 *
 * Here we only support size and pre-image query.
 */

// clang-format off
static PySequenceMethods perm_as_sequence = {
    (lenfunc)perm_length,                       /* sq_length */
    0,                                          /* sq_concat */
    0,                                          /* sq_repeat */
    (ssizeargfunc)perm_item,                    /* sq_item */
    0,                                          /* sq_slice */
    0,                                          /* sq_ass_item */
    0,                                          /* sq_ass_slice */
    0                                           /* sq_contains */
};
// clang-format on

/** Accessors for Perms.
 *
 * The accompanied action query is made here.
 */

// clang-format off
static PyGetSetDef perm_getsets[] = {
    { "acc", (getter)perm_get_acc, NULL, "The accompanied action.", NULL },
    { NULL }
};
// clang-format on

/** Perm type doc string.
  */

static char* perm_doc =
    R"__doc__(Permutation of points with accompanied action.

Permutations can be constructed from an iterable giving the pre-image of the
points and an optional integral value for the accompanied action.  The
accompanied action can be given positionally or by the keyword ``acc``, and it
will be manipulated according to the convention in libcanon.

Querying the length of a Perm object gives the size of the permutation domain,
while indexing it gives the pre-image of the given integral point.  The
accompanied action can be obtained by getting the attribute ``acc``.
Otherwise, this data type is mostly opaque.

)__doc__"

/** Type definition for Perm class.
 */

// clang-format off
static PyTypeObject perm_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "drudge.canonpy.Perm",
    sizeof(Perm_object),
    0,
    (destructor) perm_dealloc,                  /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_reserved */
    (reprfunc) perm_repr,                       /* tp_repr */
    0,                                          /* tp_as_number */
    &perm_as_sequence,                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    (hashfunc) perm_hash,                       /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0, /* In main. */                           /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    perm_doc,                                   /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    perm_methods,                               /* tp_methods */
    0,                                          /* tp_members */
    perm_getsets,                               /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    perm_new,                                   /* tp_new */
    0,                                          /* tp_free */
};
// clang-format on

//
// Python module definition
// ========================
//

/** Docstring for the canonpy module.
 */

static const char* canonpy_docstring
    = R"__doc__(Canonpy, simple wrapper over libcanon for Python.

This wrapper of libcanon is directly targeted towards usage using drudge.
Here, we have a class `Perm`, which wraps over the `Simple_perm` class in
libcanon, another class `SimsTransv`, which wraps over the `Sims_trasv` class
for `Simple_perm`.  And we also have the function `canon_eldag` to canonicalize
an Eldag.

)__doc__";

/** Methods in the canonpy module.
 */

static PyMethodDef canonpy_methods[] = { { NULL, NULL, 0, NULL } };

/** Executes the initialization of the canonpy module.
 */

static int canopy_exec(PyObject* m)
{
    //
    // Add the class for Perm.
    //

    perm_type.tp_getattro = PyObject_GenericGetAttr;
    if (PyType_Ready(&perm_type) < 0)
        return NULL;
    Py_INCREF(&perm_type);
    PyModule_AddObject(m, "Perm", (PyObject*)&perm_type);

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
