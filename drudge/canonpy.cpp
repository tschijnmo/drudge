/** Implementation of canonpy.
 */

#include <canonpy.h>

#include <Python.h>

#include <algorithm>
#include <string>
#include <vector>

#include <libcanon/perm.h>
#include <libcanon/sims.h>

using libcanon::Simple_perm;
using libcanon::Point;
using libcanon::Point_vec;

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

static PyObject* build_perm_to_tuple(const Simple_perm& perm)
{

    PyObject* pre_images = NULL;
    PyObject* res = NULL;
    PyObject* acc = NULL;
    size_t size = perm.size();

    pre_images = PyList_New(size);
    if (!pre_images)
        goto error;
    for (size_t i = 0; i < size; ++i) {
        PyObject* curr = Py_BuildValue("n", perm >> i);
        if (curr) {
            PyList_SetItem(pre_images, i, curr);
        } else {
            goto error;
        }
    }

    acc = Py_BuildValue("b", perm.acc());
    if (!acc)
        goto error;

    res = PyTuple_New(2);
    if (!res)
        goto error;

    PyTuple_SET_ITEM(res, 0, pre_images);
    PyTuple_SET_ITEM(res, 1, acc);

    return (PyObject*)res;

error:
    Py_XDECREF(pre_images);
    Py_XDECREF(res);
    Py_XDECREF(acc);
    return NULL;
}

/** Builds a permutation from its construction arguments.
 *
 * An iterable of positive integers for the pre-image array needs to be given
 * as the first argument.  The accompanied action can be optionally given as
 * another integral argument, or by the keyword ``acc``.
 *
 * If the arguments are not valid, an integer exception will be thrown and the
 * Python exception will be set.
 *
 * This function is designed to be compatible with the result from the function
 * `build_perm_to_tuple`.  However, more general input format is accepted.
 */

static Simple_perm make_perm_from_args(PyObject* args, PyObject* kwargs)
{
    PyObject* pre_images;
    char acc = 0;

    // We only need a simple internal code, actual error is set to the Python
    // stack.
    constexpr int err_code = 1;

    static char* kwlist[] = { "pre_images", "acc", NULL };

    auto args_stat = PyArg_ParseTupleAndKeywords(
        args, kwargs, "O|b", kwlist, &pre_images, &acc);

    if (!args_stat)
        throw err_code;

    Point_vec pre_images_vec{};
    std::vector<bool> image_set{};

    PyObject* pre_images_iter = PyObject_GetIter(pre_images);
    if (!pre_images_iter)
        throw err_code;

    // Iterator of pre-images is always going to be decrefed after the
    // following block.  This boolean controls if we are going to return or
    // throw.
    bool if_err = false;

    try {
        PyObject* pre_image_obj;
        while ((pre_image_obj = PyIter_Next(pre_images_iter))) {

            if (!PyLong_Check(pre_image_obj)) {
                PyErr_SetString(PyExc_TypeError, "Non-integral point given");
                throw err_code;
            }
            Point pre_image = PyLong_AsUnsignedLong(pre_image_obj);

            // Release reference right here since its content is already
            // extracted.  In this way, the error handling does not need to
            // worry about it any more.
            Py_DECREF(pre_image_obj);

            if (PyErr_Occurred()) {
                throw err_code;
            }

            pre_images_vec.push_back(pre_image);
            size_t req_size = pre_image + 1;
            if (image_set.size() < req_size)
                image_set.resize(req_size, false);
            if (image_set[pre_image]) {
                std::string err_msg("The image of ");
                err_msg.append(std::to_string(pre_image));
                err_msg.append(" has already been set.");
                PyErr_SetString(PyExc_ValueError, err_msg.c_str());
                throw err_code;
            } else {
                image_set[pre_image] = true;
            }
        }

        // Non StopIteration error.
        if (PyErr_Occurred()) {
            throw err_code;
        }

        auto first_not_set
            = std::find(image_set.begin(), image_set.end(), false);
        if (first_not_set != image_set.end()) {
            std::string err_msg("The image of ");
            err_msg.append(std::to_string(first_not_set - image_set.begin()));
            err_msg.append(" is not set.");
            PyErr_SetString(PyExc_ValueError, err_msg.c_str());
            throw err_code;
        }

    } catch (int) {
        if_err = true;
    }

    Py_DECREF(pre_images_iter);
    if (if_err) {
        throw err_code;
    } else {
        return Simple_perm(std::move(pre_images_vec), acc);
    }
}

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

/** Gets the size of the permutation domain of a Perm.
 */

static Py_ssize_t perm_length(Perm_object* self)
{
    // The type should be the same, size_t and Py_ssize_t.

    return self->perm.size();
}

/** Gets the pre-image of a point.
 *
 * A new integer object will be built by this function.
 */

static PyObject* perm_item(Perm_object* self, Py_ssize_t i)
{
    if (i < 0) {
        PyErr_SetString(PyExc_IndexError, "Points should be positive.");
        return NULL;
    }
    size_t idx = i;
    if (idx >= self->perm.size()) {
        PyErr_SetString(PyExc_IndexError, "Point outside permutation domain");
        return NULL;
    }

    return Py_BuildValue("n", self->perm >> i);
}

/** Gets the accompanied action of a Perm.
 */

static PyObject* perm_get_acc(Perm_object* self, void* closure)
{
    // Note that the accompanied action is a byte in simple perm.

    return Py_BuildValue("b", self->perm.acc());
}

/** Deallocates a perm instance.
 */

static void perm_dealloc(Perm_object* self)
{
    self->perm.~Simple_perm();

    // For subclassing.
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Forms the string representation of a Perm object.
 */

static PyObject* perm_repr(Perm_object* self)
{
    const Simple_perm& perm = self->perm;

    std::wstring repr(L"Perm(");

    size_t size = perm.size();

    if (size > 0) {
        for (size_t i = 0; i < size; ++i) {
            if (i == 0) {
                repr.append(L"[");
            } else {
                repr.append(L", ");
            }
            repr.append(std::to_wstring(perm >> i));
        }
        repr.append(L"]");

        // Add the accompanied action only when we need.
        char acc = perm.acc();
        if (acc != 0) {
            repr.append(L", ");
            repr.append(std::to_wstring(acc));
        }
    }

    // This is used for empty or non-empty permutation.
    repr.append(L")");

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

    Simple_perm perm{};
    try {
        perm = make_perm_from_args(args, kwargs);
    } catch (int) {
        Py_DECREF(self);
        return NULL;
    }

    new (&self->perm) Simple_perm(std::move(perm));
    return (PyObject*)self;
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

static const char* perm_doc =
    R"__doc__(Permutation of points with accompanied action.

Permutations can be constructed from an iterable giving the pre-image of the
points and an optional integral value for the accompanied action.  The
accompanied action can be given positionally or by the keyword ``acc``, and it
will be manipulated according to the convention in libcanon.

Querying the length of a Perm object gives the size of the permutation domain,
while indexing it gives the pre-image of the given integral point.  The
accompanied action can be obtained by getting the attribute ``acc``.
Otherwise, this data type is mostly opaque.

)__doc__";

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
    0,                                          /* tp_hash */
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
// Permutation group type definition
// =================================
//
// Internal functions
// ------------------
//

//
// Interface functions
// -------------------
//

const static char* group_getnewargs_doc
    = "Get the arguments for new to construct the Group.";

static PyObject* group_getnewargs(Group_object* self)
{
    // Here we directly use the list format of a perm.

    return serialize_group(self->transv.get());
}

/** Deallocates a group instance.
 */

static void group_dealloc(Group_object* self)
{
    self->transv.~Transv_ptr();

    // For subclassing.
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/** Creates a new permutation group object.
 *
 * The actual work is delegated to the core functions.
 */

static PyObject* group_new(PyTypeObject* type, PyObject* args, PyObject* kwargs)
{
    Perm_object* self;

    // Pay attention to subclassing.
    self = (Perm_object*)type->tp_alloc(type, 0);

    if (!self)
        return NULL;

    Transv_ptr trasv = build_sims_transv_from_args(args, kwargs);

    if (!trasv) {
        Py_DECREF(self);
        return NULL;
    }

    self->transv = std::move(transv);
    return (PyObject*)self;
}

//
// Class definition
// ----------------
//

/** Methods for permutation group objects.
 */

static PyMethodDef group_methods[] = {
    { "__getnewargs__", (PyCFunction)group_getnewargs, METH_NOARGS,
        group_getnewargs_doc },
    { NULL, NULL } /* sentinel */
};

/** Sims transversal type doc string.
  */

static const char* group_doc =
    R"__doc__(Permutations groups.

To create a permutation group, an iterable of Perm objects or pre-image array
action pair can be given for the generators of the group.  Then the
Schreier-Sims algorithm in libcanon will be invoked to generate the Sims
transversal system, which will be stored internally for the group.  This class
is mostly designed to be used to give input for the Eldag canonicalization
facility.  So it is basically an opaque object after its creation.

Internally, the transversal system can also be constructed directly from the
transversal system, without going through the Schreier-Sims algorithm.
However, that is more intended for serialization rather than direct user
invocation.

)__doc__";

/** Type definition for permutation group class.
 */

// clang-format off
static PyTypeObject group_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "drudge.canonpy.Group",
    sizeof(Group_object),
    0,
    (destructor) group_dealloc,                 /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_reserved */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0, /* In main. */                           /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    group_doc,                                  /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    group_methods,                              /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    group_new,                                   /* tp_new */
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

static int canonpy_exec(PyObject* m)
{
    //
    // Add the class for Perm.
    //

    perm_type.tp_getattro = PyObject_GenericGetAttr;
    if (PyType_Ready(&perm_type) < 0)
        return -1;
    Py_INCREF(&perm_type);
    PyModule_AddObject(m, "Perm", (PyObject*)&perm_type);

    //
    // Add the class for Group.
    //

    group_type.tp_getattro = PyObject_GenericGetAttr;
    if (PyType_Ready(&group_type) < 0)
        return -1;
    Py_INCREF(&group_type);
    PyModule_AddObject(m, "Group", (PyObject*)&group_type);

    return 0;
}

/** Slots for for canonpy module definition.
 */

static struct PyModuleDef_Slot canonpy_slots[] = {
    { Py_mod_exec, (void*)canonpy_exec }, { 0, NULL },
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
