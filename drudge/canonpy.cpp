/** Implementation of canonpy.
 */

#include <canonpy.h>

#include <Python.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <libcanon/eldag.h>
#include <libcanon/perm.h>
#include <libcanon/sims.h>

using libcanon::Simple_perm;
using libcanon::Point;
using libcanon::Point_vec;
using libcanon::build_sims_sys;
using libcanon::Eldag;
using libcanon::Node_symms;
using libcanon::canon_eldag;
using libcanon::Eldag_perm;

//
// General utilities
// =================
//
// Here we have some general utilities useful throughout the code.
//

/** Type for internal errors.
 *
 * Different from the common treatment of exceptions by goto in the CPython
 * code base, here goto cannot be liberally used due to the more complicated
 * goto rules in C++.  So here we choose to use C++ exception handling
 * facilities for this problem.  Since the actual problem are always set to the
 * Python exception facility, here we just use a simple data type to jump to an
 * internal location.
 */

using I_err = int;

/** A default value to be used for internal errors.
 *
 * Since normally the details of the problem should be set to the Python
 * exception stack rather than set to the C++ exception, this value can be
 * conveniently used to indicate the presence of a problem.
 */

static constexpr I_err err = 1;

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

    PyObject* res = NULL;
    PyObject* pre_images = NULL;
    PyObject* acc = NULL;
    size_t size = perm.size();

    try {
        pre_images = PyList_New(size);
        if (!pre_images) {
            throw err;
        }

        for (size_t i = 0; i < size; ++i) {
            PyObject* curr = Py_BuildValue("n", perm >> i);
            if (curr) {
                PyList_SetItem(pre_images, i, curr);
            } else {
                throw err;
            }
        }

        acc = Py_BuildValue("b", perm.acc());
        if (!acc) {
            throw err;
        }

        res = PyTuple_New(2);
        if (!res)
            throw err;

        PyTuple_SET_ITEM(res, 0, pre_images);
        PyTuple_SET_ITEM(res, 1, acc);

        return (PyObject*)res;

    } catch (I_err) {
        Py_XDECREF(pre_images);
        Py_XDECREF(res);
        Py_XDECREF(acc);
        return NULL;
    }
}

/** Builds a permutation from its construction arguments.
 *
 * An iterable of positive integers for the pre-image array needs to be given
 * as the first argument.  The accompanied action can be optionally given as
 * another integral argument, or by the keyword ``acc``.
 *
 * If the arguments are not valid, an internal exception will be thrown and the
 * Python exception will be set.
 *
 * This function is designed to be compatible with the result from the function
 * `build_perm_to_tuple`.  However, more general input format is accepted.
 */

static Simple_perm make_perm_from_args(PyObject* args, PyObject* kwargs)
{
    PyObject* pre_images;
    char acc = 0;

    static char* kwlist[] = { "pre_images", "acc", NULL };

    auto args_stat = PyArg_ParseTupleAndKeywords(
        args, kwargs, "O|b", kwlist, &pre_images, &acc);

    if (!args_stat)
        throw err;

    Point_vec pre_images_vec{};
    std::vector<bool> image_set{};

    PyObject* pre_images_iter = PyObject_GetIter(pre_images);
    if (!pre_images_iter)
        throw err;

    // Iterator of pre-images is always going to be decrefed after the
    // following block.  This boolean controls if we are going to return or
    // throw.
    bool if_err = false;

    try {
        PyObject* pre_image_obj;
        while ((pre_image_obj = PyIter_Next(pre_images_iter))) {

            if (!PyLong_Check(pre_image_obj)) {
                Py_DECREF(pre_image_obj);
                PyErr_SetString(PyExc_TypeError, "Non-integral point given");
                throw err;
            }
            Point pre_image = PyLong_AsSsize_t(pre_image_obj);

            // Release reference right here since its content is already
            // extracted.  In this way, the error handling does not need to
            // worry about it any more.
            Py_DECREF(pre_image_obj);

            if (PyErr_Occurred()) {
                throw err;
            }

            pre_images_vec.push_back(pre_image);
            size_t req_size = pre_image + 1;
            if (image_set.size() < req_size)
                image_set.resize(req_size, false);
            if (image_set[pre_image]) {
                std::string err_msg("The image of ");
                err_msg += std::to_string(pre_image);
                err_msg += " has already been set.";
                PyErr_SetString(PyExc_ValueError, err_msg.c_str());
                throw err;
            } else {
                image_set[pre_image] = true;
            }
        }

        // Non StopIteration error.
        if (PyErr_Occurred()) {
            throw err;
        }

        auto first_not_set
            = std::find(image_set.begin(), image_set.end(), false);
        if (first_not_set != image_set.end()) {
            std::string err_msg("The image of ");
            err_msg += std::to_string(first_not_set - image_set.begin());
            err_msg += " is not set.";
            PyErr_SetString(PyExc_ValueError, err_msg.c_str());
            throw err;
        }

    } catch (I_err) {
        if_err = true;
    }

    Py_DECREF(pre_images_iter);
    if (if_err) {
        throw err;
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

/** Serializes a group into a Python list.
 *
 * This function creates a new Python list for the given transversal system.
 * Each entry is for one level of transversal system, which is a tuple starting
 * with an integer for the anchor point, and then followed by a list of pairs
 * for the coset representative permutations.
 */

static PyObject* serialize_group(const Transv* transv)
{
    // Create a new list for the result.
    PyObject* res = PyList_New(0);
    if (!res) {
        return NULL;
    }
    constexpr int err_code = 1;

    for (; transv; transv = transv->next()) {

        // Initialize all of them to NULL, so that things can be handled
        // consistently on error.  No memory is going to be accidentally
        // touched by XDECREF.

        PyObject* pair = NULL;
        PyObject* target = NULL;
        PyObject* perms = NULL;
        PyObject* perm = NULL;

        try {
            pair = PyTuple_New(2);
            if (!pair) {
                throw err_code;
            }

            target = Py_BuildValue("n", transv->target());
            if (!target) {
                throw err_code;
            }
            PyTuple_SET_ITEM(pair, 0, target);
            target
                = NULL; // Reference is stolen, no need to handle it on error.

            perms = PyList_New(0);
            if (!perms) {
                throw err_code;
            }

            for (const auto& i : *transv) {
                perm = build_perm_to_tuple(i);
                if (!perm) {
                    throw err_code;
                }

                int stat = PyList_Append(perms, perm);
                if (stat != 0) {
                    throw err_code;
                }
                Py_DECREF(perm);
                perm = NULL;
            }

            PyTuple_SET_ITEM(pair, 1, perms);
            perms = NULL;

            int stat = PyList_Append(res, pair);
            if (stat != 0) {
                throw err_code;
            }
            Py_DECREF(pair);
            pair = NULL;

        } catch (int) {
            Py_XDECREF(res);
            Py_XDECREF(pair);
            Py_XDECREF(target);
            Py_XDECREF(perms);
            Py_XDECREF(perm);
            return NULL;
        }
    }

    // No need to handle memory issue here after the loop finished
    // successfully.  All references are transferred to the result list.

    return res;
}

/** Reads permutation generators from the given iterator.
 *
 * The front element should already be taken out. This function stoles the
 * references.
 */

std::vector<Simple_perm> read_gens(PyObject* front, PyObject* iter)
{
    std::vector<Simple_perm> gens{};
    size_t size; // Size of the permutation domain.

    do {
        if (PyObject_IsInstance(front, (PyObject*)&perm_type)) {
            gens.push_back(((Perm_object*)front)->perm);
        } else {

            Simple_perm gen = make_perm_from_args(front, NULL);

            if (gen.size() == 0) {
                goto error;
            }
            if (gens.empty()) {
                size = gen.size();
            } else if (gen.size() != size) {
                std::string err_msg("Generator on ");
                err_msg.append(std::to_string(gen.size()));
                err_msg.append(" points has been found, expecting ");
                err_msg.append(std::to_string(size));
                err_msg.append(". ");
                PyErr_SetString(PyExc_ValueError, err_msg.c_str());
                goto error;
            }

            gens.push_back(std::move(gen));
            continue;

        error:
            Py_DECREF(front);
            Py_DECREF(iter);
            return {};
        }

        Py_DECREF(front);
    } while ((front = PyIter_Next(iter)));
    Py_DECREF(iter);

    if (PyErr_Occurred()) {
        return {};
    }

    return gens;
}

/** Builds a Sims transversal from scratch.
 *
 * Note that this function steals references to the iterator for generators and
 * its front.
 */

Transv_ptr build_sims_scratch(PyObject* front, PyObject* iter)
{
    std::vector<Simple_perm> gens = read_gens(front, iter);
    if (gens.size() == 0) {
        return nullptr;
    }

    Transv_ptr res = build_sims_sys(gens.front().size(), std::move(gens));
    if (!res) {
        PyErr_SetString(PyExc_ValueError, "Identity group found.");
    }
    return res;
}

/** Build a Sims transversal system from transversals directly.
 *
 * Similar to the scratch mode function, here the references will be stolen for
 * the front element and iterator.
 */

Transv_ptr deserialize_sims(PyObject* front, PyObject* iter)
{
    Transv head(0, 1); // The dummy head.
    Transv* back = &head;

    constexpr int err_code = 1;

    do {

        try {

            // Here we still need some checking, since we might be working on
            // non-first elements from the iterable, which has not been checked.

            if (!PySequence_Check(front) || PySequence_Size(front) != 2) {
                PyErr_SetString(PyExc_ValueError, "Invalid transversal.");
                throw err_code;
            }
            PyObject* first = PySequence_GetItem(front, 0);
            if (!first) {
                throw err_code;
            }

            if (!PyLong_Check(first)) {
                PyErr_SetString(PyExc_TypeError, "Invalid target point.");
                Py_DECREF(first);
                throw err_code;
            }
            Point target = PyLong_AsUnsignedLong(first);
            Py_DECREF(first);

            PyObject* second = PySequence_GetItem(front, 1);
            if (!second) {
                throw err_code;
            }

            // We need at least one element in the transversal.  Both for
            // checking
            // and for the interface of read_gens.

            PyObject* gens_iter = PyObject_GetIter(second);
            Py_DECREF(second);
            if (!gens_iter) {
                throw err_code;
            }

            PyObject* gens_front = PyIter_Next(gens_iter);
            if (!gens_front) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(
                        PyExc_ValueError, "Empty coset representatives.");
                }

                Py_DECREF(gens_iter);
                throw err_code;
            }

            std::vector<Simple_perm> gens = read_gens(gens_front, gens_iter);
            if (gens.size() == 0) {
                throw err_code;
            }
            size_t size = gens.front().size();

            auto new_transv = std::make_unique<Transv>(target, size);
            for (auto& i : gens) {
                new_transv->insert(std::move(i));
            }

            back->set_next(std::move(new_transv));
            back = back->next();

            Py_DECREF(front);
            continue;

        } catch (int) {
            Py_DECREF(front);
            Py_DECREF(iter);
            return nullptr;
        }
    } while ((front = PyIter_Next(iter)));
    Py_DECREF(iter);

    return head.release_next();
}

/** Builds a Sims transversal system from Python arguments.
 *
 * The building has two modes of operation, scratch mode and de-serialisation
 * mode.  In either case, an iterable is expected as the only parameter, which
 * can be given under the keyword `gens`.
 *
 * Empty iterable is not considered valid.  If the first item from the iterable
 * is a pair containing an integral value on its first field and an iterable
 * value on its second field, the given iterable is considered to contain a
 * serialized transversal system, or it will be attempted to be built from
 * scratch.
 *
 * For invalid inputs, an empty unique pointer will be returned and the
 * exception for the Python stack will be set.
 */

static Transv_ptr build_sims_transv_from_args(PyObject* args, PyObject* kwargs)
{
    PyObject* input;

    static char* kwlist[] = { "gens", NULL };

    auto args_stat
        = PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &input);
    if (!args_stat) {
        return nullptr;
    }

    PyObject* input_iter = PyObject_GetIter(input);
    if (!input_iter) {
        return nullptr;
    }

    PyObject* front = PyIter_Next(input_iter);
    if (!front) {
        // Here we either had an error or reached end.

        if (!PyErr_Occurred()) {
            // When we reached the end, need to set our own error.
            PyErr_SetString(PyExc_ValueError, "No generator is given.");
        }

        Py_DECREF(input_iter);
        return nullptr;
    }

    // Here we first check the front to test if we are in scratch mode or
    // de-serialization mode.
    //
    // After the determination is finished, all local references are destroyed.
    // This is slightly wasteful in that some of them might be helpful for
    // later de-serialization.  But it helps with code clarity and modularity.

    // We assume we work in scratch mode, unless a strong indication is given
    // for the de-serialization mode.
    bool scratch = true;

    if (PySequence_Check(front) && PySequence_Size(front) == 2) {
        // Here it is possible that we are in de-serialization mode.
        PyObject* first = PySequence_GetItem(front, 0);
        PyObject* second = PySequence_GetItem(front, 1);

        if (PyLong_Check(first)) {
            PyObject* curr_transv = PyObject_GetIter(second);

            if (curr_transv) {
                scratch = false;
                Py_DECREF(curr_transv);
            } else {
                PyErr_Clear();
            }
        }

        Py_DECREF(first);
        Py_DECREF(second);
    }

    if (scratch) {
        return build_sims_scratch(front, input_iter);
    } else {
        return deserialize_sims(front, input_iter);
    }
}

//
// Interface functions
// -------------------
//

const static char* group_getnewargs_doc
    = "Get the arguments for new to construct the Group.";

static PyObject* group_getnewargs(Group_object* self)
{
    // Here we need to put the list format of the transversal into a tuple.
    PyObject* args;
    PyObject* transvs;

    args = PyTuple_New(1);
    if (!args) {
        return NULL;
    }

    transvs = serialize_group(self->transv.get());
    if (!transvs) {
        return NULL;
    }

    PyTuple_SET_ITEM(args, 0, transvs);

    return args;
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
    Group_object* self;

    // Pay attention to subclassing.
    self = (Group_object*)type->tp_alloc(type, 0);

    if (!self)
        return NULL;

    Transv_ptr transv = build_sims_transv_from_args(args, kwargs);

    if (!transv) {
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
    group_new,                                  /* tp_new */
    0,                                          /* tp_free */
};
// clang-format on

//
// Eldag canonicalization function
// ===============================
//
// Internal functions
// ------------------
//

/** Reads values from Python iterable into a vector.
 *
 * An integral value will be raised when something goes wrong, with the Python
 * exception set.  The get value functor should be a callable that accepts a
 * borrowed reference to the item and return the actual value.  Integral
 * exceptions can be thrown by this function after the Python exception has
 * been set.
 */

template <typename T, typename F>
static std::vector<T> read_py_iter(PyObject* iterable, F get_value)
{
    std::vector<T> res{};
    constexpr int err_code = 1;

    PyObject* iterator = PyObject_GetIter(iterable);

    if (iterator == NULL) {
        throw err_code;
    }

    PyObject* item;
    while ((item = PyIter_Next(iterator))) {

        try {
            res.push_back(get_value(item));
        } catch (int) {
            Py_DECREF(item);
            Py_DECREF(iterator);
            throw err_code;
        }

        Py_DECREF(item);
    }

    Py_DECREF(iterator);

    if (PyErr_Occurred()) {
        throw err_code;
    }

    return res;
}

/** Reads unsigned integral points from a Python iterable.
 */

static Point_vec read_points(PyObject* iterable)
{
    return read_py_iter<size_t>(iterable, [](PyObject* item) {
        if (!PyLong_Check(item)) {
            throw 1;
        }

        size_t value = PyLong_AsSize_t(item);

        if (PyErr_Occurred()) {
            throw 1;
        }

        return value;
    });
}

/** Reads pointer to Sims transversal from Python iterable.
 */

static Node_symms<Simple_perm> read_symms(PyObject* iterable)
{
    using Transv_ptr = const Sims_transv<Simple_perm>*;
    return read_py_iter<Transv_ptr>(iterable, [](PyObject* item) -> Transv_ptr {
        auto check = PyObject_Not(item);
        if (check == 1) {
            // This is an indication of a false value used by user for show
            // the absence of symmetries.
            return nullptr;
        } else if (check == -1) {
            // This means something wrong happened during the boolean
            // evaluation.
            throw 1;
        }

        if (!PyObject_IsInstance(item, (PyObject*)&group_type)) {
            PyErr_SetString(
                PyExc_TypeError, "Invalid symmetry, Group expected.");
        }
        Group_object* group = (Group_object*)item;

        return group->transv.get();
    });
}

/** Builds the Python results for Eldag canonicalization result.
 *
 * The result is a pair of the global nodes ordering given as a list and a list
 * of Perms for each of the nodes.
 */

static PyObject* build_canon_res(const Eldag_perm<Simple_perm>& canon_res)
{
    PyObject* res = NULL;
    PyObject* gl_order = NULL;
    PyObject* node_perms = NULL;

    size_t n_nodes = canon_res.partition.size();

    constexpr int err_code = 1;

    try {

        gl_order = PyList_New(n_nodes);
        if (!gl_order) {
            throw err_code;
        }
        for (size_t i = 0; i < n_nodes; ++i) {
            Point pre_img = canon_res.partition.get_pre_imgs()[i];
            PyObject* pre_img_obj = PyLong_FromSize_t(pre_img);
            if (!pre_img_obj) {
                throw err_code;
            }
            PyList_SetItem(gl_order, i, pre_img_obj);
        }

        node_perms = PyList_New(n_nodes);
        if (!node_perms) {
            throw err_code;
        }
        for (size_t i = 0; i < n_nodes; ++i) {
            auto& perm = canon_res.perms[i];
            if (!perm) {
                Py_INCREF(Py_None);
                PyList_SetItem(node_perms, i, Py_None);
            } else {
                Perm_object* perm_obj = PyObject_New(Perm_object, &perm_type);
                if (!perm_obj) {
                    throw err_code;
                }
                new (&perm_obj->perm) Simple_perm(std::move(*perm));
                PyList_SetItem(node_perms, i, (PyObject*)perm_obj);
            }
        }

        res = PyTuple_New(2);
        if (!res) {
            throw err_code;
        }
        PyTuple_SET_ITEM(res, 0, gl_order);
        PyTuple_SET_ITEM(res, 1, node_perms);

    } catch (int) {
        Py_XDECREF(res);
        Py_XDECREF(gl_order);
        Py_XDECREF(node_perms);
        return NULL;
    }

    return res;
}

//
// Interface functions
// -------------------
//

/** Docstring for Eldag canonicalization function.
 */

static const char* canon_eldag_docstring = R"__doc__(Canonicalizes an Eldag.

This is the core function of canonpy.  An Eldag can be given and it will be
canonicalized.

The Eldag should be given by four arguments,

Parameters
----------

edges

    An iterable of integers giving the edges in the Eldag.  It will be cast
    into a sequence internally.

ia

    An iterable of integers giving the starting index of edges in the edges
    array, as in CSR format for sparse matrices.  It determines the number of
    nodes in the Eldag.

symms

    An iterable giving the allowed symmetry operations on each node, should be
    given as a Group instance when symmetries are allowed, or a false value
    should be used.  Each node should be given explicitly.

colours

    An iterable initial colours of the nodes.  Positive integral values should
    be used and will be used for an initial partitioning of the nodes.  All
    nodes should be given one explicit initial colour.

Returns
-------

order

    A list of integers giving the order of the nodes in the canonical form.

perms

    A list for the permutations to be applied to each node.

)__doc__";

/** Eldag canonicalization driver function.
 */

static PyObject* canon_eldag_func(
    PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* edges_arg;
    PyObject* ia_arg;
    PyObject* symms_arg;
    PyObject* colours_arg;

    constexpr int err_code = 1;

    static char* kwlist[] = { "edges", "ia", "symms", "colours", NULL };

    auto arg_stat = PyArg_ParseTupleAndKeywords(args, keywds, "OOOO", kwlist,
        &edges_arg, &ia_arg, &symms_arg, &colours_arg);
    if (!arg_stat) {
        return NULL;
    }

    Point_vec edges{};
    Point_vec ia{};
    size_t n_nodes;
    Node_symms<Simple_perm> symms{};
    Point_vec colours{};

    try {
        edges = read_points(edges_arg);
        ia = read_points(ia_arg);
        n_nodes = ia.size() - 1;

        symms = read_symms(symms_arg);
        if (symms.size() != n_nodes) {
            std::string err_msg("Expecting ");
            err_msg.append(std::to_string(n_nodes));
            err_msg.append(" symmetries, ");
            err_msg.append(std::to_string(symms.size()));
            err_msg.append(" given.");
            PyErr_SetString(PyExc_ValueError, err_msg.c_str());
            throw err_code;
        }

        colours = read_points(colours_arg);
        if (colours.size() != n_nodes) {
            std::string err_msg("Expecting ");
            err_msg.append(std::to_string(n_nodes));
            err_msg.append(" colours, ");
            err_msg.append(std::to_string(colours.size()));
            err_msg.append(" given.");
            PyErr_SetString(PyExc_ValueError, err_msg.c_str());
            throw err_code;
        }

    } catch (int) {
        return NULL;
    }

    Eldag eldag{ std::move(edges), std::move(ia) };

    auto canon_res
        = canon_eldag(eldag, symms, [&](auto point) { return colours[point]; });

    // Currently, we just neglect the automorphism group.
    return build_canon_res(canon_res.first);
}

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

static PyMethodDef canonpy_methods[]
    = { { "canon_eldag", (PyCFunction)canon_eldag_func,
            METH_VARARGS | METH_KEYWORDS, canonpy_docstring },
        { NULL, NULL, 0, NULL } };

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
