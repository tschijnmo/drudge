/* vim: set filetype=cpp: */

/** Header file for canonpy.
 *
 * Currently it merely contains the definition of the object structure of the
 * classes defined in canonpy.  They are put here in case a C API is intended
 * to be added for canonpy.
 */

#ifndef DRUDGE_CANONPY_H
#define DRUDGE_CANONPY_H

#include <Python.h>

#include <memory>

#include <libcanon/perm.h>
#include <libcanon/sims.h>

using libcanon::Simple_perm;
using libcanon::Sims_transv;

//
// Permutation type
// ----------------
//

/** Object type for canonpy Perm objects.
 */

// clang-format off
typedef struct {
    PyObject_HEAD
    Simple_perm perm;
} Perm_object;
// clang-format on

//
// Permutation group type
// ----------------------
//

// clang-format off
typedef struct {
    PyObject_HEAD
    std::unique_ptr<Sims_transv<Simple_perm>> transv;
} Group_object;
// clang-format on

#endif
