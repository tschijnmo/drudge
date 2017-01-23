/* vim: set filetype=cpp: */

/** Header file for canonpy.
 *
 * Currently it merely contains the definition of the object structure of the
 * classes defined in canonpy.  They are put here in case a C API is intended
 * to be added for canonpy.
 */

#include <Python.h>

#ifndef DRUDGE_CANONPY_H
#define DRUDGE_CANONPY_H

#include <libcanon/perm.h>

using libcanon::Simple_perm;

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

#endif
