"""
Tensor Contraction Engine output parser.

This module provides parsers of the output of the Tensor Contraction Engine of
So Hirata into Tensor objects in drudge.


"""

import itertools
import re

from sympy import nsimplify, sympify

from drudge import Term


#
# The driver function
# -------------------
#


def parse_tce_out(tce_out, symb_cb, base_cb):
    """Parse a TCE output into a list of terms.
    """

    lines = []
    for line in tce_out.splitlines():
        stripped = line.strip()
        if len(stripped) > 0:
            lines.append(stripped)
        continue

    return list(itertools.chain.from_iterable(
        _parse_tce_line(line, symb_cb, base_cb)
        for line in lines
    ))


#
# Internal functions
# ------------------
#


def _parse_tce_line(line, symb_cb, base_cb):
    """Parse a TCE output line into a list of terms.
    """

    # Get the initial part in the bracket and the actual term specification
    # part after it.
    match_res = re.match(
        r'^\s*\[(?P<factors>.*)\](?P<term>[^\[\]]+)$',
        line
    )
    if match_res is None:
        raise ValueError('Invalid TCE output line', line)

    factors_str = match_res.group('factors').strip()
    term_str = match_res.group('term').strip()

    # Get the actual term in its raw form.
    raw_term = _parse_term(term_str, symb_cb, base_cb)

    # Generates the actual list of terms based on the factors, possibly with
    # permutations.
    return _gen_terms(factors_str, raw_term, symb_cb)


#
# Some constants for the TCE output format
#


_SUM_BASE = 'Sum'


#
# Parsing the term specification
#


def _parse_term(term_str, symb_cb, base_cb):
    """Parse the term string after the square bracket into a Term.
    """

    # First break the string into indexed values.
    summed_vars, idxed_vals = _break_into_idxed(term_str)

    sums = tuple(symb_cb(i) for i in summed_vars)
    amp = sympify('1')

    for base, indices in idxed_vals:
        indices_symbs = tuple(symb_cb(i)[0] for i in indices)
        base_symb = base_cb(base, indices_symbs)
        amp *= base_symb[indices_symbs]
        continue

    return Term(sums=sums, amp=amp, vecs=())


def _break_into_idxed(term_str):
    """Break the term string into pairs of indexed base and indices.

    Both the base and the indices variables are going to be simple strings in
    the return value.
    """

    # First break it into fields separated by the multiplication asterisk.
    fields = (i for i in re.split(r'\s*\*\s*', term_str) if len(i) > 0)

    # Parse the fields one-by-one.
    idxed_vals = []
    for field in fields:

        # Break the field into the base part and the indices part.
        match_res = re.match(
            r'(?P<base>\w+)\s*\((?P<indices>.*)\)', field
        )
        if match_res is None:
            raise ValueError('Invalid indexed value', field)

        # Generate the final result.
        idxed_vals.append((
            match_res.group('base'),
            tuple(match_res.group('indices').split())
        ))

        continue

    # Summation always comes first in TCE output.
    if idxed_vals[0][0] == _SUM_BASE:
        return idxed_vals[0][1], idxed_vals[1:]
    else:
        return (), idxed_vals


#
# Final term generation based on the raw term
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


def _gen_terms(factors_str, raw_term, symb_cb):
    """Generate the actual terms based on the initial factor string.

    The raw term should be a term directly parsed from the term specification
    part of the TCE line. This function will use the factors string in the
    square bracket to turn it into a list of terms for the final value of the
    line.
    """

    # The regular expression for a factor.
    factor_regex = r'\s*'.join([
        r'(?P<sign>[+-])',
        r'(?P<factor_number>[0-9.]+)',
        r'(?:\*\s*P\((?P<perm_from>[^=>]*)=>(?P<perm_to>[^)]*)\))?',
    ]) + r'\s*'
    mismatch_regex = r'.'
    regex = '(?P<factor>{})|(?P<mismatch>{})'.format(
        factor_regex, mismatch_regex
    )

    # Iterate over the factors.
    terms = []
    for match_res in re.finditer(regex, factors_str):

        # Test if the result matches a factor.
        if match_res.group('factor') is None:
            raise ValueError('Invalid factor string', factors_str)

        # The value of the factor.
        factor_value = nsimplify(''.join(
            match_res.group('sign', 'factor_number')
        ), rational=True)

        # Get the substitution for the permutation of the indices.
        if match_res.group('perm_from') is not None:
            from_vars = match_res.group('perm_from').split()
            to_vars = match_res.group('perm_to').split()
            subs = [
                (symb_cb(from_var)[0], symb_cb(to_var)[0])
                for from_var, to_var in zip(from_vars, to_vars)
                ]
        else:
            subs = []

        # Add the result.
        terms.append(raw_term.subst(subs) * factor_value)

        # Continue to the next factor.
        continue

    return terms
