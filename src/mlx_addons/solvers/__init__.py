# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""General-purpose iterative-solver primitives.

Currently exposes:

- :func:`pulay_diis` — Pulay's DIIS (Direct Inversion of the Iterative
  Subspace) extrapolator. Given a history of fixed-point iterates and their
  residuals, returns a least-squares-optimal linear combination. Standard
  accelerator for SCF, but applicable to any contractive fixed-point map.

- :func:`commutator_error` — the canonical SCF DIIS error vector
  ``F P - P F``. Lives here (not in linalg) because it is the most common
  Pulay-residual choice and is paired with :func:`pulay_diis` in practice.
"""

from ._diis import pulay_diis, commutator_error

__all__ = ["pulay_diis", "commutator_error"]
