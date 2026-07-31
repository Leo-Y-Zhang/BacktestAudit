"""Single source of truth for the package version.

Kept in its own module so that submodules (e.g. :mod:`lyravalidate.report`) can
import the version without triggering a circular import via the package
``__init__``.
"""

from __future__ import annotations

__version__ = "0.3.0"
