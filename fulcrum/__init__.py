"""fulcrum: a prioritized, multi-tier, context-dependent preference calculus
for conflict resolution in formal argumentation, inspired by source_domain fulcrum.

Public API is intentionally small. See the project design notes for the build plan and,
crucially, the Phase-2 GATE that governs what is and is not yet a proven
contribution.
"""

from fulcrum.defeat import ArgumentationFramework
from fulcrum.meta_controller import (
    RECONCILE_FIRST,
    SUPERSEDE_FIRST,
    Resolution,
    resolve,
)
from fulcrum.preference import Preferred, weigh_compare
from fulcrum.semantics import acceptance, grounded_extension, is_admissible
from fulcrum.structures import (
    HEAD_ORDER,
    Argument,
    EpistemicStrength,
    PreferenceHead,
    ResolutionMode,
)

__all__ = [
    "Argument",
    "EpistemicStrength",
    "PreferenceHead",
    "ResolutionMode",
    "HEAD_ORDER",
    "weigh_compare",
    "Preferred",
    "resolve",
    "Resolution",
    "RECONCILE_FIRST",
    "SUPERSEDE_FIRST",
    "ArgumentationFramework",
    "grounded_extension",
    "acceptance",
    "is_admissible",
]

__version__ = "0.1.0"
