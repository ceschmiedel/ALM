"""L5 · Arbitration and synthesis — explicit conflict resolution with a trail."""

from alm.arbitration.arbiter import Arbiter, ArbitrationResult
from alm.arbitration.conflict import Conflict, ConflictDetector, ConflictKind
from alm.arbitration.strategies import (
    ArbitrationStrategy,
    AuthorityStrategy,
    ConfidenceStrategy,
    JudgeStrategy,
    Resolution,
    VerifierStrategy,
    build_strategies,
)
from alm.arbitration.synthesis import SynthesisResult, Synthesizer

__all__ = [
    "Arbiter",
    "ArbitrationResult",
    "ArbitrationStrategy",
    "AuthorityStrategy",
    "ConfidenceStrategy",
    "Conflict",
    "ConflictDetector",
    "ConflictKind",
    "JudgeStrategy",
    "Resolution",
    "SynthesisResult",
    "Synthesizer",
    "VerifierStrategy",
    "build_strategies",
]
