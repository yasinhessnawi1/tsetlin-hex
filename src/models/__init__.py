"""Model modules for Graph Tsetlin Machine"""

from .hex_graph_tm import HexGraphTM
from .predictor import Predictor
from .tm_composite import (
    HexTMComposite,
    SpecialistConfig,
    create_depth_diverse_composite,
    create_specificity_diverse_composite,
    create_mixed_composite
)
from .weighted_gtm import WeightedGTM, create_weighted_gtm
from .drop_clause_gtm import DropClauseGTM, create_drop_clause_gtm
from .coalesced_gtm import CoalescedGTM, create_coalesced_gtm

__all__ = [
    'HexGraphTM',
    'Predictor',
    'HexTMComposite',
    'SpecialistConfig',
    'create_depth_diverse_composite',
    'create_specificity_diverse_composite',
    'create_mixed_composite',
    'WeightedGTM',
    'create_weighted_gtm',
    'DropClauseGTM',
    'create_drop_clause_gtm',
    'CoalescedGTM',
    'create_coalesced_gtm'
]
