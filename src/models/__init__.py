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

__all__ = [
    'HexGraphTM',
    'Predictor',
    'HexTMComposite',
    'SpecialistConfig',
    'create_depth_diverse_composite',
    'create_specificity_diverse_composite',
    'create_mixed_composite'
]
