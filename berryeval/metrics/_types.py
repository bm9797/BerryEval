"""Type definitions and constants for the metrics engine."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

PADDING_ID: Final[int] = -1

# Type aliases for metric function signatures
QueryDocArray = NDArray[np.int32]
RelevanceArray = NDArray[np.int32]
MetricResult = NDArray[np.float32]
ScoreArray = NDArray[np.float64]
