import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class BaseInitializer(ABC):
    """Abstract base class for cluster initializers"""
    
    @abstractmethod
    def initialize(self,
                  X: npt.NDArray[np.float64],
                  n_clusters: int,
                  cl_constraints: List[Tuple[int, int]],
                  random_state: Optional[int] = None) -> npt.NDArray[np.float64]:
        pass