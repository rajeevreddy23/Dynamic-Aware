"""Quality calibration models and backends for EAT-Sched."""

from __future__ import annotations

import abc
import math
from typing import Optional

import numpy as np


class QualityBackend(abc.ABC):
    """Abstract interface for task quality evaluation [PAPER Sec. C1]."""

    @abc.abstractmethod
    def compute_quality(self, steps: int, rng: Optional[np.random.Generator] = None) -> float:
        """Compute visual or output quality score in [0.0, 1.0]."""
        raise NotImplementedError


class SimulatedQualityBackend(QualityBackend):
    """Saturating monotone quality model calibrated against Stable Diffusion empirical data [ASM-02]."""

    def __init__(
        self,
        q_max: float = 0.85,
        tau_q: float = 18.0,
        sigma_q: float = 0.015,
    ) -> None:
        self.q_max = q_max
        self.tau_q = tau_q
        self.sigma_q = sigma_q

    def compute_quality(self, steps: int, rng: Optional[np.random.Generator] = None) -> float:
        """Compute saturating quality as a function of diffusion inference steps.

        q(s) = clip(q_max * (1 - exp(-s / tau_q)) + N(0, sigma_q^2), 0.0, 1.0)
        """
        base_quality = self.q_max * (1.0 - math.exp(-steps / self.tau_q))
        noise = 0.0
        if rng is not None and self.sigma_q > 0.0:
            noise = float(rng.normal(0.0, self.sigma_q))
        return float(np.clip(base_quality + noise, 0.0, 1.0))
