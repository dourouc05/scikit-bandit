from typing import Union

import math
import numpy as np

from skbandit.bandits import Bandit


class LinUCB(Bandit):
    """LinUCB player for contextual linear bandits (disjoint model).

    Source: http://www.yisongyue.com/courses/cs159/lectures/LinUCB.pdf
    """

    def __init__(self, n_arms: int, n_features: int):
        Bandit.__init__(self, n_arms)

        self._current_round = 0
        self._n_features = n_features

        self._estimate_A = [np.identity(n_features) for _ in range(n_arms)]
        self._estimate_b = [np.zeros((n_features, 1)) for _ in range(n_arms)]

    def _check_context(self, context: Union[None, np.ndarray]):
        if context is None:
            raise AssertionError("Contextual LinUCB requires a context.")

        if context.shape != (self._n_features,):
            raise AssertionError("Contextual LinUCB requires a context of {} features.".format(self._n_features))

    def pull(self, context: Union[None, np.ndarray] = None) -> int:
        self._check_context(context)

        self._current_round += 1

        # Initialisation phase: explore once each arm.
        if self._current_round < self.n_arms:
            return self._current_round - 1

        # UCB phase.
        # Parameters are estimated as: A^-1 * b
        estimated_params = [np.linalg.solve(self._estimate_A[arm], self._estimate_b[arm]) for arm in range(self.n_arms)]
        # Confidence term is (x: context): sqrt(x A^-1 x) = sqrt(y x), letting y = x A^-1
        index = [
            estimated_params[arm] @ context + math.sqrt(np.linalg.solve(self._estimate_A[arm], context) @ context)
            for arm in range(self.n_arms)
        ]

        return max(range(self.n_arms), key=lambda arm: index[arm])

    def reward(self, arm: int, reward: float, context: Union[None, np.ndarray] = None) -> None:
        self._check_context(context)

        self._estimate_A[arm] += context @ context.T
        self._estimate_b[arm] += reward * context.T
