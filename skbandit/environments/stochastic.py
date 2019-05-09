import collections
from abc import ABC
from typing import Union, List, TypeVar

from scipy.stats import rv_continuous, rv_discrete, rv_histogram

from skbandit.environments.base import Environment, BanditFeedbackEnvironment

random_variable = TypeVar('random_variable', rv_continuous, rv_discrete, rv_histogram)


class StochasticEnvironment(Environment, ABC):
    def true_reward(self, arm: Union[int, List[int]]) -> float:
        """Returns the (theoretical) mean for each (combination of) arm(s) in argument.

        These are also called true means.

        By default, this function calls `true_rewards` and sums the rewards corresponding to the arms that are
        being played.
        """

        if isinstance(arm, collections.Sequence):  # Several arms.
            return sum(self.true_rewards[a] for a in arm)
        else:  # Single arm.
            return self.true_rewards[arm]

    @property
    def true_rewards(self) -> List[float]:
        """Returns the (theoretical) mean for each (combination of) arms that may be played.

        Not all environments have (easily) enumerable arms: this function is not necessarily implemented by subclasses.
        """
        raise NotImplemented()


class StochasticMultiArmedEnvironment(BanditFeedbackEnvironment, StochasticEnvironment):
    """A stochastic environment on which a multi-armed bandit acts.

    The only parameter is a list of probability distributions (SciPy random variables, subclasses of either
    `rv_continuous` or `rv_discrete`). There is one distribution per arm in the experiment. Their random states are
    supposed to be defined before being given to objects of this class (for reproducible experiments).
    """

    def __init__(self, distributions: List[random_variable]):
        self._distributions = distributions

        # Determine the best arm. As this requires computing the best reward and the true means, store them.
        self._means = [d.mean() for d in self._distributions]
        self._best_reward = max(self._means)
        self._best_arm = self._means.index(self._best_reward)

    @property
    def n_arms(self) -> int:
        return len(self._distributions)

    @property
    def true_rewards(self) -> List[float]:
        return self._means

    def regret(self, reward: float) -> float:
        return self._best_reward - reward

    def reward(self, arm: int) -> float:
        # Just draw one random number from the corresponding arm.
        return self._distributions[arm].rvs()
