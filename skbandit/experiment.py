import math
from typing import List, TypeVar
from scipy.stats import rv_continuous, rv_discrete

from .bandit import Bandit


random_variable = TypeVar('random_variable', rv_continuous, rv_discrete)


class MultiArmedStochasticExperiment:
    """Performs an experiment with a multi-armed bandit algorithm facing a stochastic setting.

    The main parameter is a list of probability distributions (SciPy random variables, subclasses of either
    `rv_continuous` or `rv_discrete`). There is one distribution per arm in the experiment. Their random states are
    supposed to be defined before being given to objects of this class (for reproducible experiments).

    The other parameter is the bandit for this experiment.
    """
    def __init__(self, distributions: List[random_variable], bandit: Bandit):
        self.distributions = distributions
        self.means = [d.mean() for d in distributions]
        self.best_arm = self.means.index(max(self.means))
        self.bandit = bandit

    @property
    def n_arms(self) -> int:
        return len(self.distributions)

    def round(self) -> float:
        """Performs one round of experiment, yielding the regret for this round."""
        arm = self.bandit.pull()
        reward = self.distributions[arm].rvs()
        return self.means[self.best_arm] - reward
