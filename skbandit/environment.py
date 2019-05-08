from abc import ABC, abstractmethod
from typing import List, TypeVar, Union

from scipy.stats import rv_continuous, rv_discrete, rv_histogram

random_variable = TypeVar('random_variable', rv_continuous, rv_discrete, rv_histogram)


class Environment(ABC):
    """An environment on which the bandit acts.

    This class is made for subclassing. Its subclasses are supposed to be used in experiments.

    An environment mainly implements a method `reward`, which returns the reward(s) obtained when the bandit plays
    a (set of) arms. Depending on the exact setting, this environment performs stochastically or adversarially,
    giving one reward (full-bandit feedback) or one reward per arm (semi-bandit feedback/full information),
    when these terms make sense (linear bandits, adversarial settings, mostly).

    A round corresponds to one call of the `reward` method. If calling this method becomes impossible (because
    resources are exhausted, for instance), you must implement both `may_stop_accepting_inputs` and
    `will_accept_input`.
    """

    @abstractmethod
    def reward(self, arm: Union[int, List[int]]) -> Union[float, List[float]]:
        pass

    @abstractmethod
    def true_reward(self, arm: Union[int, List[int]]) -> float:
        """Returns the (theoretical) mean for the given (combination of) arm(s)."""
        pass

    # No true_rewards, as not all environments have easily enumerable arms.

    @property
    def may_stop_accepting_inputs(self) -> bool:
        """Indicates whether the environment may stop accepting inputs or always accepts bandit actions.

        For instance, the environment has the notion of resource: when it is exhausted, the bandit can no more play.
        In this case, the property should be overwritten to return `True`.
        """
        return False

    # noinspection PyMethodMayBeStatic
    def will_accept_input(self) -> bool:
        """Indicates whether the environment will react correctly at the next round."""
        return True


class EnvironmentNoMoreAcceptingInputsException(Exception):
    """The environment no more accepts interactions."""
    pass


class MultiArmedEnvironment(Environment):
    """An environment on which a multi-armed bandit acts.

    The environments subclassing `MultiArmedEnvironment` are supposed to only allow one arm at a time, thus giving only
    a scalar reward.
    """

    @abstractmethod
    def reward(self, arm: int) -> float:
        pass

    @property
    @abstractmethod
    def n_arms(self) -> int:
        pass


class StochasticMultiArmedEnvironment(MultiArmedEnvironment):
    """A stochastic environment on which a multi-armed bandit acts.

    The only parameter is a list of probability distributions (SciPy random variables, subclasses of either
    `rv_continuous` or `rv_discrete`). There is one distribution per arm in the experiment. Their random states are
    supposed to be defined before being given to objects of this class (for reproducible experiments).
    """

    def __init__(self, distributions: List[random_variable]):
        self.distributions = distributions

    @property
    def n_arms(self) -> int:
        return len(self.distributions)

    def true_reward(self, arm: int) -> float:
        """Returns the (theoretical) mean for the given arm."""
        return self.distributions[arm].mean()

    @property
    def true_rewards(self) -> List[float]:
        """Returns the (theoretical) mean for each arm that may be played.

        These are also called true means.
        """
        return [d.mean() for d in self.distributions]

    def reward(self, arm: int) -> float:
        # Just draw one random number from the corresponding arm.
        return self.distributions[arm].rvs()
