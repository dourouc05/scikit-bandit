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
    def regret(self, reward: float) -> float:
        pass

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


class StochasticEnvironment(ABC):
    @abstractmethod
    def true_reward(self, arm: Union[int, List[int]]) -> float:
        """Returns the (theoretical) mean for each arm that may be played.

        These are also called true means.
        """
        pass

    def true_rewards(self) -> List[float]:
        # Not an abstract method, as not all environments have easily enumerable arms.
        raise NotImplemented()


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


class StochasticMultiArmedEnvironment(MultiArmedEnvironment, StochasticEnvironment):
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

    def true_reward(self, arm: int) -> float:
        return self._means[arm]

    @property
    def true_rewards(self) -> List[float]:
        return self._means

    def regret(self, reward: float) -> float:
        return self._best_reward - reward

    def reward(self, arm: int) -> float:
        # Just draw one random number from the corresponding arm.
        return self._distributions[arm].rvs()


class Adversary(ABC):
    """The adversary against which a bandit plays.

    The adversary decides a reward for each arm, without knowing which arm the bandit is playing: the method `pull`
    returns a vector of rewards. Then, the adversary learns the arm that was played (with the reward that it got
    from it), in order to prepare for future rounds, through the method `reward`.
    """

    def __init__(self, n_arms: int):
        self._n_arms = n_arms

    @property
    def n_arms(self):
        return self._n_arms

    @abstractmethod
    def reward(self, arm: int, reward: float) -> None:
        pass

    @abstractmethod
    def pull(self) -> List[float]:
        pass


class AdversarialMultiArmedEnvironment(MultiArmedEnvironment):
    """An adversarial environment on which a multi-armed bandit acts.

    The main parameter is an adversary.
    """

    def __init__(self, n_arms: int, adversary: Adversary):
        assert n_arms == adversary.n_arms

        self._n_arms = n_arms
        self._adversary = adversary
        self._last_rewards = None

    @property
    def n_arms(self) -> int:
        return self._n_arms

    def reward(self, arm: int) -> float:
        self._last_rewards = self._adversary.pull()
        reward = self._last_rewards[arm]
        self._adversary.reward(arm, reward)
        return reward

    def regret(self, reward: float) -> float:
        if self._last_rewards is None:
            raise AssertionError("reward() not called before regret(). The following order must be respected: "
                                 "first, the bandit chooses an arm; then, the adversary decides the rewards"
                                 "(without knowing the arm the bandit plays); finally, the environment returns the "
                                 "reward. All of this is done when calling reward(arm). ")
        return max(self._last_rewards) - reward


class FullInformationAdversarialMultiArmedEnvironment(AdversarialMultiArmedEnvironment):
    def rewards(self) -> List[float]:
        return self._last_rewards
