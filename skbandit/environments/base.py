from abc import ABC, abstractmethod
from typing import List, Union, Dict


class Environment(ABC):
    """An environment on which the bandit acts.

    This class is made for subclassing. Its subclasses are supposed to be used in experiments.

    An environment mainly implements a method `reward`, which returns the reward(s) obtained when the bandit plays
    a (set of) arms. Depending on the exact setting, this environment performs stochastically or adversarially,
    giving one reward (full-bandit feedback) or one reward per arm (semi-bandit feedback/full information),
    when these terms make sense (linear bandits, adversarial settings, mostly).

    A round corresponds to one call of the `reward` or the `rewards` methods. If calling this method becomes impossible
    (because resources are exhausted, for instance), you must implement both `may_stop_accepting_inputs` and
    `will_accept_input`. `reward` must be implemented when only full-bandit information is available, i.e. one reward
    per round. `rewards` must be implemented when the environment provides full information, i.e. one reward per arm
    and per round (the function returns a list, indexed by the arms), or semi-bandit feedback, i.e. one reward per
    played arm and per round (the function returns a dictionary, indexed by the played arms); in all cases,
    this function also returns the (scalar!) reward associated with the arm combination.

    Neither `reward` nor `rewards` should consider that their input is well-formed, in the set that, if constraints on
    the set of arms to play should be enforced (like in combinatorial bandits), these constraints are not necessarily
    satisfied by the input value. Indeed, the bandit algorithm is not always ensured to be able to take these
    constraints into account when choosing an arm combination to play.
    """

    def reward(self, arm: Union[int, List[int]]) -> float:
        """Record the interaction and return a scalar reward."""
        raise NotImplementedError

    def rewards(self, arm: Union[int, List[int]]) -> (Union[List[float], Dict[int, float]], float):
        """Record the interaction and return a vector reward."""
        raise NotImplementedError

    @abstractmethod
    def regret(self, reward: float) -> float:
        """Compute the exact regret when getting a given reward at the last round.

        The regret is defined as the difference in reward between the best action and the one that was taken.
        For stationary stochastic bandits, the best action does not change in time, and thus calling this function
        with the same reward should return the same value between round. This is not necessarily the case for
        nonstationary bandits or adversarial bandits.
        """
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


class EnvironmentNoMoreAcceptingInputsException(Exception):
    """The environment no more accepts interactions."""
    pass


class FullInformationEnvironment(Environment, ABC):
    @abstractmethod
    def rewards(self, arm: Union[int, List[int]]) -> (List[float], float):
        """Record the interaction and return a reward for each arm (be it played or not).

        As a consequence, the `arm` argument is ignored: the reward is automatically returned for all arms.
        """
        pass


class SemiBanditFeedbackEnvironment(Environment, ABC):
    @abstractmethod
    def rewards(self, arm: Union[int, List[int]]) -> (Dict[int, float], float):
        """Record the interaction and return a reward for each of the played arms."""
        pass


class BanditFeedbackEnvironment(Environment, ABC):
    @abstractmethod
    def reward(self, arm: Union[int, List[int]]) -> float:
        """Record the interaction and return a scalar reward."""
        pass
