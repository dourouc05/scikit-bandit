import collections
from abc import ABC, abstractmethod
from typing import List, Union

from skbandit.environments.base import Environment, BanditFeedbackEnvironment, FullInformationEnvironment


class Adversary(ABC):
    """The adversary against which a bandit plays.

    The adversary decides a reward for each arm, without knowing which arm the bandit is playing: the method
    `decide_rewards` returns a vector of rewards. Then, the adversary learns the arm that was played, in order to
    prepare for future rounds, through the method `register_interaction`.
    """

    def __init__(self, n_arms: int):
        self._n_arms = n_arms

    @property
    def n_arms(self):
        return self._n_arms

    @abstractmethod
    def register_interaction(self, arm: Union[int, List[int]]) -> None:
        """Registers an interaction with the environment."""
        pass

    @abstractmethod
    def decide_rewards(self) -> List[float]:
        """Returns the decided rewards for all the arms."""
        pass


class AdversarialEnvironment(Environment, ABC):
    def __init__(self, adversary: Adversary):
        self._last_rewards = None
        self._adversary = adversary

    @property
    def n_arms(self) -> int:
        return self._adversary.n_arms

    def regret(self, reward: float) -> float:
        if self._last_rewards is None:
            raise AssertionError("reward() not called before regret(). The following order must be respected: "
                                 "first, the bandit chooses an arm; then, the adversary decides the rewards"
                                 "(without knowing the arm the bandit plays); finally, the environment returns the "
                                 "reward. All of this is done when calling reward(arm). ")
        return max(self._last_rewards) - reward


class AdversarialMultiArmedEnvironment(BanditFeedbackEnvironment, AdversarialEnvironment):
    """An adversarial environment on which a multi-armed bandit acts."""

    def reward(self, arm: int) -> float:
        self._last_rewards = self._adversary.decide_rewards()

        if isinstance(arm, collections.abc.Sequence):  # Several arms.
            reward = sum(self._last_rewards[a] for a in arm)
        else:  # Single arm.
            reward = self._last_rewards[arm]

        self._adversary.register_interaction(arm)
        return reward


class FullInformationAdversarialMultiArmedEnvironment(FullInformationEnvironment, AdversarialEnvironment):
    def rewards(self, arm: Union[int, List[int]]) -> List[float]:
        self._last_rewards = self._adversary.decide_rewards()
        self._adversary.register_interaction(arm)
        return self._last_rewards
