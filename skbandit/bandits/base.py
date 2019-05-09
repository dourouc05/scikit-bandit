from abc import ABC, abstractmethod
from typing import Union, List, Dict


class Bandit(ABC):
    """A stochastic, multi-armed bandit player.

    A bandit player can be implemented using three methods:

    * a constructor: to create the player (required parameters, number of arms, etc.)
    * `pull()`: decide the current arm to pull (an integer number, starting at 0)
    * `reward(arm, reward)`: when pulling the `arm`, the player got a `reward`. This function is supposed to, but does
      not have to, be called after each call to `pull()`
    * `rewards(rewards)`: in a full-information or semi-bandit setting, the rewards associated with all arms (or
    just those that were played) at a given round
    """

    def __init__(self, n_arms: int):
        self._n_arms = n_arms

    @abstractmethod
    def pull(self) -> Union[int, List[int]]:
        pass

    def reward(self, arm: int, reward: float) -> None:
        raise NotImplemented()

    def rewards(self, reward: Union[List[float], Dict[int, float]]) -> None:
        raise NotImplemented()

    @property
    def n_arms(self):
        return self._n_arms


class RewardAccumulatorMixin:
    def __init__(self, n_arms: int):
        self._arm_counts = [0] * n_arms
        self._total_rewards = [0.0] * n_arms

    def reward(self, arm: int, reward: float) -> None:
        self._arm_counts[arm] += 1
        self._total_rewards[arm] += reward

    @property
    def total_rewards(self):
        return self._total_rewards

    @property
    def arm_counts(self):
        return self._arm_counts

# TODO: Explore what others implement:
#   https://github.com/jkomiyama/banditlib
#   https://www.di.ens.fr/~cappe/Code/PymaBandits/
#   https://github.com/flaviotruzzi/AdBandits
#   http://banditslilian.gforge.inria.fr/index.html -> https://smpybandits.readthedocs.io/en/latest/docs/Policies.html#submodules
