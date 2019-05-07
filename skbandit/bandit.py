import math
from abc import ABC, abstractmethod


class Bandit(ABC):
    """A stochastic, multi-armed bandit player.

    A bandit player can be implemented using three methods:

    * a constructor: to create the player (required parameters, number of arms, etc.)
    * `pull()`: decide the current arm to pull (an integer number, starting at 0)
    * `reward(arm, reward)`: when pulling the `arm`, the player got a `reward`. This function is supposed to, but does
      not have to, be called after each call to `pull()`
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms

    @abstractmethod
    def pull(self) -> int:
        pass

    @abstractmethod
    def reward(self, arm: int, reward: float) -> None:
        pass


class RewardAccumulatorMixin:
    def __init__(self, n_arms):
        self.arm_counts = [0] * n_arms
        self.total_rewards = [0.0] * n_arms

    def reward(self, arm, reward):
        self.arm_counts[arm] += 1
        self.total_rewards[arm] += reward


class ExploreThenCommitBandit(Bandit, RewardAccumulatorMixin):
    """Player that plays all arms in a round-robin fashion for a given number of epochs, then commits to the best arm.

    This bandit has two separate phases, one for exploration (it plays all arms, one after the other, for a given
    number of epochs), one for exploitation (afterwards, it only plays the best arm found so far).

    Its regret scales as the square root of the number of rounds, if the number of epochs is chosen accordingly to
    the minimum gap and the time horizon (using the `gap` and `horizon` parameter). Otherwise, it grows linearly.

    See also: https://tor-lattimore.com/downloads/book/book.pdf, Chapter 6.
    """

    def __init__(self, n_arms, n_epochs=None, gap=None, horizon=None):
        super().__init__(n_arms)
        self.current_round = 0
        self.best_arm = None

        if gap is not None and gap is not None:
            self.n_epochs = max(1, math.ceil((4 / gap**2) * math.log(horizon * gap**2 / 4)))
            self.n_remaining_epochs = self.n_epochs
        elif n_epochs is not None:
            self.n_epochs = n_epochs
            self.n_remaining_epochs = n_epochs
        else:
            self.n_epochs = 1
            self.n_remaining_epochs = 1

    def pull(self):
        # Exploration phase: explore once each arm.
        if self.current_round < self.n_arms * self.n_epochs:
            return self.current_round % self.n_arms
        # Exploitation phase: always return the best arm (computed only once).
        elif self.best_arm is not None:
            return self.best_arm
        else:
            self.best_arm = self.total_rewards.index(max(self.total_rewards))
            return self.best_arm

    def reward(self, arm, reward):
        # Stop storing rewards after the exploration phase.
        if self.current_round < self.n_arms:
            super().reward(arm, reward)

            self.current_round += 1
            if self.current_round % self.n_arms == 0:
                self.n_remaining_epochs -= 1
