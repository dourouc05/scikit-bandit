from typing import Union

import math

from skbandit.bandits.base import Bandit, RewardAccumulatorMixin


class ExploreThenCommitBandit(Bandit, RewardAccumulatorMixin):
    """Player that plays all arms in a round-robin fashion for a given number of epochs, then commits to the best arm.

    This bandit has two separate phases, one for exploration (it plays all arms, one after the other, for a given
    number of epochs), one for exploitation (afterwards, it only plays the best arm found so far).

    Its regret scales as the square root of the number of rounds, if the number of epochs is chosen accordingly to
    the minimum gap and the time horizon (using the `gap` and `horizon` parameter). Otherwise, it grows linearly.

    Also known as a greedy bandit.

    See also: https://tor-lattimore.com/downloads/book/book.pdf, Chapter 6.
    """

    def __init__(self, n_arms: int, n_epochs: Union[int, None] = None, gap: Union[float, None] = None,
                 horizon: Union[int, None] = None):
        Bandit.__init__(self, n_arms)
        RewardAccumulatorMixin.__init__(self, n_arms)

        self._current_round = 0
        self._best_arm = None

        if gap is not None and horizon is not None:
            self._n_epochs = max(1, math.ceil((4 / gap ** 2) * math.log(horizon * gap ** 2 / 4)))
            self._n_remaining_epochs = self._n_epochs
        elif n_epochs is not None:
            self._n_epochs = n_epochs
            self._n_remaining_epochs = n_epochs
        else:
            self._n_epochs = 1
            self._n_remaining_epochs = 1

    def _next_round(self):
        self._current_round += 1
        if self._current_round % self._n_arms == 0:
            self._n_remaining_epochs -= 1

    def pull(self, **kwargs) -> int:
        # Exploration phase: explore once each arm.
        if self._current_round < self._n_arms * self._n_epochs:
            arm = self._current_round % self._n_arms
        # Exploitation phase: always return the best arm (computed only once).
        elif self._best_arm is not None:
            arm = self._best_arm
        else:
            self._best_arm = self.total_rewards.index(max(self.total_rewards))
            arm = self._best_arm

        self._next_round()
        return arm

    def reward(self, arm: int, reward: float, **kwargs) -> None:
        # Stop storing rewards after the exploration phase.
        if self._current_round <= self._n_arms * self._n_epochs:
            RewardAccumulatorMixin.reward(self, arm, reward)


# TODO: Thompson sampling
# TODO: epsilon-greedy
#   Example source: https://towardsdatascience.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50
# TODO: softmax-greedy
#   Example source: https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html
# TODO: UCB
#   Example source: https://tor-lattimore.com/downloads/book/book.pdf, chapter 7
# TODO: MOSS
#   Example source: https://tor-lattimore.com/downloads/book/book.pdf, chapter 9
