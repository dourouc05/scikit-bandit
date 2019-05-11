import math

from skbandit.bandits import Bandit, RewardAccumulatorMixin


class LinUCB(Bandit, RewardAccumulatorMixin):
    """LinUCB player for linear bandits (disjoint model).

    Source: http://www.yisongyue.com/courses/cs159/lectures/LinUCB.pdf
    """

    def __init__(self, n_arms: int):
        Bandit.__init__(self, n_arms)
        RewardAccumulatorMixin.__init__(self, n_arms)

        self._current_round = 0

    def pull(self, **kwargs) -> int:
        self._current_round += 1

        # Initialisation phase: explore once each arm.
        if self._current_round < self.n_arms:
            return self._current_round - 1

        # UCB phase.
        estimated_rewards = [self.total_rewards[arm] / self.arm_counts[arm] for arm in range(self.n_arms)]
        index = [
            estimated_rewards[arm] + math.sqrt(2 * math.log(self._current_round) / self.arm_counts[arm])
            for arm in range(self.n_arms)
        ]

        return max(range(self.n_arms), key=lambda arm: estimated_rewards[arm])
