from .bandit import Bandit
from .environment import StochasticMultiArmedEnvironment


class MultiArmedStochasticExperiment:
    """Performs an experiment with a multi-armed bandit algorithm facing a stochastic setting.

    The other parameter is the bandit for this experiment.
    """
    def __init__(self, environment: StochasticMultiArmedEnvironment, bandit: Bandit):
        self.environment = environment
        means = environment.true_means
        self.best_arm = means.index(max(means))
        self.bandit = bandit

    def regret(self, reward) -> float:
        return self.environment.true_mean(self.best_arm) - reward

    def round(self) -> float:
        """Performs one round of experiment, yielding the regret for this round."""
        arm = self.bandit.pull()
        reward = self.environment.reward(arm)
        self.bandit.reward(arm, reward)
        return self.regret(reward)

    def rounds(self, n: int) -> float:
        """Performs several rounds of experiment, yielding the total regret."""
        return sum(self.round() for _ in range(n))
