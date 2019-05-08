from .bandit import Bandit
from abc import ABC
from .environment import Environment, StochasticMultiArmedEnvironment


class Experiment(ABC):
    """Performs an experiment with a bandit algorithm.

    An experiment takes two parameters: a `bandit`, which acts on an `environment`.

    The constructor is supposed to set the `best_arm` field to the best possible action on the environment (i.e.
    the one that generates the highest reward -- or one such combination of arms, depending on the setting).
    This is the main task for subclassing an environment.
    """
    def __init__(self, environment: Environment, bandit: Bandit):
        self.environment = environment
        self.bandit = bandit
        self.best_arm = None

    def regret(self, reward) -> float:
        """Determines the regret when getting a given reward."""
        return self.environment.true_reward(self.best_arm) - reward

    def round(self) -> float:
        """Performs one round of experiment, yielding the regret for this round."""
        arm = self.bandit.pull()
        reward = self.environment.reward(arm)
        self.bandit.reward(arm, reward)
        return self.regret(reward)

    def rounds(self, n: int) -> float:
        """Performs several rounds of experiment, yielding the total regret."""
        return sum(self.round() for _ in range(n))


class MultiArmedStochasticExperiment(Experiment):
    """Performs an experiment with a multi-armed bandit algorithm facing a stochastic setting.

    An experiment takes two parameters: a `bandit`, which acts on an `environment`.
    """
    def __init__(self, environment: StochasticMultiArmedEnvironment, bandit: Bandit):
        super().__init__(environment, bandit)
        means = environment.true_rewards
        self.best_arm = means.index(max(means))
