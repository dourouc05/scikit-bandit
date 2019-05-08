from abc import ABC

from .bandit import Bandit
from .environment import Environment, StochasticMultiArmedEnvironment, EnvironmentNoMoreAcceptingInputsException, \
    AdversarialMultiArmedEnvironment


class Experiment(ABC):
    """Performs an experiment with a bandit algorithm.

    An experiment takes two parameters: a `bandit`, which acts on an `environment`.

    The constructor is supposed to set the `best_arm` field to the best possible action on the environment (i.e.
    the one that generates the highest reward -- or one such combination of arms, depending on the setting).
    This is the main task for subclassing an environment. The `best_arm` is however not necessarily set, if it
    does not make sense for the specific environment.
    """

    def __init__(self, environment: Environment, bandit: Bandit):
        self.environment = environment
        self.bandit = bandit
        self._best_arm = None

    @property
    def best_arm(self):
        return self._best_arm

    def regret(self, reward) -> float:
        """Determines the regret when getting a given reward."""
        return self.environment.true_reward(self._best_arm) - reward

    def round(self) -> float:
        """Performs one round of experiment, yielding the regret for this round."""

        if self.environment.may_stop_accepting_inputs and not self.environment.will_accept_input():
            raise EnvironmentNoMoreAcceptingInputsException()

        arm = self.bandit.pull()
        reward = self.environment.reward(arm)
        self.bandit.reward(arm, reward)
        return self.regret(reward)

    def rounds(self, n: int) -> float:
        """Performs several rounds of experiment, yielding the total regret.

        If the environment stops accepting inputs within the `n` rounds, execution automatically stops.
        """
        if not self.environment.may_stop_accepting_inputs:
            return sum(self.round() for _ in range(n))
        else:
            total_reward = 0.0
            for _ in range(n):
                if not self.environment.will_accept_input():
                    break

                total_reward += self.round()
            return total_reward


class MultiArmedStochasticExperiment(Experiment):
    """Performs an experiment with a multi-armed bandit algorithm facing a stochastic setting.

    An experiment takes two parameters: a `bandit`, which acts on an `environment`.
    """

    def __init__(self, environment: StochasticMultiArmedEnvironment, bandit: Bandit):
        super().__init__(environment, bandit)

        assert environment.n_arms == bandit.n_arms

        means = environment.true_rewards
        self._best_arm = means.index(max(means))


class MultiArmedAdversarialExperiment(Experiment):
    """Performs an experiment with a multi-armed bandit algorithm facing an adversarial setting.

    An experiment takes two parameters: a `bandit`, which acts on an `environment`.
    """

    def __init__(self, environment: AdversarialMultiArmedEnvironment, bandit: Bandit):
        super().__init__(environment, bandit)
        assert environment.n_arms == bandit.n_arms
