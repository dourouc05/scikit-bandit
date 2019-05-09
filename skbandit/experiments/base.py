import collections
from abc import ABC, abstractmethod

from skbandit.bandits import Bandit
from skbandit.environments import Environment, EnvironmentNoMoreAcceptingInputsException, FullInformationEnvironment, \
    SemiBanditFeedbackEnvironment, BanditFeedbackEnvironment
from skbandit.environments.stochastic import StochasticMultiArmedEnvironment
from skbandit.environments.adversarial import AdversarialMultiArmedEnvironment


class Experiment(ABC):
    """Performs an experiment with a bandit algorithm.

    An experiment takes two parameters: a `bandit`, which acts on an `environment`.

    The constructor is supposed to set the `best_arm` field to the best possible action on the environment (i.e.
    the one that generates the highest reward -- or one such combination of arms, depending on the setting).
    This is the main task for subclassing an environment. The `best_arm` is however not necessarily set, if it
    does not make sense for the specific environment.
    """

    def __init__(self, environment: Environment, bandit: Bandit):
        self._environment = environment
        self._bandit = bandit
        self._best_arm = None

    @property
    def best_arm(self):
        return self._best_arm

    def regret(self, reward: float) -> float:
        """Determines the regret when getting a given reward."""
        return self._environment.regret(reward)

    @abstractmethod
    def round(self) -> float:
        """Performs one round of experiment, yielding the regret for this round."""
        pass

    def rounds(self, n: int) -> float:
        """Performs several rounds of experiment, yielding the total regret.

        If the environment stops accepting inputs within the `n` rounds, execution automatically stops.
        """
        if not self._environment.may_stop_accepting_inputs:
            return sum(self.round() for _ in range(n))
        else:
            total_reward = 0.0
            for _ in range(n):
                if not self._environment.will_accept_input():
                    break

                total_reward += self.round()
            return total_reward


class FullInformationExperiment(Experiment):
    """Performs an experiment with full information, i.e. one reward is known per arm and per round"""

    def __init__(self, environment: FullInformationEnvironment, bandit: Bandit):
        super().__init__(environment, bandit)

    def round(self) -> float:
        if self._environment.may_stop_accepting_inputs and not self._environment.will_accept_input():
            raise EnvironmentNoMoreAcceptingInputsException()

        arm = self._bandit.pull()
        rewards, reward = self._environment.rewards(arm)  # List and float.
        self._bandit.rewards(rewards)
        return self.regret(reward)


class SemiBanditFeedbackExperiment(Experiment):
    """Performs an experiment with semi-bandit information, i.e. rewards are known per played arm and per round"""

    def __init__(self, environment: SemiBanditFeedbackEnvironment, bandit: Bandit):
        super().__init__(environment, bandit)

    def round(self) -> float:
        if self._environment.may_stop_accepting_inputs and not self._environment.will_accept_input():
            raise EnvironmentNoMoreAcceptingInputsException()

        arm = self._bandit.pull()
        rewards, reward = self._environment.rewards(arm)  # Dictionary and float.
        self._bandit.rewards(rewards)
        return self.regret(reward)


class BanditFeedbackExperiment(Experiment):
    """Performs an experiment with full-bandit information, i.e. only one reward is known per round"""

    def __init__(self, environment: BanditFeedbackEnvironment, bandit: Bandit):
        super().__init__(environment, bandit)

    def round(self) -> float:
        if self._environment.may_stop_accepting_inputs and not self._environment.will_accept_input():
            raise EnvironmentNoMoreAcceptingInputsException()

        arm = self._bandit.pull()
        reward = self._environment.reward(arm)
        self._bandit.reward(arm, reward)
        return self.regret(reward)
