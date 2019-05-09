from skbandit.bandits import Bandit
from skbandit.environments.stochastic import StochasticMultiArmedEnvironment
from skbandit.experiments import BanditFeedbackExperiment


class MultiArmedStochasticExperiment(BanditFeedbackExperiment):
    """Performs an experiment with a multi-armed bandit algorithm facing a stochastic setting.

    An experiment takes two parameters: a `bandit`, which acts on an `environment`.
    """

    def __init__(self, environment: StochasticMultiArmedEnvironment, bandit: Bandit):
        super().__init__(environment, bandit)

        assert environment.n_arms == bandit.n_arms

        means = environment.true_rewards
        self._best_arm = means.index(max(means))