from skbandit.bandits import Bandit
from skbandit.environments.adversarial import AdversarialMultiArmedEnvironment
from skbandit.experiments import BanditFeedbackExperiment


class MultiArmedAdversarialExperiment(BanditFeedbackExperiment):
    """Performs an experiment with a multi-armed bandit algorithm facing an adversarial setting.

    An experiment takes two parameters: a `bandit`, which acts on an `environment`.
    """

    def __init__(self, environment: AdversarialMultiArmedEnvironment, bandit: Bandit):
        super().__init__(environment, bandit)
        assert environment.n_arms == bandit.n_arms
