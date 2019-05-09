import unittest
from typing import List, Union

from scipy.stats import rv_histogram

from skbandit.bandits.mab import ExploreThenCommitBandit
from skbandit.environments.stochastic import StochasticMultiArmedEnvironment
from skbandit.environments.adversarial import AdversarialMultiArmedEnvironment, Adversary
from skbandit.experiments.stochastic import MultiArmedStochasticExperiment
from skbandit.experiments.adversarial import MultiArmedAdversarialExperiment


class TestExploreThenCommitBandit(unittest.TestCase):
    def test_one_epoch(self):
        b = ExploreThenCommitBandit(n_arms=2)

        # Exploration.
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 1)

        # Exploitation.
        arm = b.pull()  # Compute best arm.
        self.assertIn(arm, [0, 1])
        self.assertEqual(b.pull(), arm)
        self.assertEqual(b.pull(), arm)
        self.assertEqual(b.pull(), arm)

    def test_fixed_epoch(self):
        b = ExploreThenCommitBandit(n_arms=2, n_epochs=3)

        # Exploration.
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 1)
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 1)
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 1)

        # Exploitation.
        arm = b.pull()  # Compute best arm.
        self.assertIn(arm, [0, 1])
        self.assertEqual(b.pull(), arm)
        self.assertEqual(b.pull(), arm)
        self.assertEqual(b.pull(), arm)

    def test_computed_epoch(self):
        b = ExploreThenCommitBandit(n_arms=2, gap=1.0, horizon=10)
        # ceil(4 / 1^2 * ln(10 * 1^2 / 4)) = 4

        # Exploration.
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 1)
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 1)
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 1)
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 1)

        # Exploitation.
        arm = b.pull()  # Compute best arm.
        self.assertIn(arm, [0, 1])
        self.assertEqual(b.pull(), arm)
        self.assertEqual(b.pull(), arm)
        self.assertEqual(b.pull(), arm)

    def test_one_epoch_rewarded(self):
        b = ExploreThenCommitBandit(n_arms=3)

        # Exploration.
        self.assertEqual(b.pull(), 0)
        b.reward(0, 10)
        self.assertEqual(b.pull(), 1)
        b.reward(1, 0)
        self.assertEqual(b.pull(), 2)
        b.reward(2, 5)

        # Exploitation.
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 0)

        # RewardAccumulatorMixin.
        self.assertEqual(b.total_rewards, [10.0, 0.0, 5.0])
        self.assertEqual(b.arm_counts, [1, 1, 1])

        # Even if giving more rewards for another arm, no change in exploitation phase.
        # This use is outside the expected use of the bandit.
        b.reward(0, 0)
        b.reward(1, 0)
        b.reward(2, 15)
        b.reward(0, 0)
        b.reward(1, 0)
        b.reward(2, 45)
        b.reward(0, 0)
        b.reward(1, 0)
        b.reward(2, 35)

        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 0)
        self.assertEqual(b.pull(), 0)

        # RewardAccumulatorMixin did not see any change.
        self.assertEqual(b.total_rewards, [10.0, 0.0, 5.0])
        self.assertEqual(b.arm_counts, [1, 1, 1])


class TestStochasticMultiArmedEnvironment(unittest.TestCase):
    def test_one(self):
        rv0 = rv_histogram(([1], [0, 0.000000001]))
        rv1 = rv_histogram(([1], [1, 1.000000001]))

        env = StochasticMultiArmedEnvironment([rv0, rv1])

        self.assertEqual(env.n_arms, 2)
        self.assertAlmostEqual(env.true_reward(0), 0.0)
        self.assertAlmostEqual(env.true_reward(1), 1.0)
        self.assertEqual(env.true_rewards[0], env.true_reward(0))
        self.assertEqual(env.true_rewards[1], env.true_reward(1))

        # For these specific distributions, the rewards are known in advance.
        self.assertAlmostEqual(env.reward(0), 0.0)
        self.assertAlmostEqual(env.reward(1), 1.0)


class DeterministicAdversary(Adversary):
    def __init__(self):
        super().__init__(2)

    def register_interaction(self, arm: Union[int, List[int]]) -> None:
        pass

    def decide_rewards(self) -> List[float]:
        return [0.0, 1.0]


class TestAdversarialMultiArmedEnvironment(unittest.TestCase):
    def test_one(self):
        env = AdversarialMultiArmedEnvironment(DeterministicAdversary())

        self.assertEqual(env.n_arms, 2)

        # For this specific adversary, the rewards are known in advance.
        self.assertAlmostEqual(env.reward(0), 0.0)
        self.assertAlmostEqual(env.reward(1), 1.0)
        self.assertAlmostEqual(env.regret(0.0), 1.0)
        self.assertAlmostEqual(env.regret(1.0), 0.0)


class TestMultiArmedStochasticExperiment(unittest.TestCase):
    def test_mismatch_env_bandit(self):
        rv0 = rv_histogram(([1], [0, 0.000000001]))
        rv1 = rv_histogram(([1], [1, 1.000000001]))
        env = StochasticMultiArmedEnvironment([rv0, rv1])

        b = ExploreThenCommitBandit(n_arms=20)

        with self.assertRaises(AssertionError):
            MultiArmedStochasticExperiment(env, b)

    def test_one(self):
        rv0 = rv_histogram(([1], [0, 0.000000001]))
        rv1 = rv_histogram(([1], [1, 1.000000001]))
        env = StochasticMultiArmedEnvironment([rv0, rv1])

        b = ExploreThenCommitBandit(n_arms=2)
        exp = MultiArmedStochasticExperiment(env, b)

        self.assertEqual(exp.best_arm, 1)

        self.assertAlmostEqual(exp.regret(0.0), 1.0)
        self.assertAlmostEqual(exp.regret(1.0), 0.0)

        # Perform a few rounds.
        self.assertAlmostEqual(exp.round(), 1.0)  # The bandit plays the first arm.
        self.assertAlmostEqual(exp.round(), 0.0)  # The bandit plays the second arm.
        self.assertAlmostEqual(exp.round(), 0.0)  # The bandit plays the best arm.
        self.assertAlmostEqual(exp.round(), 0.0)

        # Then a few more.
        self.assertAlmostEqual(exp.rounds(10), 0.0)

    def test_two(self):
        rv0 = rv_histogram(([1], [0, 0.000000001]))
        rv1 = rv_histogram(([1], [1, 1.000000001]))
        env = StochasticMultiArmedEnvironment([rv0, rv1])

        b = ExploreThenCommitBandit(n_arms=2)
        exp = MultiArmedStochasticExperiment(env, b)

        # Perform a few rounds only with rounds. The bandit will play the first arm, then the second, then the best.
        self.assertAlmostEqual(exp.rounds(10), 1.0)


class TestMultiArmedAdversarialExperiment(unittest.TestCase):
    def test_one(self):
        env = AdversarialMultiArmedEnvironment(DeterministicAdversary())
        b = ExploreThenCommitBandit(n_arms=2)
        exp = MultiArmedAdversarialExperiment(env, b)

        self.assertEqual(exp.best_arm, None)
        with self.assertRaises(AssertionError):
            exp.regret(1.0)

        self.assertAlmostEqual(exp.round(), 1.0)  # The bandit plays the first arm.
        self.assertAlmostEqual(exp.round(), 0.0)  # The bandit plays the second arm.
        self.assertAlmostEqual(exp.round(), 0.0)  # The bandit plays the best arm.

        self.assertAlmostEqual(exp.regret(0.0), 1.0)
        self.assertAlmostEqual(exp.regret(1.0), 0.0)

        # Then a few more.
        self.assertAlmostEqual(exp.rounds(10), 0.0)

    def test_two(self):
        env = AdversarialMultiArmedEnvironment(DeterministicAdversary())
        b = ExploreThenCommitBandit(n_arms=2)
        exp = MultiArmedAdversarialExperiment(env, b)

        # Perform a few rounds only with rounds. The bandit will play the first arm, then the second, then the best
        # (even though it does not consider the environment is adversary).
        self.assertAlmostEqual(exp.rounds(10), 1.0)
