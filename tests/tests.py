import unittest

from scipy.stats import rv_histogram

from skbandit.bandit import ExploreThenCommitBandit
from skbandit.environment import StochasticMultiArmedEnvironment


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
