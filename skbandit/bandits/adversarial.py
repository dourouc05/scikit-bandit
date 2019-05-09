from typing import Union

import math

from skbandit.bandits.base import Bandit


class WeightedMajority(Bandit):
    """Optimal bandit in an adversarial setting.

    As all good algorithms for adversarial bandits, it randomises its decisions. This algorithm supposes access
    to full information (i.e. the reward for each arm at each round, not just the arms that were played).

    Its regret scales as the square root of the number of rounds, if the parameter eta is chosen accordingly to
    the the time horizon (using the `horizon` parameter). Otherwise, it grows linearly.

    See also:
        - https://www.sciencedirect.com/science/article/pii/S0890540184710091
        - https://948da3d8-a-62cb3a1a-s-sites.googlegroups.com/site/banditstutorial/home/slides/Bandit_small.pdf
    """

    def __init__(self, n_arms: int, eta: Union[float, None], horizon: Union[int, None] = None):
        super().__init__(n_arms)

        if eta is not None:
            self._eta = eta
        elif horizon is not None:
            self._eta = math.sqrt(8 * math.log(n_arms) / horizon)
        else:
            raise AssertionError("One of eta or horizon parameters must be set")

    def reward(self, arm: int, reward: float) -> None:
        pass

    def pull(self) -> int:
        pass



# TODO: EXP3
#   Example source: https://tor-lattimore.com/downloads/book/book.pdf, chapter 11
# TODO: Exponentially weighted forecaster
#   Example source: https://948da3d8-a-62cb3a1a-s-sites.googlegroups.com/site/banditstutorial/home/slides/Bandit_small.pdf, slide 35
# TODO: Implicitly normalised forecaster
#   Example source: https://948da3d8-a-62cb3a1a-s-sites.googlegroups.com/site/banditstutorial/home/slides/Bandit_small.pdf, slide 36
