"""Quoting rules: given the book and the inventory, where to post the bid and the ask.

Every strategy is a small frozen dataclass whose ``encode()`` returns
``(kind, params)`` for the compiled simulator. Distances are in ticks from the
mid, inventory ``q`` is measured in lots (multiples of the quote size).

* ``Touch``      -- join the best bid and best ask, no matter what.
* ``FixedSpread`` -- mid +- a constant half-spread, with an optional linear
  inventory skew; with ``skew=0`` this is the naive symmetric market maker.
All of them except ``Touch`` can centre their quotes on an *alpha-adjusted*
fair value ``mid + alpha * imbalance`` instead of the mid, where
``imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)`` at the top of the
book and ``alpha`` (ticks) is fitted on a calibration day
(``calibration.fit_alpha``). This is how a market maker leans away from the
side that is about to be run over.

* ``AvellanedaStoikov`` -- the stationary (infinite-horizon) solution of the
  Avellaneda-Stoikov problem in the closed form of Gueant, Lehalle and
  Fernandez-Tapia (2013). With fills arriving at depth delta at rate
  ``A exp(-k delta)``, mid volatility ``sigma`` and risk aversion ``gamma``:

      delta_bid(q) = c1 + (2q + 1)/2 * c2        delta_ask(q) = c1 - (2q - 1)/2 * c2
      c1 = (1/gamma) ln(1 + gamma/k)
      c2 = sqrt( sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma) )

  ``c1`` is the half-spread a risk-neutral maker would quote against this
  order flow; ``c2`` is how far one lot of inventory shifts both quotes. Being
  long pushes the bid away and pulls the ask in, so the strategy sells down
  its inventory -- the whole point of the model.
"""
from dataclasses import dataclass

import numpy as np

TOUCH, FIXED, GLFT = 0, 1, 2


@dataclass(frozen=True)
class Touch:
    name: str = 'touch'

    def encode(self):
        return TOUCH, np.zeros(4)


@dataclass(frozen=True)
class FixedSpread:
    half_spread: float          # ticks from mid
    skew: float = 0.0           # ticks of quote shift per lot of inventory
    alpha: float = 0.0          # ticks of fair-value shift per unit of book imbalance
    name: str = 'fixed'

    def encode(self):
        return FIXED, np.array([self.half_spread, self.skew, self.alpha, 0.0])


@dataclass(frozen=True)
class AvellanedaStoikov:
    gamma: float                # risk aversion, 1 / (tick x lot)
    sigma: float                # mid volatility, ticks / sqrt(s)
    A: float                    # fill intensity at the mid, 1 / s
    k: float                    # decay of fill intensity with depth, 1 / tick
    alpha: float = 0.0          # ticks of fair-value shift per unit of book imbalance
    name: str = 'avellaneda_stoikov'

    @property
    def c1(self):
        return np.log1p(self.gamma / self.k) / self.gamma

    @property
    def c2(self):
        g, k = self.gamma, self.k
        return np.sqrt(self.sigma ** 2 * g / (2 * k * self.A) * (1 + g / k) ** (1 + k / g))

    def depths(self, q):
        """(bid depth, ask depth) in ticks from mid for inventory q (lots)."""
        q = np.asarray(q, dtype=float)
        return self.c1 + (2 * q + 1) / 2 * self.c2, self.c1 - (2 * q - 1) / 2 * self.c2

    def encode(self):
        return GLFT, np.array([self.c1, self.c2, self.alpha, 0.0])
