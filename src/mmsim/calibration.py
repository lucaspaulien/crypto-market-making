"""Estimating the Avellaneda-Stoikov inputs from the data itself.

* ``sigma``: standard deviation of 1-second mid changes, in ticks / sqrt(s).
* ``A, k``: the model assumes a quote resting ``delta`` ticks from the mid is
  hit by market orders at rate ``lambda(delta) = A exp(-k delta)``. For every
  taker trade we measure how far from the pre-trade mid it printed; a quote
  at depth ``delta`` would have been reached by every burst of market orders
  sweeping to ``delta`` or beyond, so ``lambda(delta)`` is the rate of such
  bursts (per side). ``log lambda`` is then fitted linearly in ``delta``.

``fit_alpha`` regresses the mid change over the next ``horizon_ms`` on the
top-of-book imbalance; the slope (ticks per unit of imbalance) is the shift
the strategies apply to their fair value.

The fitted intensity is a property of the *order flow*, measured without any
strategy in the market, which is exactly what the model needs; it is
estimated on a calibration day and the strategy is evaluated on later days.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    sigma: float        # ticks / sqrt(s)
    A: float            # 1 / s, per side
    k: float            # 1 / tick
    depths: np.ndarray  # grid used in the fit (ticks)
    rates: np.ndarray   # empirical lambda on that grid (1 / s, per side)


def mid_on_grid(md, step_ms=1000):
    t = np.arange(md.start, md.end, step_ms)
    i = np.clip(np.searchsorted(md.book_t, t, side='right') - 1, 0, len(md.book_t) - 1)
    return t, 0.5 * (md.bid[i] + md.ask[i])


def trade_depths(md):
    """Deepest level reached by each burst of market orders, in ticks from the pre-burst mid.

    The trade feed has one row per (taker order, price level), and an
    aggressive order -- or a cascade of them within the same millisecond --
    sweeps several levels before the next book update. One burst = the trades
    of one side within one millisecond; its depth is the furthest price it
    printed at, measured from the mid of the last book update before it.
    Counting rows instead would weight every sweep by the number of levels
    it crossed.
    """
    key = md.trade_t * 2 + (md.trade_side > 0)
    start = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    i = np.searchsorted(md.book_t, md.trade_t[start], side='left') - 1
    ok = i >= 0
    mid = 0.5 * (md.bid[i] + md.ask[i])
    dev = np.abs(md.trade_px - np.repeat(mid, np.diff(np.r_[start, len(key)])))
    depth = np.maximum.reduceat(dev, start)
    return depth[ok]


def fit_intensity(depths, duration_s, grid):
    """Least-squares fit of log(rate of trades at depth >= delta) = log A - k delta."""
    d = np.sort(depths)
    counts = len(d) - np.searchsorted(d, grid, side='left')
    rates = counts / duration_s / 2.0          # two sides
    ok = counts >= 20
    slope, intercept = np.polyfit(grid[ok], np.log(rates[ok]), 1)
    return float(np.exp(intercept)), float(-slope), rates


def calibrate(md, grid=None, step_ms=1000):
    _, mid = mid_on_grid(md, step_ms)
    sigma = float(np.std(np.diff(mid)) / np.sqrt(step_ms / 1000))
    depths = trade_depths(md)
    if grid is None:
        # from one tick beyond the touch out to the depth reached by ~1 burst in 1,000.
        # The touch itself is excluded: whether a quote *at* the touch fills is a
        # queue question, not a distance question, and its rate is an order of
        # magnitude above the exponential that fits every deeper level.
        far = max(np.quantile(depths, 0.999), 3.5)
        if far <= 12:
            # large-tick market: only a few price levels exist, evaluate exactly at them
            grid = np.arange(1.5, np.floor(far - 0.5) + 1.0)
        else:
            grid = np.linspace(1.5, far, 25)
    A, k, rates = fit_intensity(depths, (md.end - md.start) / 1000, grid)
    return Calibration(sigma=sigma, A=A, k=k, depths=grid, rates=rates)


def imbalance_at(md, t):
    i = np.clip(np.searchsorted(md.book_t, t, side='right') - 1, 0, len(md.book_t) - 1)
    bq, aq = md.bid_qty[i], md.ask_qty[i]
    return (bq - aq) / (bq + aq), 0.5 * (md.bid[i] + md.ask[i])


def fit_alpha(md, horizon_ms=1000, step_ms=100):
    """OLS slope and R^2 of (mid_{t+h} - mid_t) on the top-of-book imbalance at t."""
    t = np.arange(md.start, md.end - horizon_ms, step_ms)
    imb, mid = imbalance_at(md, t)
    _, mid_h = imbalance_at(md, t + horizon_ms)
    y = mid_h - mid
    slope, intercept = np.polyfit(imb, y, 1)
    r2 = 1 - np.var(y - (slope * imb + intercept)) / np.var(y)
    return float(slope), float(r2)
