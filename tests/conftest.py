import numpy as np
import pytest

from mmsim.data import MarketData


def make_md(book, trades, tick=1.0, symbol='TEST'):
    """MarketData from hand-written rows.

    book:   [(t_ms, bid, bid_qty, ask, ask_qty), ...]   prices in ticks
    trades: [(t_ms, price, qty, side), ...]             side +1 taker buy, -1 taker sell
    """
    b = np.array(book, dtype=float)
    tr = np.array(trades, dtype=float).reshape(-1, 4)
    return MarketData(symbol=symbol, tick=tick,
                      book_t=b[:, 0].astype(np.int64), bid=b[:, 1].astype(np.int64), bid_qty=b[:, 2],
                      ask=b[:, 3].astype(np.int64), ask_qty=b[:, 4],
                      trade_t=tr[:, 0].astype(np.int64), trade_px=tr[:, 1].astype(np.int64),
                      trade_qty=tr[:, 2], trade_side=tr[:, 3].astype(np.int8))


def random_market(n_steps=20_000, seed=0, tick=0.1, p_trade=0.3):
    """A random-walk market with a 1-tick spread and random trades, 10 ms apart."""
    rng = np.random.RandomState(seed)
    mid_bid = 10_000 + np.cumsum(rng.choice([-1, 0, 0, 0, 1], n_steps))
    t = np.arange(n_steps, dtype=np.int64) * 10
    book = np.c_[t, mid_bid, rng.uniform(0.5, 5, n_steps), mid_bid + 1, rng.uniform(0.5, 5, n_steps)]
    k = rng.rand(n_steps) < p_trade
    side = rng.choice([-1, 1], k.sum())
    depth = rng.geometric(0.6, k.sum()) - 1
    px = np.where(side > 0, mid_bid[k] + 1 + depth, mid_bid[k] - depth)
    trades = np.c_[t[k] + 5, px, rng.exponential(1.0, k.sum()), side]
    return make_md(book.tolist(), trades.tolist(), tick=tick)


@pytest.fixture
def market():
    return random_market()
