"""Hand-built order-book scenarios with a known answer for every fill rule."""
import numpy as np
import pytest

from conftest import make_md
from mmsim.simulator import simulate
from mmsim.strategies import FixedSpread, Touch

BOOK0 = (0, 100, 5.0, 101, 5.0)          # bid 100 x 5, ask 101 x 5
END = (10_000, 100, 5.0, 101, 5.0)
SLOW = dict(latency_ms=0, requote_ms=100_000)   # quote once at t=0, never again


def fills(res):
    f = res.fills
    return list(zip(f.t, f.side, f.price, f.qty, f.how.astype(str)))


def test_queue_ahead_is_consumed_before_we_fill():
    md = make_md([BOOK0, END], [(10, 100, 3.0, -1), (20, 100, 3.0, -1)])
    r = simulate(md, Touch(), size=1.0, **SLOW)
    # 5 lots ahead of us: the first 3 and 2 of the next 3 go to them, 1 to us
    assert fills(r) == [(20, 1, 100.0, 1.0, 'at_price')]


def test_cancellations_ahead_move_us_up():
    md = make_md([BOOK0, (5, 100, 1.0, 101, 5.0), END], [(10, 100, 2.0, -1)])
    r = simulate(md, Touch(), size=1.0, **SLOW)
    assert fills(r) == [(10, 1, 100.0, 1.0, 'at_price')]


def test_trade_through_fills_whole_order_whatever_its_size():
    md = make_md([BOOK0, END], [(10, 99, 0.01, -1)])
    r = simulate(md, Touch(), size=1.0, **SLOW)
    assert fills(r) == [(10, 1, 100.0, 1.0, 'trade_through')]


def test_market_crossing_our_order_fills_it_at_our_price():
    md = make_md([BOOK0, (10, 98, 1.0, 99, 2.0), END], [])
    r = simulate(md, FixedSpread(0.5), size=1.0, **SLOW)
    f = fills(r)
    assert f == [(10, 1, 100.0, 1.0, 'crossed')]
    # measured against the mid *before* the move that crossed us
    assert r.fills.mid.iloc[0] == 100.5


def test_order_improving_the_spread_has_no_queue():
    # spread of 3 ticks: a fixed half-spread of 0.5 around mid 101.5 posts 101 / 102
    md = make_md([(0, 100, 5.0, 103, 5.0), (10_000, 100, 5.0, 103, 5.0)], [(10, 101, 0.4, -1)])
    r = simulate(md, FixedSpread(0.5), size=1.0, **SLOW)
    assert fills(r) == [(10, 1, 101.0, 0.4, 'at_price')]     # partial fill, first in line


def test_deeper_order_learns_its_queue_when_its_level_becomes_best():
    # bid posted at 98, two ticks behind the best bid; later 98 becomes the best with 4 lots
    md = make_md([BOOK0, (10, 98, 4.0, 99, 3.0), END], [(20, 98, 4.0, -1), (30, 98, 1.0, -1)])
    r = simulate(md, FixedSpread(2.5), size=1.0, **SLOW)
    assert fills(r) == [(30, 1, 98.0, 1.0, 'at_price')]


def test_latency_delays_orders():
    md = make_md([BOOK0, END], [(5, 99, 1.0, -1), (15, 99, 1.0, -1)])
    r = simulate(md, Touch(), size=1.0, latency_ms=10, requote_ms=100_000)
    assert [f[0] for f in fills(r)] == [15]


def test_post_only_rejects_orders_that_would_cross_on_arrival():
    # decided at t=0 against bid 100 / ask 101; by arrival (t=10) the ask is at 100
    md = make_md([BOOK0, (5, 99, 1.0, 100, 1.0), END], [])
    r = simulate(md, Touch(), size=1.0, latency_ms=10, requote_ms=100_000)
    assert len(r.fills) == 0
    assert r.n_rejected == 1          # the bid at 100; the ask at 101 is fine


def test_requoting_at_the_same_price_keeps_queue_position():
    md = make_md([BOOK0, END], [(50, 100, 4.0, -1), (500, 100, 2.0, -1)])
    r = simulate(md, Touch(), size=1.0, latency_ms=0, requote_ms=10)
    assert fills(r) == [(500, 1, 100.0, 1.0, 'at_price')]
    # one bid and one ask, never replaced during 500 requotes; plus the bid re-posted after the fill
    assert r.n_orders == 3


@pytest.mark.parametrize('model, expected', [('touch', [(10, 1, 100.0, 1.0, 'at_price')]),
                                             ('queue', []),
                                             ('through', [])])
def test_fill_models_bound_the_queue_model(model, expected):
    md = make_md([BOOK0, END], [(10, 100, 3.0, -1)])
    assert fills(simulate(md, Touch(), size=1.0, fill_model=model, **SLOW)) == expected


def test_inventory_limit_stops_the_buying_side():
    trades = [(t, 99, 1.0, -1) for t in range(10, 2000, 20)]
    md = make_md([BOOK0, END], trades)
    r = simulate(md, Touch(), size=1.0, max_lots=3, latency_ms=0, requote_ms=10)
    assert r.s_inv.max() == 3.0
    assert len(r.fills) == 3


def test_inventory_skew_moves_both_quotes(market):
    flat = simulate(market, FixedSpread(3, skew=0.0), size=1.0, max_lots=5)
    skew = simulate(market, FixedSpread(3, skew=2.0), size=1.0, max_lots=5)
    assert np.abs(skew.s_inv).mean() < np.abs(flat.s_inv).mean()


def test_latency_longer_than_requote_interval_still_quotes():
    md = make_md([BOOK0, END], [(1000, 99, 1.0, -1)])
    r = simulate(md, Touch(), size=1.0, latency_ms=250, requote_ms=100)
    assert fills(r) == [(1000, 1, 100.0, 1.0, 'trade_through')]
    # nothing is live before the first decision arrives at t = 250
    r = simulate(make_md([BOOK0, END], [(200, 99, 1.0, -1)]), Touch(), size=1.0, latency_ms=250, requote_ms=100)
    assert len(r.fills) == 0
