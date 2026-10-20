import numpy as np
import pytest

from mmsim.simulator import simulate
from mmsim.strategies import AvellanedaStoikov, FixedSpread, Touch


@pytest.mark.parametrize('strategy', [Touch(), FixedSpread(2, skew=1), AvellanedaStoikov(0.1, 1.0, 1.0, 0.5)])
def test_attribution_adds_up_exactly(market, strategy):
    r = simulate(market, strategy, size=1.0, max_lots=5, fee_rate=1e-4)
    a = r.attribution(horizon_ms=500)
    parts = a['spread_capture'] + a['adverse_selection'] + a['inventory_carry'] + a['fees']
    assert np.isclose(parts, a['net'])
    assert np.isclose(a['net'], r.equity.iloc[-1])


def test_cash_and_inventory_match_the_fill_log(market):
    r = simulate(market, Touch(), size=1.0, max_lots=5)
    f = r.fills
    assert np.isclose((f.side * f.qty).sum(), r.s_inv[-1])
    assert np.isclose(-(f.side * f.qty * f.price).sum(), r.s_cash[-1])


def test_passive_fills_never_pay_the_spread_at_the_quote_mid(market):
    f = simulate(market, Touch(), size=1.0, max_lots=5).fills
    trade_fills = f[f.how != 'crossed']
    assert (trade_fills.edge >= -1e-12).all()


def test_fees_cost_exactly_rate_times_notional(market):
    free = simulate(market, Touch(), size=1.0, max_lots=5)
    paid = simulate(market, Touch(), size=1.0, max_lots=5, fee_rate=2e-4)
    assert np.isclose(free.equity.iloc[-1] - paid.equity.iloc[-1], 2e-4 * free.fills.notional.sum())


def test_markout_at_zero_is_spread_capture(market):
    r = simulate(market, Touch(), size=1.0, max_lots=5)
    f = r.fills
    assert np.isclose(r.markouts((0,)).iloc[0], 1e4 * f.edge.sum() / f.notional.sum())
