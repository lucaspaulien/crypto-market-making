import numpy as np
import pandas as pd

from conftest import make_md
from mmsim.calibration import fit_alpha, fit_intensity, mid_on_grid, trade_depths
from mmsim.data import convert_day, load, to_ticks
from mmsim.strategies import AvellanedaStoikov


def test_glft_quotes():
    s = AvellanedaStoikov(gamma=0.05, sigma=2.0, A=1.5, k=0.3)
    b0, a0 = s.depths(0)
    assert np.isclose(b0, a0) and np.isclose(b0, s.c1 + s.c2 / 2)
    b, a = s.depths(2)                      # long 2 lots: buy less eagerly, sell more eagerly
    assert b > b0 and a < a0
    assert np.isclose(b - a, (2 * 2) * s.c2)
    # risk-neutral limit: half-spread -> 1/k
    assert np.isclose(AvellanedaStoikov(1e-9, 2.0, 1.5, 0.3).c1, 1 / 0.3, rtol=1e-6)
    # more risk aversion -> stronger inventory skew
    assert AvellanedaStoikov(0.2, 2.0, 1.5, 0.3).c2 > s.c2


def test_intensity_fit_recovers_an_exponential():
    rng = np.random.RandomState(0)
    A, k, T = 2.0, 0.4, 50_000.0
    n = rng.poisson(2 * A * T)                # two sides
    depths = rng.exponential(1 / k, n)
    grid = np.linspace(0.5, 10, 20)
    A_hat, k_hat, _ = fit_intensity(depths, T, grid)
    assert abs(k_hat - k) / k < 0.03 and abs(A_hat - A) / A < 0.05


def test_burst_depth_is_the_deepest_level_of_a_sweep():
    book = [(0, 100, 1.0, 101, 1.0), (100, 100, 1.0, 101, 1.0)]
    # one taker buy sweeping 101 -> 103 in the same ms, then a lone sell at the bid
    trades = [(10, 101, 1, 1), (10, 102, 1, 1), (10, 103, 1, 1), (20, 100, 1, -1)]
    d = trade_depths(make_md(book, trades))
    assert d.tolist() == [2.5, 0.5]


def test_sigma_and_alpha_on_synthetic_book():
    rng = np.random.RandomState(1)
    n = 200_000
    t = np.arange(n) * 100
    imb = rng.uniform(-1, 1, n)
    # the next 100 ms step drifts 0.3 tick per unit of current imbalance; imbalance is
    # redrawn every step, so the 1-second-ahead slope is 0.3 as well
    steps = np.r_[0, 0.3 * imb[:-1] + rng.normal(0, 1, n - 1)]
    bid = np.floor(10_000 + np.cumsum(steps)).astype(int)
    bq, aq = 1 + imb, 1 - imb + 1e-9
    md = make_md(np.c_[t, bid, bq, bid + 1, aq].tolist(), [(50, bid[0], 1, -1), (t[-1], bid[-1], 1, 1)])
    _, mid = mid_on_grid(md, 1000)
    sigma = np.std(np.diff(mid))              # 10 steps of ~1 tick per second
    assert 2.7 < sigma < 3.7
    alpha, r2 = fit_alpha(md, horizon_ms=1000)
    assert 0.25 < alpha < 0.35 and r2 > 0.002


def test_convert_day_roundtrip(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    pd.DataFrame({'update_id': [1, 2], 'best_bid_price': [100.1, 100.2], 'best_bid_qty': [1.0, 2.0],
                  'best_ask_price': [100.2, 100.3], 'best_ask_qty': [3.0, 4.0],
                  'transaction_time': [1000, 1005], 'event_time': [1001, 1006]}
                 ).to_csv(raw / 'XUSDT-bookTicker-2024-01-01.zip', index=False, compression='zip')
    pd.DataFrame({'agg_trade_id': [1, 2], 'price': [100.2, 100.1], 'quantity': [0.5, 0.25],
                  'first_trade_id': [1, 2], 'last_trade_id': [1, 2], 'transact_time': [1002, 1003],
                  'is_buyer_maker': [False, True]}
                 ).to_csv(raw / 'XUSDT-aggTrades-2024-01-01.zip', index=False, compression='zip')
    convert_day('XUSDT', '2024-01-01', tmp_path, tick=0.1)
    md = load('XUSDT', ['2024-01-01'], tmp_path)
    assert md.bid.tolist() == [1001, 1002] and md.ask.tolist() == [1002, 1003]
    assert md.trade_side.tolist() == [1, -1]        # buyer-maker = taker sell
    assert md.trade_px.tolist() == [1002, 1001]
    assert to_ticks([0.30000000000000004], 0.1).tolist() == [3]
