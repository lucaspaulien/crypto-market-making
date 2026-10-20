"""Running a strategy over several days and aggregating the results.

Each day is simulated independently, starting flat and marked to the mid at
the end (so a day's PnL includes the open position's mark, not a liquidation
cost). Loading one day at a time keeps memory bounded: a BTCUSDT day is
~13 million book updates.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import load
from .simulator import simulate

MARKOUT_HORIZONS_MS = (0, 100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 60_000)
USD_KEYS = ('volume_usd', 'spread_capture', 'adverse_selection', 'inventory_carry', 'fees', 'net')


@dataclass
class Backtest:
    symbol: str
    strategy: object
    days: list = field(default_factory=list)
    daily: list = field(default_factory=list)          # one summary dict per day
    equity: list = field(default_factory=list)         # one pd.Series per day
    inventory: list = field(default_factory=list)
    markout_num: dict = field(default_factory=dict)
    markout_den: float = 0.0
    fill_types: dict = field(default_factory=dict)

    def add(self, day, res, horizon_ms):
        s = res.summary(horizon_ms)
        s['day'] = day
        self.days.append(day)
        self.daily.append(s)
        self.equity.append(res.equity)
        self.inventory.append(res.inventory_lots)
        f = res.fills
        if len(f):
            for h in MARKOUT_HORIZONS_MS:
                m = res.mid_at(f.t.values + h) if h else f.mid.values
                self.markout_num[h] = self.markout_num.get(h, 0.0) + (f.edge + f.side * f.qty * (m - f.mid)).sum()
            self.markout_den += f.notional.sum()
            for k, v in f.how.value_counts().items():
                self.fill_types[k] = self.fill_types.get(k, 0) + int(v)

    @property
    def equity_curve(self):
        """Cumulative net PnL across days (each day's curve offset by the previous close)."""
        out, offset = [], 0.0
        for e in self.equity:
            out.append(e + offset)
            offset += e.iloc[-1]
        return pd.concat(out)

    def markouts(self):
        """Volume-weighted mean markout (bp of notional) at each horizon."""
        return pd.Series({h: 1e4 * v / self.markout_den for h, v in self.markout_num.items()}, name='markout_bp')

    def summary(self):
        d = pd.DataFrame(self.daily)
        out = {k: d[k].sum() for k in USD_KEYS}
        out['fills'] = int(d.fills.sum())
        out['net_bp_of_volume'] = 1e4 * out['net'] / out['volume_usd'] if out['volume_usd'] else np.nan
        out['gross_bp_of_volume'] = (1e4 * (out['net'] + (-out['fees'])) / out['volume_usd']
                                     if out['volume_usd'] else np.nan)
        out['days_positive'] = int((d.net > 0).sum())
        eq = self.equity_curve
        hourly = eq.resample('1h').last().diff().dropna()
        out['sharpe_hourly_annualised'] = (hourly.mean() / hourly.std() * np.sqrt(24 * 365)
                                           if hourly.std() > 0 else np.nan)
        out['max_drawdown'] = float((eq.cummax() - eq).max())
        inv = pd.concat(self.inventory)
        out['mean_abs_inventory_lots'] = float(inv.abs().mean())
        out['inventory_std_lots'] = float(inv.std())
        total = sum(self.fill_types.values())
        for k, v in self.fill_types.items():
            out[f'share_{k}'] = v / total
        return out


def run(symbol, days, strategy, data_dir, size, horizon_ms=10_000, **sim_kw):
    """Simulate ``strategy`` on each day and return the aggregated ``Backtest``."""
    bt = Backtest(symbol=symbol, strategy=strategy)
    for day in days:
        md = load(symbol, [day], data_dir)
        bt.add(day, simulate(md, strategy, size, **sim_kw), horizon_ms)
    return bt


def run_many(symbol, days, configs, data_dir, horizon_ms=10_000):
    """Several runs over the same days, loading each day once.

    ``configs`` maps a label to a dict of ``simulate`` keyword arguments
    (``strategy``, ``size`` and any of latency_ms, requote_ms, fill_model...).
    Returns {label: Backtest}.
    """
    out = {label: Backtest(symbol=symbol, strategy=cfg['strategy']) for label, cfg in configs.items()}
    for day in days:
        md = load(symbol, [day], data_dir)
        for label, cfg in configs.items():
            out[label].add(day, simulate(md, **cfg), horizon_ms)
    return out
