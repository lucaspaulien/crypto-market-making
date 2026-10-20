"""PnL accounting and attribution for a simulated market maker.

Marking the position to the mid, the PnL splits exactly (no residual) into

    PnL = sum_fills edge_i  +  sum_fills side_i * qty_i * (mid_T - mid_i)  -  fees

* **spread capture** ``edge_i = side_i * qty_i * (mid_i - price_i)``: what each
  fill earns against the mid at the moment it happens (always >= 0 for a
  passive order, up to half the spread);
* **inventory PnL**: what the position acquired by each fill makes or loses
  until the end, as the mid moves. Splitting it at a short horizon h,
    - **adverse selection** = ``side_i * qty_i * (mid_{t_i + h} - mid_i)``: the
      mid moving against the fill right after it, because the counterparty
      knew something -- the cost of being picked off;
    - **inventory carry** = the rest: drift of the position after h, the risk
      the inventory skew is meant to control;
* **fees**: maker fees (negative for a rebate).

``markouts`` gives the same split per fill at several horizons, the standard
way desks measure toxicity of the flow they trade against.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Result:
    md: object
    strategy: object
    size: float
    fills_raw: np.ndarray       # (n, 6): t, side, price ticks, qty, mid ticks, how
    s_t: np.ndarray
    s_inv: np.ndarray
    s_cash: np.ndarray
    s_mid: np.ndarray
    s_fees: np.ndarray
    n_orders: int
    n_rejected: int

    # ------------------------------------------------------------------ data
    @property
    def fills(self):
        F, tick = self.fills_raw, self.md.tick
        df = pd.DataFrame({'t': F[:, 0].astype(np.int64), 'side': F[:, 1].astype(int),
                           'price': F[:, 2] * tick, 'qty': F[:, 3], 'mid': F[:, 4] * tick,
                           'how': pd.Categorical.from_codes(F[:, 5].astype(int),
                                                            ['at_price', 'trade_through', 'crossed'])})
        df['notional'] = df.price * df.qty
        df['edge'] = df.side * df.qty * (df.mid - df.price)
        return df

    @property
    def equity(self):
        """Mark-to-mid PnL net of fees, sampled on the simulator's clock (USD)."""
        eq = self.s_cash + self.s_inv * self.s_mid * self.md.tick - self.s_fees
        return pd.Series(eq, index=pd.to_datetime(self.s_t, unit='ms'), name='pnl')

    @property
    def inventory_lots(self):
        return pd.Series(self.s_inv / self.size, index=pd.to_datetime(self.s_t, unit='ms'))

    def mid_at(self, t_ms):
        """Mid (USD) as of each time: last book update at or before it."""
        md = self.md
        i = np.searchsorted(md.book_t, np.asarray(t_ms), side='right') - 1
        i = np.clip(i, 0, len(md.book_t) - 1)
        return 0.5 * (md.bid[i] + md.ask[i]) * md.tick

    # ---------------------------------------------------------- attribution
    def attribution(self, horizon_ms=10_000):
        """Exact PnL split (USD): spread capture, adverse selection, inventory carry, fees."""
        f = self.fills
        mid_T = self.s_mid[-1] * self.md.tick
        gross = self.s_cash[-1] + self.s_inv[-1] * mid_T
        fees = self.s_fees[-1]
        if len(f) == 0:
            return dict(spread_capture=0.0, adverse_selection=0.0, inventory_carry=gross,
                        fees=-fees, net=gross - fees)
        mid_h = self.mid_at(f.t.values + horizon_ms)
        edge = f.edge.sum()
        adverse = (f.side * f.qty * (mid_h - f.mid)).sum()
        return dict(spread_capture=edge, adverse_selection=adverse,
                    inventory_carry=gross - edge - adverse, fees=-fees, net=gross - fees)

    def markouts(self, horizons_ms=(0, 100, 1_000, 5_000, 10_000, 30_000, 60_000, 300_000)):
        """Mean PnL per fill, in bp of notional, marked to the mid h ms after the fill."""
        f = self.fills
        out = {}
        for h in horizons_ms:
            m = self.mid_at(f.t.values + h) if h else f.mid.values
            out[h] = 1e4 * ((f.edge + f.side * f.qty * (m - f.mid)).sum() / f.notional.sum()) if len(f) else np.nan
        return pd.Series(out, name='markout_bp')

    # --------------------------------------------------------------- summary
    def summary(self, horizon_ms=10_000):
        f = self.fills
        eq = self.equity
        hourly = eq.resample('1h').last().diff().dropna()
        a = self.attribution(horizon_ms)
        volume = f.notional.sum()
        inv = self.s_inv / self.size
        return dict(
            strategy=getattr(self.strategy, 'name', type(self.strategy).__name__),
            fills=len(f),
            volume_usd=volume,
            **{k: v for k, v in a.items()},
            net_bp_of_volume=1e4 * a['net'] / volume if volume else np.nan,
            sharpe_hourly_annualised=(hourly.mean() / hourly.std() * np.sqrt(24 * 365)
                                      if len(hourly) > 2 and hourly.std() > 0 else np.nan),
            max_drawdown=float((eq.cummax() - eq).max()),
            mean_abs_inventory_lots=float(np.abs(inv).mean()),
            max_abs_inventory_lots=float(np.abs(inv).max()),
            orders=self.n_orders,
            rejected_post_only=self.n_rejected,
        )
