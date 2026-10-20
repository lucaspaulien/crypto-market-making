"""Every number and figure in the README, from the converted Binance data.

Calibration (sigma, A, k, imbalance alpha) uses the first day only; every
strategy is then evaluated on the following days, out of sample.

    python scripts/run_experiments.py --data-dir data --out docs

Takes ~5 minutes for BTCUSDT + XRPUSDT over 7 test days.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmsim.backtest import run_many
from mmsim.calibration import calibrate, fit_alpha, imbalance_at, trade_depths
from mmsim.data import default_data_dir, load
from mmsim.strategies import AvellanedaStoikov, FixedSpread, Touch

# quote size ~ USD 650 per side on both symbols; inventory limit 10 lots
SETUP = {
    'BTCUSDT': dict(size=0.01, gamma=1e-3, gammas=[1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]),
    'XRPUSDT': dict(size=1000.0, gamma=1e-2, gammas=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]),
}
LATENCIES = [0, 1, 5, 20, 50, 100, 250, 500]
FEES = {'zero': 0.0, 'binance_vip0_maker_2bp': 2e-4, 'rebate_0.5bp': -0.5e-4}


def summarise(bts):
    rows = []
    for label, bt in bts.items():
        rows.append(dict(config=label, **bt.summary()))
    return pd.DataFrame(rows).set_index('config')


def signal_study(symbol, days, data_dir, alpha_cal, horizons=(100, 250, 500, 1000, 2000, 5000, 10000)):
    """Out-of-sample R^2 of the calibrated imbalance signal, per test day and horizon."""
    rows = []
    for d in days:
        md = load(symbol, [d], data_dir)
        t = np.arange(md.start, md.end - max(horizons), 100)
        imb, mid = imbalance_at(md, t)
        for h in horizons:
            _, mid_h = imbalance_at(md, t + h)
            y = mid_h - mid
            r2_fixed = 1 - np.var(y - alpha_cal * imb) / np.var(y)       # calibration-day slope, 1 s
            slope = np.polyfit(imb, y, 1)[0]
            rows.append(dict(day=d, horizon_ms=h, r2_calibrated_alpha=r2_fixed,
                             in_sample_slope=slope, corr=np.corrcoef(imb, y)[0, 1]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=str(default_data_dir()))
    ap.add_argument('--out', default='docs')
    ap.add_argument('--symbols', nargs='+', default=list(SETUP))
    ap.add_argument('--calibration-day', default='2024-03-23')
    ap.add_argument('--test-days', nargs='+',
                    default=[f'2024-03-{d}' for d in range(24, 31)])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    calib = {}

    for sym in args.symbols:
        cfg = SETUP[sym]
        size = cfg['size']
        md = load(sym, [args.calibration_day], args.data_dir)
        c = calibrate(md)
        alpha, r2 = fit_alpha(md)
        d = trade_depths(md)
        touch_rate = float(len(d) / ((md.end - md.start) / 1000) / 2)   # every burst reaches the touch
        tick, px = md.tick, float(np.median(md.mid)) * md.tick
        del md
        calib[sym] = dict(sigma_ticks_per_sqrt_s=c.sigma, A_per_s=c.A, k_per_tick=c.k,
                          alpha_ticks=alpha, alpha_r2_in_sample=r2, tick=tick, price=px,
                          tick_bp=1e4 * tick / px, quote_size=size,
                          intensity_depths=c.depths.tolist(), intensity_rates=c.rates.tolist(),
                          touch_rate=touch_rate)
        print(sym, {k: round(v, 5) for k, v in calib[sym].items() if not isinstance(v, list)}, flush=True)

        def AS(g, a=0.0):
            return AvellanedaStoikov(gamma=g, sigma=c.sigma, A=c.A, k=c.k, alpha=a)

        g0 = cfg['gamma']
        depth = AS(g0).c1
        base = dict(size=size)
        configs = {
            'touch': dict(strategy=Touch(), **base),
            'naive_symmetric': dict(strategy=FixedSpread(depth), **base),
            'linear_skew': dict(strategy=FixedSpread(depth, skew=AS(g0).c2), **base),
            'avellaneda_stoikov': dict(strategy=AS(g0), **base),
            'avellaneda_stoikov_alpha': dict(strategy=AS(g0, alpha), **base),
            'touch_alpha': dict(strategy=FixedSpread(0.5, skew=0.0, alpha=alpha), **base),
        }
        for g in cfg['gammas']:
            configs[f'as_gamma_{g:g}'] = dict(strategy=AS(g), **base)
        for h in ([0.5, 2, 5, 15, 40, 100, 300] if sym == 'BTCUSDT' else [0.5, 1.5, 2.5, 5]):
            configs[f'fixed_{h:g}'] = dict(strategy=FixedSpread(h), **base)
        for fm in ['touch', 'queue', 'through']:
            configs[f'touch_fill_{fm}'] = dict(strategy=Touch(), fill_model=fm, **base)
            configs[f'touch_alpha_fill_{fm}'] = dict(strategy=FixedSpread(0.5, alpha=alpha), fill_model=fm, **base)
            configs[f'as_alpha_fill_{fm}'] = dict(strategy=AS(g0, alpha), fill_model=fm, **base)
        for lat in LATENCIES:
            configs[f'touch_latency_{lat}'] = dict(strategy=Touch(), latency_ms=lat, **base)
            configs[f'as_alpha_latency_{lat}'] = dict(strategy=AS(g0, alpha), latency_ms=lat, **base)

        bts = run_many(sym, args.test_days, configs, args.data_dir)
        table = summarise(bts)
        for name, rate in FEES.items():
            table[f'net_bp_fee_{name}'] = table.gross_bp_of_volume - 1e4 * rate
        table.to_csv(out / f'results_{sym}.csv')
        pd.DataFrame({k: bt.markouts() for k, bt in bts.items()}).to_csv(out / f'markouts_{sym}.csv')
        eq = pd.DataFrame({k: bts[k].equity_curve for k in
                           ['touch', 'naive_symmetric', 'avellaneda_stoikov', 'avellaneda_stoikov_alpha']})
        eq.resample('1min').last().to_csv(out / f'equity_{sym}.csv')
        inv = pd.DataFrame({k: pd.concat(bts[k].inventory) for k in ['naive_symmetric', 'avellaneda_stoikov']})
        inv.resample('10s').last().to_csv(out / f'inventory_{sym}.csv')
        main_cols = ['fills', 'volume_usd', 'spread_capture', 'adverse_selection', 'inventory_carry', 'net',
                     'gross_bp_of_volume', 'mean_abs_inventory_lots', 'inventory_std_lots', 'max_drawdown']
        print(table.loc[list(configs)[:6], main_cols].round(3).to_string(), flush=True)

        sig = signal_study(sym, args.test_days, args.data_dir, alpha)
        sig.to_csv(out / f'signal_{sym}.csv', index=False)
        print(sig.groupby('horizon_ms')[['r2_calibrated_alpha', 'in_sample_slope']].mean().round(4), flush=True)

    with open(out / 'calibration.json', 'w') as fh:
        json.dump(calib, fh, indent=2)
    print('wrote results to', out)


if __name__ == '__main__':
    main()
