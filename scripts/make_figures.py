"""README figures from the CSVs written by run_experiments.py.

    python scripts/make_figures.py --results docs
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# fixed categorical order (validated palette), muted ink for text and axes
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SYMBOLS = ['BTCUSDT', 'XRPUSDT']
STRATS = [('touch', 'Join the touch'), ('naive_symmetric', 'Naive symmetric'),
          ('linear_skew', 'Linear skew'), ('avellaneda_stoikov', 'Avellaneda-Stoikov'),
          ('avellaneda_stoikov_alpha', 'A-S + imbalance alpha')]

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
    'axes.titlesize': 11, 'legend.frameon': False, 'lines.linewidth': 2,
})


def load(results):
    r = {s: pd.read_csv(results / f'results_{s}.csv', index_col=0) for s in SYMBOLS}
    m = {s: pd.read_csv(results / f'markouts_{s}.csv', index_col=0) for s in SYMBOLS}
    return r, m


def bp(t, col):
    return 1e4 * t[col] / t.volume_usd


def attribution(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    parts = [('spread_capture', 'Spread capture'), ('adverse_selection', 'Adverse selection (10 s)'),
             ('inventory_carry', 'Inventory carry')]
    for ax, s in zip(axes, SYMBOLS):
        t = r[s].loc[[k for k, _ in STRATS]]
        x = np.arange(len(t))
        w = 0.26
        for j, (col, label) in enumerate(parts):
            ax.bar(x + (j - 1) * w, bp(t, col), w - 0.03, color=C[j], label=label)
        ax.scatter(x, t.gross_bp_of_volume, color=INK, marker='D', s=36, zorder=5, label='Total (gross of fees)')
        for xi, v in zip(x, t.gross_bp_of_volume):
            ax.annotate(f'{v:+.2f}', (xi, v), textcoords='offset points', xytext=(14, -3), fontsize=9, color=INK)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x, [lbl for _, lbl in STRATS], rotation=18, ha='right')
        ax.set_title(f'{s}: PnL per unit of volume traded')
        ax.set_ylabel('bp of traded notional')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / 'attribution.png', dpi=140)


def markouts(m, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    lines = [('touch_fill_touch', 'Touch, front of queue (upper bound)', C[2], '-'),
             ('touch_fill_queue', 'Touch, queue model', C[0], '-'),
             ('touch_fill_through', 'Touch, trade-throughs only (lower bound)', C[7], '-'),
             ('avellaneda_stoikov_alpha', 'A-S + imbalance alpha', C[1], '--')]
    for ax, s in zip(axes, SYMBOLS):
        mk = m[s]
        h = mk.index.values.astype(float)
        hx = np.where(h == 0, 30, h)                  # place h = 0 at the left edge of the log axis
        for col, label, c, ls in lines:
            ax.plot(hx, mk[col], color=c, ls=ls, label=label, marker='o', ms=4)
        ax.set_xscale('log')
        show = {0, 100, 500, 1000, 5000, 10000, 60000}
        ax.set_xticks(hx, ['0' if v == 0 else ('' if v not in show else
                                               (f'{v / 1000:g}s' if v >= 1000 else f'{v:g}ms')) for v in h])
        ax.minorticks_off()
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(title=f'{s}: markout per fill', xlabel='time after the fill', ylabel='bp of notional')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / 'markouts.png', dpi=140)


def frontier(r, results, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, s in zip(axes[:2], SYMBOLS):
        t = r[s]
        g = t[t.index.str.startswith('as_gamma_')]
        gam = g.index.str.replace('as_gamma_', '').astype(float)
        ax.plot(g.inventory_std_lots, g.net, color=C[0], marker='o', ms=6, label='Avellaneda-Stoikov, gamma sweep')
        for xi, yi, gg in zip(g.inventory_std_lots, g.net, gam):
            ax.annotate(f'{gg:g}', (xi, yi), textcoords='offset points', xytext=(5, 5), fontsize=8, color=MUTED)
        n = t.loc['naive_symmetric']
        ax.scatter([n.inventory_std_lots], [n.net], color=C[1], s=60, zorder=5, label='Naive, same depth, no skew')
        ax.set(title=f'{s}: risk vs PnL (7 test days)', xlabel='inventory std (lots)', ylabel='gross PnL (USD)')
    h, lab = axes[0].get_legend_handles_labels()
    inv = pd.read_csv(results / 'inventory_BTCUSDT.csv', index_col=0, parse_dates=True)
    day = inv.loc['2024-03-26']
    ax = axes[2]
    ax.plot(day.index, day.naive_symmetric, color=C[1], lw=1.2, label='Naive symmetric')
    ax.plot(day.index, day.avellaneda_stoikov, color=C[0], lw=1.2, label='Avellaneda-Stoikov')
    ax.set(title='BTCUSDT inventory, 2024-03-26', ylabel='inventory (lots of 0.01 BTC)')
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    h2, lab2 = ax.get_legend_handles_labels()
    fig.legend(h + h2, lab + lab2, loc='lower center', ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / 'frontier.png', dpi=140)


def latency(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, s in zip(axes, SYMBOLS):
        t = r[s]
        for prefix, label, c in [('touch_latency_', 'Join the touch', C[0]),
                                 ('as_alpha_latency_', 'A-S + imbalance alpha', C[1])]:
            rows = t[t.index.str.startswith(prefix)]
            lat = rows.index.str.replace(prefix, '').astype(int)
            o = np.argsort(lat)
            ax.plot(np.asarray(lat)[o] + 1, rows.gross_bp_of_volume.values[o], color=c, marker='o', ms=5, label=label)
        ax.set_xscale('log')
        ticks = [0, 1, 5, 20, 50, 100, 250, 500]
        ax.set_xticks([v + 1 for v in ticks], [str(v) for v in ticks])
        ax.minorticks_off()
        ax.set(title=f'{s}: cost of latency', xlabel='order latency (ms), requote every 100 ms',
               ylabel='gross PnL, bp of volume')
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'latency.png', dpi=140)


def calibration(results, out):
    cal = json.load(open(results / 'calibration.json'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, s, c in zip(axes[:2], SYMBOLS, C):
        d, lam = np.array(cal[s]['intensity_depths']), np.array(cal[s]['intensity_rates'])
        ax.semilogy(d, lam, 'o', color=c, ms=6, label='measured')
        dd = np.linspace(0.5, d.max(), 200)
        ax.semilogy([0.5], [cal[s]['touch_rate']], 's', color=c, mfc='white', ms=7, mew=2,
                    label='at the touch (not fitted)')
        ax.semilogy(dd, cal[s]['A_per_s'] * np.exp(-cal[s]['k_per_tick'] * dd), color=INK, lw=1.5,
                    label=f"fit A exp(-k d), k = {cal[s]['k_per_tick']:.3g}/tick")
        ax.set(title=f'{s}: market orders reaching depth d', xlabel='depth from mid (ticks)',
               ylabel='rate per side (1/s)')
        ax.legend(fontsize=8.5)
    ax = axes[2]
    for s, c in zip(SYMBOLS, C):
        sig = pd.read_csv(results / f'signal_{s}.csv').groupby('horizon_ms')['corr'].mean()
        ax.plot(sig.index, sig.values ** 2 * 100, color=c, marker='o', ms=5, label=s)
    ax.set_xscale('log')
    ax.set(title='Book imbalance: predictive power, test days', xlabel='horizon (ms)',
           ylabel='squared correlation with future mid change (%)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'calibration.png', dpi=140)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='docs')
    args = ap.parse_args()
    res = Path(args.results)
    r, m = load(res)
    attribution(r, res)
    markouts(m, res)
    frontier(r, res, res)
    latency(r, res)
    calibration(res, res)
    print('figures written to', res)


if __name__ == '__main__':
    main()
