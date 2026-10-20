"""Binance USD-M futures top-of-book and trade data.

Binance publishes, for every perpetual, one file per day of

* ``bookTicker``: every change of the best bid / best ask (price and size),
  with millisecond timestamps -- ~7 million updates per day on BTCUSDT;
* ``aggTrades``: every trade, aggregated per taker order and price level,
  with the aggressor side.

``download_day`` fetches them from https://data.binance.vision (public, no
key needed; bookTicker files exist up to 2024-03-30) and ``convert_day``
turns them into one compact ``.npz`` with prices stored as **integer ticks**,
so the simulator compares prices exactly instead of comparing floats.
"""
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = 'https://data.binance.vision/data/futures/um/daily'
TICK_SIZE = {'BTCUSDT': 0.1, 'ETHUSDT': 0.01, 'SOLUSDT': 0.01, 'XRPUSDT': 0.0001,
             'DOGEUSDT': 0.00001, 'ADAUSDT': 0.0001, 'BNBUSDT': 0.01}


def default_data_dir():
    return Path(os.environ.get('MMSIM_DATA', 'data'))


def _url(kind, symbol, day):
    return f'{BASE_URL}/{kind}/{symbol}/{symbol}-{kind}-{day}.zip'


def download_day(symbol, day, data_dir, overwrite=False):
    """Download the bookTicker and aggTrades zips of one day into ``data_dir/raw``."""
    raw = Path(data_dir) / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in ('bookTicker', 'aggTrades'):
        dest = raw / f'{symbol}-{kind}-{day}.zip'
        if overwrite or not dest.exists():
            tmp = dest.with_suffix('.part')
            urllib.request.urlretrieve(_url(kind, symbol, day), tmp)
            tmp.rename(dest)
        paths.append(dest)
    return paths


@dataclass
class MarketData:
    """Top of book and trades of one or several consecutive days, prices in ticks."""
    symbol: str
    tick: float
    book_t: np.ndarray     # int64 ms
    bid: np.ndarray        # int64 ticks
    bid_qty: np.ndarray    # float64, base units
    ask: np.ndarray
    ask_qty: np.ndarray
    trade_t: np.ndarray    # int64 ms
    trade_px: np.ndarray   # int64 ticks
    trade_qty: np.ndarray  # float64
    trade_side: np.ndarray  # int8: +1 taker buy (hits asks), -1 taker sell (hits bids)

    @property
    def mid(self):
        """Mid price in ticks (a half-integer when the spread is odd)."""
        return 0.5 * (self.bid + self.ask)

    @property
    def start(self):
        return int(min(self.book_t[0], self.trade_t[0]))

    @property
    def end(self):
        return int(max(self.book_t[-1], self.trade_t[-1]))

    def to_price(self, ticks):
        return np.asarray(ticks) * self.tick


def to_ticks(price, tick):
    return np.rint(np.asarray(price, dtype=float) / tick).astype(np.int64)


def convert_day(symbol, day, data_dir, tick=None):
    """Raw zips -> ``data_dir/<symbol>-<day>.npz``. Returns the npz path."""
    tick = TICK_SIZE[symbol] if tick is None else tick
    raw = Path(data_dir) / 'raw'
    b = pd.read_csv(raw / f'{symbol}-bookTicker-{day}.zip',
                    usecols=['best_bid_price', 'best_bid_qty', 'best_ask_price', 'best_ask_qty',
                             'transaction_time'])
    a = pd.read_csv(raw / f'{symbol}-aggTrades-{day}.zip',
                    usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'])
    # both files are sorted by time; keep that order on ties (stable sort is a no-op check)
    b = b.sort_values('transaction_time', kind='stable')
    a = a.sort_values('transact_time', kind='stable')
    out = Path(data_dir) / f'{symbol}-{day}.npz'
    np.savez_compressed(
        out, tick=tick,
        book_t=b.transaction_time.to_numpy(np.int64),
        bid=to_ticks(b.best_bid_price, tick), bid_qty=b.best_bid_qty.to_numpy(float),
        ask=to_ticks(b.best_ask_price, tick), ask_qty=b.best_ask_qty.to_numpy(float),
        trade_t=a.transact_time.to_numpy(np.int64),
        trade_px=to_ticks(a.price, tick), trade_qty=a.quantity.to_numpy(float),
        # is_buyer_maker = the buyer was resting, so the seller was the aggressor
        trade_side=np.where(a.is_buyer_maker.to_numpy(bool), -1, 1).astype(np.int8),
    )
    return out


def load(symbol, days, data_dir):
    """Load and concatenate converted days (must be consecutive, in order)."""
    parts = [np.load(Path(data_dir) / f'{symbol}-{d}.npz') for d in days]
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    md = MarketData(symbol=symbol, tick=float(parts[0]['tick']),
                    book_t=cat('book_t'), bid=cat('bid'), bid_qty=cat('bid_qty'),
                    ask=cat('ask'), ask_qty=cat('ask_qty'), trade_t=cat('trade_t'),
                    trade_px=cat('trade_px'), trade_qty=cat('trade_qty'), trade_side=cat('trade_side'))
    if np.any(np.diff(md.book_t) < 0) or np.any(np.diff(md.trade_t) < 0):
        raise ValueError('days are not consecutive / not in chronological order')
    return md
