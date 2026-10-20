"""Download and convert Binance USD-M futures bookTicker + aggTrades days.

    python scripts/download_data.py --symbols BTCUSDT XRPUSDT --start 2024-03-23 --end 2024-03-30

~100 MB of zips per symbol-day; the converted .npz files are ~3x smaller.
"""
import argparse

import pandas as pd

from mmsim.data import convert_day, default_data_dir, download_day


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'XRPUSDT'])
    ap.add_argument('--start', default='2024-03-23')
    ap.add_argument('--end', default='2024-03-30')
    ap.add_argument('--data-dir', default=str(default_data_dir()))
    ap.add_argument('--keep-raw', action='store_true', help='keep the downloaded zips')
    args = ap.parse_args()

    for day in pd.date_range(args.start, args.end).strftime('%Y-%m-%d'):
        for s in args.symbols:
            zips = download_day(s, day, args.data_dir)
            out = convert_day(s, day, args.data_dir)
            if not args.keep_raw:
                for z in zips:
                    z.unlink()
            print('ready', out, flush=True)


if __name__ == '__main__':
    main()
