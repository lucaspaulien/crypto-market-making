"""Event-driven simulation of a single market maker against historical Binance data.

The historical book does not contain our orders, so the simulator has to
decide when they would have been filled. It replays three event streams in
time order -- trades, best-bid/ask updates, and our own quoting clock -- and
applies these rules (prices are integer ticks, so every comparison is exact):

**Order placement.** Every ``requote_ms`` the strategy picks a bid and an ask
from the book it sees *now*; the orders only become live ``latency_ms``
later (decisions in flight are queued, so latency may exceed the requote
interval). Orders are post-only: one that would cross the book when it arrives is
rejected, as Binance's GTX orders are. Re-quoting at an unchanged price keeps
the order and therefore its place in the queue.

**Queue position** (``fill_model=QUEUE``, the default). An order that joins
the best bid starts behind the whole displayed size at that price. An order
that improves the best bid has nobody ahead. An order behind the best bid has
an unknown queue and only learns it when its level becomes the best: it then
assumes it is behind everything displayed. Trades at our price consume the
queue ahead of us first; any size decrease of the level that is not
explained by trades (cancellations) also moves us up, conservatively capped
by the displayed size.

**Fills.**
* a taker sell at our bid price fills us once the queue ahead is consumed;
* a taker sell *below* our bid price fills us completely: under price-time
  priority the whole level, us included, was taken first;
* if the best ask drops to our bid or below (the market crossed our resting
  order), we are filled at our price.
Asks are symmetric. ``TOUCH_FILL`` ignores the queue (fills on any trade at
our price; optimistic) and ``THROUGH_FILL`` only fills on trade-throughs and
crosses (pessimistic); they bound the queue model from both sides.

**Same-millisecond events** are processed trades first, then book updates,
then order activations, then our clock. Applying a trade before the book
update it caused avoids counting the same volume twice in the queue.

What is *not* modelled: our orders' impact on other participants, and
queue-position changes that the top-of-book feed cannot reveal (e.g. orders
joining or cancelling *ahead* of us inside the displayed size).
"""
import numpy as np
from numba import njit

from .strategies import FIXED, TOUCH

QUEUE, TOUCH_FILL, THROUGH_FILL = 0, 1, 2
AT_PRICE, TRADE_THROUGH, CROSSED = 0, 1, 2   # how a fill happened
FILL_MODELS = {'queue': QUEUE, 'touch': TOUCH_FILL, 'through': THROUGH_FILL}
_INF = np.inf


@njit(cache=True)
def _record(F, n, t, side, px, qty, mid, how):
    """Append one fill (time, side, price, qty, mid, how) to F, doubling it when full."""
    if n == F.shape[0]:
        G = np.empty((2 * n, 6))
        G[:n] = F
        F = G
    F[n, 0], F[n, 1], F[n, 2], F[n, 3], F[n, 4], F[n, 5] = t, side, px, qty, mid, how
    return F


@njit(cache=True)
def _floor(x):
    return np.int64(np.floor(x + 1e-9))


@njit(cache=True)
def _ceil(x):
    return np.int64(np.ceil(x - 1e-9))


@njit(cache=True)
def _simulate(book_t, bid, bid_qty, ask, ask_qty, tr_t, tr_px, tr_qty, tr_side,
              kind, params, size, max_lots, latency_ms, requote_ms, sample_ms, fill_model,
              tick, fee_rate):
    nb, nt = book_t.shape[0], tr_t.shape[0]
    t_end = max(book_t[nb - 1], tr_t[nt - 1])

    # fills: time, side (+1 buy / -1 sell), price (ticks), qty, mid before the fill (ticks), how
    F = np.empty((1 << 14, 6))
    nf = 0
    n_samples = (t_end - book_t[0]) // sample_ms + 3
    s_t = np.empty(n_samples, np.int64)
    s_inv = np.empty(n_samples)
    s_cash = np.empty(n_samples)
    s_mid = np.empty(n_samples)
    s_fees = np.empty(n_samples)
    ns = 0

    # market state starts at the first book update
    ib, it = 1, 0
    while it < nt and tr_t[it] < book_t[0]:
        it += 1
    cb, cbq, ca, caq = bid[0], bid_qty[0], ask[0], ask_qty[0]

    inv, cash, fees = 0.0, 0.0, 0.0
    n_orders, n_rejected = 0, 0
    # live orders
    b_on, b_px, b_rem, b_qa = False, np.int64(0), 0.0, _INF
    a_on, a_px, a_rem, a_qa = False, np.int64(0), 0.0, _INF
    # quote decisions in flight (sent, not yet live): FIFO ring buffer, since with
    # latency > requote interval several decisions can be travelling at once
    QN = 1024
    q_t = np.empty(QN, np.int64)
    q_bon = np.empty(QN, np.bool_)
    q_bpx = np.empty(QN, np.int64)
    q_aon = np.empty(QN, np.bool_)
    q_apx = np.empty(QN, np.int64)
    q_head, q_len = 0, 0
    t_clock = book_t[0]
    t_sample = book_t[0]

    while True:
        t_tr = tr_t[it] if it < nt else np.int64(1) << 62
        t_bk = book_t[ib] if ib < nb else np.int64(1) << 62
        t_pd = q_t[q_head] if q_len > 0 else np.int64(1) << 62
        t_now = min(t_tr, t_bk, t_pd, t_clock)
        if t_now > t_end:
            break

        fill_b, fill_a, how, ref_mid = 0.0, 0.0, AT_PRICE, 0.5 * (cb + ca)
        if t_tr == t_now:
            # ---------------------------------------------------------- trade
            px, qty, side = tr_px[it], tr_qty[it], tr_side[it]
            it += 1
            if side < 0 and b_on:            # taker sell hits bids
                if px < b_px:
                    fill_b, how = b_rem, TRADE_THROUGH
                elif px == b_px:
                    if fill_model == TOUCH_FILL:
                        fill_b = min(b_rem, qty)
                    elif fill_model == QUEUE:
                        fill_b = min(b_rem, max(qty - b_qa, 0.0))
                        b_qa = max(b_qa - qty, 0.0)
            elif side > 0 and a_on:          # taker buy lifts asks
                if px > a_px:
                    fill_a, how = a_rem, TRADE_THROUGH
                elif px == a_px:
                    if fill_model == TOUCH_FILL:
                        fill_a = min(a_rem, qty)
                    elif fill_model == QUEUE:
                        fill_a = min(a_rem, max(qty - a_qa, 0.0))
                        a_qa = max(a_qa - qty, 0.0)

        elif t_bk == t_now:
            # ----------------------------------------------------- book update
            cb, cbq, ca, caq = bid[ib], bid_qty[ib], ask[ib], ask_qty[ib]
            ib += 1
            # a fill caused by this update is measured against the mid before it
            how = CROSSED
            if b_on:
                if ca <= b_px:               # the market crossed our bid
                    fill_b = b_rem
                elif b_px == cb:
                    b_qa = min(b_qa, cbq)
                elif b_px > cb:
                    b_qa = 0.0               # everyone who was at our price is gone
            if a_on:
                if cb >= a_px:
                    fill_a = a_rem
                elif a_px == ca:
                    a_qa = min(a_qa, caq)
                elif a_px < ca:
                    a_qa = 0.0

        elif t_pd == t_now:
            # ------------------------------------------------ order activation
            pb_on, pb_px = q_bon[q_head], q_bpx[q_head]
            pa_on, pa_px = q_aon[q_head], q_apx[q_head]
            q_head = (q_head + 1) % QN
            q_len -= 1
            if not pb_on:
                b_on = False
            elif not (b_on and b_px == pb_px):
                n_orders += 1
                if pb_px >= ca:              # post-only: would take liquidity
                    b_on = False
                    n_rejected += 1
                else:
                    b_on, b_px, b_rem = True, pb_px, size
                    b_qa = cbq if pb_px == cb else (0.0 if pb_px > cb else _INF)
            if not pa_on:
                a_on = False
            elif not (a_on and a_px == pa_px):
                n_orders += 1
                if pa_px <= cb:
                    a_on = False
                    n_rejected += 1
                else:
                    a_on, a_px, a_rem = True, pa_px, size
                    a_qa = caq if pa_px == ca else (0.0 if pa_px < ca else _INF)

        else:
            # -------------------------------------------- our clock: requote
            t_clock += requote_ms
            while t_sample <= t_now:
                s_t[ns], s_inv[ns], s_cash[ns], s_mid[ns], s_fees[ns] = t_sample, inv, cash, 0.5 * (cb + ca), fees
                ns += 1
                t_sample += sample_ms
            q = inv / size
            # fair value: mid, shifted towards the heavier side of the book
            mid = 0.5 * (cb + ca) + params[2] * (cbq - caq) / (cbq + caq)
            if kind == TOUCH:
                nbp, nap = cb, ca
            else:
                if kind == FIXED:
                    db = params[0] + params[1] * q
                    da = params[0] - params[1] * q
                else:  # GLFT (Avellaneda-Stoikov)
                    db = params[0] + (2.0 * q + 1.0) / 2.0 * params[1]
                    da = params[0] - (2.0 * q - 1.0) / 2.0 * params[1]
                nbp = min(_floor(mid - db), ca - 1)
                nap = max(_ceil(mid + da), cb + 1)
            # with zero latency the orders go live on the next iteration, same ms
            if q_len < QN:
                j = (q_head + q_len) % QN
                q_t[j] = t_now + latency_ms
                q_bon[j] = q + 1.0 <= max_lots + 1e-9
                q_aon[j] = q - 1.0 >= -max_lots - 1e-9
                q_bpx[j], q_apx[j] = nbp, nap
                q_len += 1

        # ------------------------------------------------------ apply fills
        if fill_b > 0:
            F = _record(F, nf, t_now, 1, b_px, fill_b, ref_mid, how)
            nf += 1
            inv += fill_b
            cash -= fill_b * b_px * tick
            fees += fee_rate * fill_b * b_px * tick
            b_rem -= fill_b
            b_on = b_rem > 1e-12
        if fill_a > 0:
            F = _record(F, nf, t_now, -1, a_px, fill_a, ref_mid, how)
            nf += 1
            inv -= fill_a
            cash += fill_a * a_px * tick
            fees += fee_rate * fill_a * a_px * tick
            a_rem -= fill_a
            a_on = a_rem > 1e-12

    while t_sample < t_end and ns < n_samples - 1:
        s_t[ns], s_inv[ns], s_cash[ns], s_mid[ns], s_fees[ns] = t_sample, inv, cash, 0.5 * (cb + ca), fees
        ns += 1
        t_sample += sample_ms
    # the last sample is always the exact final state
    s_t[ns], s_inv[ns], s_cash[ns], s_mid[ns], s_fees[ns] = t_end, inv, cash, 0.5 * (cb + ca), fees
    ns += 1

    return (F[:nf], s_t[:ns], s_inv[:ns], s_cash[:ns], s_mid[:ns], s_fees[:ns],
            n_orders, n_rejected)


def simulate(md, strategy, size, max_lots=10, latency_ms=20, requote_ms=100, sample_ms=1000,
             fill_model='queue', fee_rate=0.0):
    """Run one strategy over ``md`` (a ``MarketData``). Returns a ``Result``.

    ``size`` is the quote size in base units (e.g. BTC), ``max_lots`` the
    inventory limit in multiples of it, ``fee_rate`` the maker fee as a
    fraction of notional (0.0002 = 2 bp; negative = rebate).
    """
    from .pnl import Result

    kind, params = strategy.encode()
    out = _simulate(md.book_t, md.bid, md.bid_qty, md.ask, md.ask_qty,
                    md.trade_t, md.trade_px, md.trade_qty, md.trade_side,
                    kind, params.astype(np.float64), float(size), float(max_lots),
                    np.int64(latency_ms), np.int64(requote_ms), np.int64(sample_ms),
                    FILL_MODELS[fill_model], float(md.tick), float(fee_rate))
    return Result(md, strategy, size, *out)
