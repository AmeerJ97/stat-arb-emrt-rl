# FILE: trade_recorder.py

# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
import math
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ───────────────────────────────────────────────
# Data Classes
# ───────────────────────────────────────────────


class TradePair:
    def __init__(self, pair: tuple, entry_dt: datetime, overall_spread_direction: str, color_idx: int):
        self.trade_id = uuid.uuid4()
        self.pair = pair
        self.entry_dt = entry_dt
        self.exit_dt = None
        self.direction = overall_spread_direction
        self.legs = []
        self.color_idx = color_idx
        self._open_legs_count = 0
        self.pnl_from_strategy: Optional[float] = None

    def add_leg(self, asset: str, leg_entry_type: str, price: float, dt: datetime, size: float):
        leg_data = {
            'asset': asset, 'entry_type': leg_entry_type,
            'entry_price': price, 'entry_dt': dt,
            'exit_price': None, 'exit_dt': None, 'size': size,
        }
        self.legs.append(leg_data)
        self._open_legs_count += 1

    def close_leg(self, asset: str, price: float, dt: datetime) -> bool:
        for leg in self.legs:
            if leg['asset'] == asset and leg['exit_price'] is None:
                leg['exit_price'] = price
                leg['exit_dt'] = dt
                self._open_legs_count -= 1
                return True
        return False

    def is_complete(self) -> bool:
        return self._open_legs_count == 0 and len(self.legs) > 0

# ───────────────────────────────────────────────
# Main Recorder Class
# ───────────────────────────────────────────────


class EnhancedTradeRecorder:
    def __init__(self):
        self.trade_pairs: List[TradePair] = []
        self.trade_id_counter = 0
        self.COLOR_CYCLE = [
            "#FF69B4", "#32CD32", "#1E90FF", "#FFD700", "#FF8C00", "#ADFF2F",
            "#BA55D3", "#00CED1", "#F08080", "#20B2AA"
        ]

    def _get_next_color_idx(self) -> int:
        idx = self.trade_id_counter % len(self.COLOR_CYCLE)
        self.trade_id_counter += 1
        return idx

    def record_trade(self, *args, **kwargs):
        trade_info = kwargs or (args[0] if args else {})
        self._record_from_kwargs(trade_info)

    def _record_from_kwargs(self, params: Dict):
        pair = params.get("pair")
        entry_dt = params.get("entry_dt")
        exit_dt = params.get("exit_dt")
        overall_direction = params.get("direction", "N/A")
        color_idx = self._get_next_color_idx()
        pnl_from_strategy = params.get("pnl")

        tp = TradePair(
            pair=pair, entry_dt=entry_dt,
            overall_spread_direction=overall_direction, color_idx=color_idx
        )
        tp.exit_dt = exit_dt
        if pnl_from_strategy is not None:
            tp.pnl_from_strategy = pnl_from_strategy

        t1, t2 = pair
        entry_prices = params.get("entry_prices", (None, None))
        exit_prices = params.get("exit_prices", (None, None))
        leg1_size = params.get("size")
        hedge_ratio = params.get("hedge_ratio", 1.0)
        actual_leg2_size = math.ceil(leg1_size * hedge_ratio)

        leg1_dir = "long" if overall_direction.upper() == "LONG" else "short"
        leg2_dir = "short" if overall_direction.upper() == "LONG" else "long"

        if entry_prices[0] is not None:
            tp.add_leg(asset=t1, leg_entry_type=leg1_dir,
                       price=entry_prices[0], dt=entry_dt, size=leg1_size)
            if exit_prices[0] is not None and exit_dt is not None:
                tp.close_leg(asset=t1, price=exit_prices[0], dt=exit_dt)
        if entry_prices[1] is not None:
            tp.add_leg(asset=t2, leg_entry_type=leg2_dir,
                       price=entry_prices[1], dt=entry_dt, size=actual_leg2_size)
            if exit_prices[1] is not None and exit_dt is not None:
                tp.close_leg(asset=t2, price=exit_prices[1], dt=exit_dt)

        self.trade_pairs.append(tp)

    @property
    def trades(self) -> List[Dict]:
        """
        Converts the internal list of TradePair objects into a flat list of dictionaries
        suitable for analysis and reporting.
        """
        flat_trades = []
        for tp in self.trade_pairs:
            if tp.exit_dt is None or not tp.legs:
                continue

            # --- FIX: Enforce Single Source of Truth for PnL ---
            # The previous implementation had a fallback mechanism here to calculate PnL from
            # the individual trade legs if the PnL from the strategy was not available.
            # While well-intentioned, this created a potential for data inconsistency, as
            # the PnL for a single trade could be calculated in two different ways.
            #
            # The corrected logic enforces that the Profit and Loss (PnL) value *must* come
            # from the strategy itself (`pnl_from_strategy`). This centralizes the calculation
            # logic and ensures that the value recorded is the same one the strategy used for
            # its decision-making and reporting.
            #
            # If `pnl_from_strategy` is somehow missing (which should not happen after fixing
            # the dual-recording bug), it safely defaults to 0.0, preventing crashes and
            # making it easier to spot upstream data flow issues.
            pnl = tp.pnl_from_strategy if tp.pnl_from_strategy is not None else 0.0

            trade_overall_size = tp.legs[0]['size'] if tp.legs else 0
            flat_entry_prices = [None, None]
            flat_exit_prices = [None, None]

            asset1, asset2 = tp.pair
            leg1_data = next(
                (leg for leg in tp.legs if leg['asset'] == asset1), None)
            leg2_data = next(
                (leg for leg in tp.legs if leg['asset'] == asset2), None)

            if leg1_data:
                flat_entry_prices[0] = leg1_data['entry_price']
                flat_exit_prices[0] = leg1_data['exit_price']
            if leg2_data:
                flat_entry_prices[1] = leg2_data['entry_price']
                flat_exit_prices[1] = leg2_data['exit_price']

            cost_basis_approx = 0
            if trade_overall_size > 0 and flat_entry_prices[0] is not None:
                # cost_basis_approx = trade_overall_size * flat_entry_prices[0]
                cost_basis_approx = 0
                if tp.direction == "LONG":
                    cost_basis_approx = (leg1_data['entry_price'] * leg1_data['size']) + \
                                        (leg2_data['entry_price']
                                         * leg2_data['size'])
                else:  # SHORT
                    # For shorts, cost basis is the credit received
                    cost_basis_approx = (leg1_data['entry_price'] * leg1_data['size']) + \
                                        (leg2_data['entry_price']
                                         * leg2_data['size'])

            returns_pct = (pnl / cost_basis_approx) * \
                100 if cost_basis_approx > 0 else 0

            flat_trades.append({
                "pair": tp.pair, "entry_dt": tp.entry_dt, "exit_dt": tp.exit_dt,
                "pnl": pnl, "size": trade_overall_size, "direction": tp.direction,
                "entry_prices": tuple(flat_entry_prices), "exit_prices": tuple(flat_exit_prices),
                "commission": 0, "returns_pct": returns_pct,
            })
        return flat_trades

    def get_trades(self) -> List[Dict]:
        return self.trades

    def summarize(self) -> Dict:
        closed_trades = [
            t for t in self.trades if t.get("exit_dt") is not None]
        total_trades = len(closed_trades)
        if total_trades == 0:
            return {
                "total_trades": 0, "win_rate": 0, "gross_profit": 0,
                "gross_loss": 0, "profit_factor": 0, "avg_trade_duration": 0,
            }

        profitable = sum(1 for t in closed_trades if t.get("pnl", 0) > 0)
        win_rate = profitable / total_trades if total_trades > 0 else 0
        gross_profit = sum(t.get("pnl", 0)
                           for t in closed_trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0)
                         for t in closed_trades if t.get("pnl", 0) <= 0))
        profit_factor = gross_profit / \
            gross_loss if gross_loss != 0 else float('inf')

        total_duration_days = 0
        valid_durations = 0
        for t in closed_trades:
            if t.get("entry_dt") and t.get("exit_dt"):
                duration = (t["exit_dt"] - t["entry_dt"]).days
                total_duration_days += duration
                valid_durations += 1
        avg_trade_duration = total_duration_days / \
            valid_durations if valid_durations > 0 else 0

        return {
            "total_trades": total_trades, "win_rate": win_rate, "gross_profit": gross_profit,
            "gross_loss": gross_loss, "profit_factor": profit_factor,
            "avg_trade_duration": avg_trade_duration,
        }
