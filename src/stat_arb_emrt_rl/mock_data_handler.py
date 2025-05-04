# mock_data_handler.py
import json
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Dict, Any


def save_backtest_data(
    strategy, end_date: str, filename: str = "backtest_mock_data.json"
):
    """Save backtest results to file from strategy object"""
    # Convert tuple keys to strings
    price_history = {f"{k[0]}/{k[1]}": v for k, v in strategy.price_history.items()}

    backtest_data = {
        "spread_analysis": convert_tuple_keys(
            strategy.analyzers.spread_tracker.get_analysis()
        ),
        "trade_analysis": strategy.trade_recorder.get_trades(),
        "price_history": price_history,
        "params": {"start_date": strategy.start_date.isoformat(), "end_date": end_date},
    }

    def convert_value(v):
        if isinstance(v, (date, datetime, np.datetime64)):
            return (
                v.isoformat()
                if hasattr(v, "isoformat")
                else pd.to_datetime(v).isoformat()
            )
        elif isinstance(v, np.generic):
            return v.item()
        return v

    with open(filename, "w") as f:
        json.dump(backtest_data, f, default=convert_value, indent=2)


def convert_tuple_keys(data):
    """Recursively convert tuple keys to strings"""
    if isinstance(data, dict):
        return {
            f"{k[0]}/{k[1]}" if isinstance(k, tuple) else k: convert_tuple_keys(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [convert_tuple_keys(item) for item in data]
    return data


def load_mock_data(filename: str = "backtest_mock_data.json") -> Dict[str, Any]:
    """Load mock data from file for development"""
    with open(filename, "r") as f:
        raw = json.load(f)

    # Convert string keys back to tuples
    raw["price_history"] = {
        tuple(k.split("/")): v for k, v in raw["price_history"].items()
    }

    # Convert spread analysis keys
    raw["spread_analysis"] = convert_str_keys(raw["spread_analysis"])

    # Convert dates
    for pair in raw["spread_analysis"]["dates"]:
        raw["spread_analysis"]["dates"][pair] = [
            datetime.fromisoformat(d) for d in raw["spread_analysis"]["dates"][pair]
        ]

    return raw


def convert_str_keys(data):
    """Convert string keys back to tuples"""
    if isinstance(data, dict):
        return {
            tuple(k.split("/")) if "/" in k else k: convert_str_keys(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [convert_str_keys(item) for item in data]
    return data


class MockStrategy:
    """Replicates strategy interface for plotting functions"""

    def __init__(self, data: Dict[str, Any]):
        self.price_history = data["price_history"]
        self.start_date = datetime.fromisoformat(data["params"]["start_date"])
        self.trade_recorder = self.MockTradeRecorder(data["trade_analysis"])

    class MockTradeRecorder:
        def __init__(self, trades):
            self.trades = trades

        def get_trades(self):
            return self.trades


class MockAnalyzer:
    """Replicates analyzer interface for plotting functions"""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def get_analysis(self):
        return self.data


def get_mock_results(filename: str = "backtest_mock_data.json") -> list:
    """Get mock results in Cerebro-compatible format"""
    raw = load_mock_data(filename)
    return [
        type(
            "",
            (),
            {
                "analyzers": [MockAnalyzer(raw["spread_analysis"])],
                "trade_recorder": MockStrategy(raw).trade_recorder,
                "price_history": MockStrategy(raw).price_history,
                "start_date": MockStrategy(raw).start_date,
            },
        )
    ]
