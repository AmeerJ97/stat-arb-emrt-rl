# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
import re
from typing import List, Tuple, Any, Optional, Union
from datetime import datetime, date, time

# ───────────────────────────────────────────────
# Third-Party Package Imports
# ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# ───────────────────────────────────────────────
# Project-Specific Imports
# ───────────────────────────────────────────────
from ..printing_system import buffered_print, BOLD, YELLOW, ENDC

COLOR_PAIRS = [
    ("#1F77B4", "#FF7F0E"),  # Blue/Orange
    ("#2CA02C", "#D62728"),  # Green/Red
    ("#9467BD", "#8C564B"),  # Purple/Brown
    ("#E377C2", "#7F7F7F"),  # Pink/Gray
    ("#BCBD22", "#17BECF"),  # Yellow/Cyan
]

# COLOR_CYCLE = [pair[0] for pair in COLOR_PAIRS]  # Reuse existing color pairs

# COLOR_CYCLE = [
#     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
#     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
#     "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
#     "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
#     "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
#     "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39"
# ]
COLOR_CYCLE = [
    "#1f77b4", "#ff0000", "#ff7f0e", "#ffff00", "#2ca02c", "#00ff00",
    "#d62728", "#ff00ff", "#9467bd", "#00ffff", "#8c564b", "#ff8000",
    "#e377c2", "#80ff00", "#7f7f7f", "#ff0080", "#bcbd22", "#00ff80",
    "#17becf", "#8000ff", "#aec7e8", "#ff7f7f", "#ffbb78", "#7fff7f",
    "#98df8a", "#7f7fff", "#ff9896", "#ff7fff", "#c5b0d5", "#7fffff",
    "#c49c94", "#ffbf7f", "#f7b6d2", "#bfff7f", "#c7c7c7", "#bf7fff",
    "#dbdb8d", "#ffbfbf", "#9edae5", "#bfffff", "#393b79", "#ff7fbf",
    "#5254a3", "#7fbfff", "#6b6ecf", "#bf7fbf", "#9c9ede", "#7fbf7f",
    "#637939", "#bfbf7f", "#8ca252", "#7f7fbf", "#b5cf6b", "#bf7f7f",
    "#cedb9c", "#7fbfbf", "#8c6d31", "#ffbf80", "#bd9e39", "#80bfff"
]
# _backend_printed = False  # Module-level flag


# def check_backend():
#     global _backend_printed
#     if not _backend_printed:
#         buffered_print(
#             f"{BOLD}{YELLOW}Matplotlib backend: {ENDC}{matplotlib.get_backend()}"
#         )
#         buffered_print(
#             f"Backend initialized at: {datetime.now().strftime('%H:%M:%S.%f')}"
#         )
#         _backend_printed = True


def sanitize_filename(name: str) -> str:
    """Replace invalid filesystem characters with underscores"""
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def get_trade_color(index):
    """Get color from predefined cycle"""
    return COLOR_CYCLE[index % len(COLOR_CYCLE)]


def find_nearest_date_index(date_list: np.ndarray,
                            target_date_value: Union[float, datetime, pd.Timestamp]) -> Optional[int]:
    """
    Finds the index of the nearest date in a list of dates to a target_date_value.
    Assumes date_list is sorted.
    Handles cases where date_list contains numeric matplotlib dates or datetime objects.

    Args:
        date_list: Sorted np.ndarray of dates (can be numeric floats or datetime objects).
        target_date_value: The date to find the nearest index for (can be a numeric float,
                           datetime.datetime, or pd.Timestamp).

    Returns:
        The index in date_list, or None if list is empty or types are incompatible.
    """
    if date_list is None or len(date_list) == 0:
        return None

    # Determine if the date_list is numeric by checking its first element.
    # This is crucial for consistent comparison.
    is_date_list_numeric = isinstance(date_list[0], (int, float, np.number))

    # Will hold the numeric representation of the target date
    processed_target_date: float

    if is_date_list_numeric:
        # If date_list is numeric, ensure target_date_value is also numeric.
        # date is imported from datetime
        if isinstance(target_date_value, (datetime, pd.Timestamp, date)):
            # Convert datetime-like target to numeric matplotlib date
            try:
                processed_target_date = mdates.date2num(target_date_value)
            except Exception as e:
                # If conversion fails, we can't proceed with numeric comparison.
                # Consider logging this error with buffered_print if available
                # buffered_print(f"find_nearest_date_index: Error converting target_date {target_date_value} to numeric: {e}", "ERROR")
                return None
        elif isinstance(target_date_value, (int, float, np.number)):
            processed_target_date = float(
                target_date_value)  # Ensure it's a float
        else:
            # Target date is of an unhandled type for numeric comparison
            return None
    else:
        # If date_list is not numeric (i.e., contains datetime objects),
        # this function is not directly suited for searchsorted without converting date_list first.
        # For the current plotting use case, date_list (x_dates_numeric) IS numeric.
        # If it were ever called with a datetime list, date_list would need conversion to numeric,
        # or a different comparison logic using datetime objects would be required.
        # For simplicity and to address the current error, we focus on the numeric path.
        # buffered_print("find_nearest_date_index: date_list is not numeric, this path is not fully handled for plotting.", "WARNING")
        # Fallback: attempt to convert target to pd.Timestamp and hope date_list contains compatible objects.
        # This part is less robust if date_list truly isn't numeric.
        try:
            processed_target_date_ts = pd.Timestamp(target_date_value)
            # searchsorted on an array of datetime objects with a Timestamp
            idx_ts = date_list.searchsorted(
                processed_target_date_ts, side="left")  # type: ignore
            # This path is less exercised by current problem, but provided for completeness if utils is generic.
            # The main fix is for the numeric path above.
            if idx_ts == 0:
                return 0
            if idx_ts == len(date_list):
                return len(date_list) - 1

            # comparison with datetime objects
            val_before_ts = pd.Timestamp(date_list[idx_ts-1])
            val_after_ts = pd.Timestamp(date_list[idx_ts])

            if abs(processed_target_date_ts - val_before_ts) < abs(val_after_ts - processed_target_date_ts):
                return idx_ts - 1
            else:
                return idx_ts

        except Exception:
            return None  # Failed to process as datetime list path

    # At this point, date_list is numeric and processed_target_date is a numeric float.
    # searchsorted works directly with these numeric types.
    try:
        idx = date_list.searchsorted(processed_target_date, side="left")
    except TypeError:
        # Should not happen if processed_target_date is correctly a float here.
        return None

    # Determine the closest index by comparing distances.
    if idx == 0:
        # If target is before or at the first element.
        return 0
    if idx == len(date_list):
        # If target is after or at the last element.
        return len(date_list) - 1

    # Compare distance to the element at idx-1 and idx.
    # Both date_list elements and processed_target_date are numeric floats.
    diff_before = abs(processed_target_date - date_list[idx - 1])
    diff_after = abs(date_list[idx] - processed_target_date)

    if diff_before < diff_after:
        return idx - 1
    else:
        return idx
# def get_pair_colors(pair_idx):
#     """Cycles through predefined color pairs with offset for unique per-trade colors"""
#     base_colors = COLOR_PAIRS[pair_idx % len(COLOR_PAIRS)]
#     return [
#         base_colors[0],  # Asset 1 color
#         base_colors[1],  # Asset 2 color
#         COLOR_CYCLE[(pair_idx * 2) % len(COLOR_CYCLE)],  # Trade marker base color
#     ]


def get_pair_colors(pair_idx):
    """Cycles through predefined color pairs with offset for unique per-trade colors"""
    base_colors = COLOR_PAIRS[pair_idx % len(COLOR_PAIRS)]
    return {
        "price1": base_colors[0],
        "price2": base_colors[1],
        "spread": COLOR_CYCLE[(pair_idx * 2) % len(COLOR_CYCLE)],
        "mean": "#FFD700",  # Consistent mean color
        "bands": "#666666",  # Consistent band color
        "trade": base_colors[2] if len(base_colors) > 2 else "#FF69B4",
    }


# def get_pair_colors(pair_idx):
#     """Cycles through predefined color pairs for consistent assignment."""
#     return COLOR_PAIRS[pair_idx % len(COLOR_PAIRS)]


def find_closest_date(target, dates):
    """Helper function for date alignment"""
    return min(dates, key=lambda d: abs(d - target))


# def find_nearest_date_index(target_date, date_list):
#     """Find index of nearest date in sorted list with type conversion"""
#     if not date_list:
#         return 0

#     # target_date = pd.to_datetime(target_date)
#     # date_list = [pd.to_datetime(d) for d in date_list]
#     # deltas = [abs(date - target_date) for date in date_list]
#     # return deltas.index(min(deltas))

#     date_list = pd.Series(date_list)
#     idx = date_list.searchsorted(pd.Timestamp(target_date), side="left")

#     if idx > 0 and (
#         idx == len(date_list)
#         or abs(date_list.iloc[idx] - target_date)
#         > abs(date_list.iloc[idx - 1] - target_date)
#     ):
#         return idx - 1
#     else:
#         return idx


def get_trade_marker_config(trade_leg):
    """Returns marker details based on trade type"""
    config = {
        "entry": {"marker": "", "color": "", "size": 100},
        "exit": {"marker": "X", "color": "", "size": 100},
    }

    color = COLOR_CYCLE[trade_leg["color_idx"] % len(COLOR_CYCLE)]

    if trade_leg["asset"] == "S1":
        if trade_leg["entry_type"] == "long":
            config["entry"]["marker"] = "^"
        else:
            config["entry"]["marker"] = "v"
    else:  # S2
        if trade_leg["entry_type"] == "long":
            config["entry"]["marker"] = "^"
        else:
            config["entry"]["marker"] = "v"

    config["entry"]["color"] = color
    config["exit"]["color"] = color
    return config


# def check_backend():
#     global _backend_printed
#     if not _backend_printed:
#         buffered_print(
#             f"{BOLD}{YELLOW}Matplotlib backend: {ENDC}{BOLD}{matplotlib.get_backend()}{ENDC}"
#         )
#         buffered_print(
#             f"Backend initialized at: {datetime.datetime.now().strftime('%H:%M:%S.%f')}"
#         )
#         _backend_printed = True
