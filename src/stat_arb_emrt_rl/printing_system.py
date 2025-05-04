# ───────────────────────────────────────────────
# Standard Library Imports
# ───────────────────────────────────────────────
import sys
import os
import time
import logging
import re
import shutil
import textwrap
from contextlib import contextmanager
from itertools import islice
from datetime import datetime
from queue import Queue, Empty
from logging.handlers import QueueHandler, QueueListener
from typing import Union

# ───────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────
MAX_LINE_WIDTH = 80
PRINTER_STARTUP_TIMEOUT = 10  # Increased timeout for heavy systems

# ───────────────────────────────────────────────
# Logging Configuration
# ───────────────────────────────────────────────
log_queue = Queue(-1)
queue_listener = None


def setup_logging():
    global queue_listener

    if queue_listener is not None:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    log_dir = os.environ.get("EMRT_LOG_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, "logs.log"), encoding="utf-8")

    # Configure formatters and other settings
    console_handler.setFormatter(ColorFormatter("%(message)s"))
    file_handler.setFormatter(
        CleanFormatter("%(asctime)s [%(threadName)s] %(levelname)s %(message)s")
    )

    # Initialize the QueueListener here
    queue_listener = QueueListener(
        log_queue, console_handler, file_handler, respect_handler_level=True
    )
    queue_listener.start()

    root_logger.addHandler(QueueHandler(log_queue))


# ───────────────────────────────────────────────
# Color Constants
# ───────────────────────────────────────────────
class Colors:
    HEADER = "\033[38;5;250m"
    OKBLUE = "\033[38;5;27m"
    OKGREEN = "\033[38;5;34m"
    WARNING = "\033[38;5;214m"
    FAIL = "\033[38;5;196m"
    CYAN = "\033[38;5;51m"
    PURPLE = "\033[95m"
    GREY = "\033[38;5;240m"
    ORANGE = "\033[38;5;202m"
    BLUE = "\033[34m"
    UNDERLINE = "\033[4m"
    DARKBG = "\033[48;5;235m"
    LABEL = "\033[38;5;33m"
    VALUE = "\033[38;5;40m"
    ERROR = "\033[38;5;196m"
    TRADE_BUY = "\033[38;5;40m"
    TRADE_SELL = "\033[38;5;196m"
    BORDER = "\033[38;5;240m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    COINTEGRATION_BG = "\033[48;5;22m"  # Dark green background
    VOLATILITY_FG = "\033[38;5;208m"  # Orange text
    LIQUIDATION = "\033[48;5;52m"  # Dark red background
    ARROW_UP = "\033[38;5;40m"  # Bright green
    ARROW_DOWN = "\033[38;5;196m"  # Bright red
    NEUTRAL_BG = "\033[48;5;235m"  # Dark gray background
    WARNING_BG = "\033[48;5;214m\033[30m"  # Orange background with black text


# Shortcuts
GREEN = Colors.OKGREEN
GREY = Colors.GREY
PURPLE = Colors.PURPLE
RED = Colors.FAIL
YELLOW = Colors.WARNING
CYAN = Colors.CYAN
ORANGE = Colors.ORANGE
BLUE = Colors.OKBLUE
BOLD = Colors.BOLD
UNDERLINE = Colors.UNDERLINE
ENDC = Colors.ENDC

ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# ───────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────
MAX_SINGLE_PRINT = 15000  # 15k character safety margin
ANSI_REGEX = re.compile(r"(\x1B\[[0-?]*[ -/]*[@-~])")
SAFE_TRUNCATE_OVERFLOW = 500  # Characters over limit to preserve before truncation

# ───────────────────────────────────────────────
# Thread Coordination
# ───────────────────────────────────────────────
# _print_lock = threading.Lock()
# PRINTER_READY = threading.Event()
# PRINTER_ACTIVE = threading.Event()
# PRINT_QUEUE = Queue(maxsize=5000)


# ───────────────────────────────────────────────
# Core Printing System
# ───────────────────────────────────────────────


class StructuredMessage:
    def __init__(self, content, color_code=None, is_dynamic=False):
        self.content = content
        self.color_code = color_code
        self.is_dynamic = is_dynamic

    def __str__(self):
        return self.content


class CleanFormatter(logging.Formatter):
    """Strips ANSI codes for file logging"""

    def format(self, record):
        if isinstance(record.msg, StructuredMessage):
            record.msg = str(record.msg.content)
        message = super().format(record)
        return ansi_escape.sub("", message)


def chunked(iterable, n):
    """Split an iterable into fixed-length chunks"""
    it = iter(iterable)
    return iter(lambda: list(islice(it, n)), [])


# Add cursor manager context
@contextmanager
def cursor_manager():
    """Context manager for cursor visibility control"""
    try:
        sys.stdout.write("\033[?25l")  # Hide cursor
        sys.stdout.flush()
        yield

    finally:
        sys.stdout.write("\033[?25h")  # Show cursor
        sys.stdout.flush()


# Update the ColorFormatter's message termination logic
class ColorFormatter(logging.Formatter):
    """Enhanced formatter with complete ANSI code management"""

    def format(self, record):
        original_msg = record.msg
        is_dynamic = (
            isinstance(original_msg, StructuredMessage) and original_msg.is_dynamic
        )

        # Extract color code from StructuredMessage
        color_code = ""
        if isinstance(original_msg, StructuredMessage):
            color_code = original_msg.color_code or ""
            message_content = str(original_msg.content)
        else:
            message_content = str(original_msg)

        # Build ANSI-safe message
        message = f"{color_code}{message_content}"

        # Handle dynamic updates
        if is_dynamic:
            message = f"\r\033[K{message}".rstrip("\n")
        else:
            if not message.endswith("\n"):
                message += "\n"

        # Clean up any residual ANSI codes
        message = _enforce_color_reset(message)
        return message


class QueueListener(logging.handlers.QueueListener):
    """Modified listener with overflow protection"""

    def enqueue(self, record):
        try:
            if not self.queue.full():
                super().enqueue(record)
            else:
                # Emergency flush procedure
                self.handle(record)
        except Exception:
            pass  # Prevent deadlocks


# ───────────────────────────────────────────────
# Strategy-specific logging
# ───────────────────────────────────────────────
class StrategyColors:
    """Color schemes for different strategy components"""

    ENTRY = "#00FF00"  # Bright green for entries
    EXIT = "#FF0000"  # Bright red for exits
    REBALANCE = "#FFA500"  # Orange for rebalancing
    WARNING = "#FFFF00"  # Yellow for warnings
    STATS = "#00FFFF"  # Cyan for statistics
    NEUTRAL = "#FFFFFF"  # White for neutral info
    SPREAD = "#FF69B4"  # Pink for spread data
    COINTEGRATION = "#7CFC00"  # Lawn green for cointegration
    VOLATILITY = "#FFD700"  # Gold for volatility
    ENTRY_FILL = "▉"  # Solid block for visualizations
    EXIT_FILL = "░"  # Light fill
    HISTOGRAM = "#00FF88"  # Cyan-green for spread visualizations


class TradeVisuals:
    """Preconfigured trade-related visual elements"""

    # @staticmethod
    # def long_symbol():
    #     return f"{GREEN}▲{ENDC}"

    # @staticmethod
    # def short_symbol():
    #     return f"{RED}▼{ENDC}"
    @staticmethod
    def long_symbol():
        return f"{GREEN}^{ENDC}"  # Replaced ▲ with ^^

    @staticmethod
    def short_symbol():
        return f"{RED}v{ENDC}"

    @staticmethod
    def profit_symbol():
        return f"{GREEN}✓{ENDC}"

    @staticmethod
    def loss_symbol():
        return f"{RED}✗{ENDC}"


def format_spread_analysis(spread_data: dict) -> str:
    """Format spread statistics with color coding"""

    return (
        f"{StrategyColors.SPREAD}Current Spread: {spread_data['current']:.2f}{ENDC} | "
        f"Mean: {spread_data['mean']:.2f} | "
        f"StdDev: {spread_data['stddev']:.2f} | "
        f"Z-Score: {spread_data['z_score']:.2f}"
    )


def format_cointegration_status(is_cointegrated: bool, p_value: float) -> str:
    """Format cointegration test results"""

    status_color = StrategyColors.COINTEGRATION if is_cointegrated else RED
    status_text = "COINTEGRATED" if is_cointegrated else "NOT COINTEGRATED"
    return f"{status_color}■ {status_text}{ENDC} | " f"P-Value: {p_value:.4f}"


def format_portfolio_allocation(allocations: dict) -> str:
    """Visualize portfolio weights"""

    max_width = 40
    lines = []
    for pair, weight in allocations.items():
        bar_width = int(weight * max_width)
        bar = f"{GREEN}{'■' * bar_width}{ENDC}"
        lines.append(f"{CYAN}{pair}:{ENDC} {bar} {weight:.1%}")
    return "\n".join(lines)


def format_volatility_alert(current_vol: float, historical_vol: float) -> str:
    """Highlight volatility changes"""

    vol_ratio = current_vol / historical_vol
    color = StrategyColors.VOLATILITY if vol_ratio < 2 else RED
    return (
        f"{color}VOLATILITY ALERT{ENDC} | "
        f"Current: {current_vol:.2%} | "
        f"Historical: {historical_vol:.2%} | "
        f"Ratio: {vol_ratio:.1f}x"
    )


# def format_rebalance_table(allocations: dict) -> str:
#     """Format portfolio allocation table with dynamic width and consistent colors"""
#     if not allocations:
#         return (
#             f"{YELLOW}┌────────────────┐\n│ No Allocations │\n└────────────────┘{ENDC}"
#         )

#     term_width = _get_terminal_width()
#     border_width = min(term_width, 120)  # Cap at 120 chars

#     # Calculate column width based on available space
#     num_cols = min(6, len(allocations))  # Adjust columns based on pairs
#     col_width = (
#         border_width - 4 - (num_cols - 1) * 3
#     ) // num_cols  # Account for borders and separators

#     # Build table components with green headers
#     border = f"{GREEN}╔{'═' * (border_width-2)}╗{ENDC}"
#     header = (
#         f"{GREEN}║{ENDC} {'Portfolio Rebalance'.center(border_width-4)} {GREEN}║{ENDC}"
#     )
#     separator = f"{GREEN}╠{'═' * (border_width-2)}╣{ENDC}"
#     footer = f"{GREEN}╚{'═' * (border_width-2)}╝{ENDC}"

#     # Format rows
#     rows = []
#     sorted_pairs = sorted(allocations.items(), key=lambda x: -x[1])

#     for i in range(0, len(sorted_pairs), num_cols):
#         row_items = []
#         for pair, weight in sorted_pairs[i : i + num_cols]:
#             percentage = weight * 100
#             color = GREEN if percentage >= 10 else YELLOW if percentage >= 5 else RED
#             pair_str = f"{ORANGE}{pair[0]}/{pair[1]}{ENDC}"
#             item = f"{pair_str:<{col_width-8}} {color}{percentage:>6.2f}%{ENDC}"
#             row_items.append(item)

#         # Fill remaining columns
#         while len(row_items) < num_cols:
#             row_items.append(" " * col_width)

#         rows.append(f"{GREEN}║{ENDC} {' │ '.join(row_items)} {GREEN}║{ENDC}")

#     return "\n".join([border, header, separator] + rows + [footer])


# In printing_system.py, update format_rebalance_table:
def format_rebalance_table(allocations: dict) -> str:
    """Format portfolio allocation table with dynamic width and consistent colors"""
    if not allocations:
        return (
            f"{YELLOW}┌────────────────┐\n│ No Allocations │\n└────────────────┘{ENDC}"
        )

    term_width = _get_terminal_width()
    border_width = min(term_width, 120)  # Cap at 120 chars
    num_cols = min(6, len(allocations))
    col_width = (border_width - 4 - (num_cols - 1) * 3) // num_cols

    # Helper to calculate visible length
    def visible_length(s):
        return len(ansi_escape.sub("", s))

    # Build components
    border = f"{GREEN}╔{'═'*(border_width-2)}╗{ENDC}"
    header = (
        f"{GREEN}║{ENDC} {'Portfolio Rebalance'.center(border_width-4)} {GREEN}║{ENDC}"
    )
    separator = f"{GREEN}╠{'═'*(border_width-2)}╣{ENDC}"
    footer = f"{GREEN}╚{'═'*(border_width-2)}╝{ENDC}"

    rows = []
    sorted_pairs = sorted(allocations.items(), key=lambda x: -x[1])

    for i in range(0, len(sorted_pairs), num_cols):
        row_items = []
        for pair, weight in sorted_pairs[i : i + num_cols]:
            percentage = weight * 100
            color = GREEN if percentage > 0 else RED  # Simplified threshold
            pair_str = f"{ORANGE}{pair[0]}/{pair[1]}{ENDC}"

            # Calculate padding based on visible length
            visible_len = visible_length(pair_str)
            text_space = col_width - 8  # Reserve 8 chars for percentage
            padding = max(0, text_space - visible_len)

            row_item = f"{pair_str}{' '*padding} " f"{color}{percentage:>6.2f}%{ENDC}"
            row_items.append(row_item.ljust(col_width))

        # Fill remaining columns
        while len(row_items) < num_cols:
            row_items.append(" " * col_width)

        row_line = f"{GREEN}║{ENDC} {' │ '.join(row_items)} {GREEN}║{ENDC}"
        rows.append(row_line)

    return "\n".join([border, header, separator] + rows + [footer])


def format_rebalance_summary(changes: dict) -> str:
    """Display portfolio rebalance details"""

    lines = [f"{StrategyColors.REBALANCE}REBALANCE SUMMARY{ENDC}"]
    for pair, change in changes.items():
        direction = "↑" if change > 0 else "↓"
        color = GREEN if change > 0 else RED
        lines.append(
            f"{CYAN}{pair}:{ENDC} " f"{color}{direction} {abs(change):.1%}{ENDC}"
        )
    return "\n".join(lines)


def format_trade_signal(
    pair: str,
    action: str,
    prices: tuple,  # Now expects (price1, price2)
    size: int,
    confidence: float,
) -> str:
    """Standardized trade signal formatting"""

    color = GREEN if action == "LONG" else RED
    symbol = (
        TradeVisuals.long_symbol() if action == "LONG" else TradeVisuals.short_symbol()
    )
    return (
        f"{symbol} {color}{action}{ENDC} | "
        f"{CYAN}{pair}{ENDC} | "
        f"Prices: {ORANGE}{prices[0]:.2f}/{prices[1]:.2f}{ENDC} | "
        f"Size: {BLUE}{size}{ENDC} | "
        f"Confidence: {color}{confidence:.1%}{ENDC}"
    )


def format_optimization_result(optimization_data: dict) -> str:
    """Display OU optimization parameters"""

    return (
        f"{ORANGE}OPTIMIZATION RESULT{ENDC} | "
        f"β: {optimization_data['beta']:.2f} | "
        f"μ: {optimization_data['mu']:.2f} | "
        f"σ: {optimization_data['sigma']:.2f} | "
        f"θ: {optimization_data['theta']:.2f}"
    )


def format_drawdown_warning(drawdown: float, threshold: float) -> str:
    """Highlight significant drawdowns"""

    color = RED if drawdown >= threshold else YELLOW
    return (
        f"{color}DRAWDOWN WARNING{ENDC} | "
        f"Current: {drawdown:.2%} | "
        f"Threshold: {threshold:.2%}"
    )


def format_liquidation_event(pair: str, loss: float) -> str:
    """Format forced liquidation messages"""

    return (
        f"{RED}⛔ LIQUIDATION{ENDC} | "
        f"Pair: {CYAN}{pair}{ENDC} | "
        f"Loss: {RED}{loss:.2%}{ENDC}"
    )


def format_hedge_ratio_update(pair: str, old_ratio: float, new_ratio: float) -> str:
    """Visualize hedge ratio changes"""
    
    change = new_ratio - old_ratio
    direction = "↑" if change > 0 else "↓"
    color = GREEN if change > 0 else RED
    return (
        f"{CYAN}HEDGE RATIO UPDATE{ENDC} | "
        f"{pair} | "
        f"From: {old_ratio:.2f} → {color}{new_ratio:.2f} {direction}{ENDC}"
    )


def _enforce_color_reset(content: str) -> str:
    """Ensure proper ANSI termination with stack tracking"""
    
    open_codes = []
    parts = []

    for match in ANSI_REGEX.finditer(content):
        code = match.group()
        if code == ENDC:
            if open_codes:
                open_codes.pop()
        else:
            open_codes.append(code)
        parts.append((match.start(), match.end()))

    # Rebuild string with safety
    rebuilt = []
    last_pos = 0
    for start, end in parts:
        rebuilt.append(content[last_pos:start])
        rebuilt.append(content[start:end])
        last_pos = end
    rebuilt.append(content[last_pos:])

    # Add necessary resets
    return "".join(rebuilt) + (ENDC * len(open_codes))


def _safe_truncate(content: str, max_len: int = MAX_LINE_WIDTH) -> str:
    """ANSI-aware truncation that preserves code integrity"""

    parts = []
    ansi_stack = []
    current_len = 0
    ansi_preserve = False

    for match in ANSI_REGEX.finditer(content):
        start, end = match.start(), match.end()
        code = match.group()

        # Text before code
        pre_text = content[len(parts) : start] if parts else content[:start]
        if pre_text:
            for char in pre_text:
                if current_len >= max_len - SAFE_TRUNCATE_OVERFLOW:
                    break
                parts.append(char)
                current_len += 1
            else:
                parts.append(match.group())
                ansi_stack.append(code) if code.startswith("\x1b[") else None
                continue
            break

        # Handle ANSI code
        if code == ENDC:
            if ansi_stack:
                ansi_stack.pop()
        elif code.startswith("\x1b["):
            ansi_stack.append(code)
        parts.append(code)

    # Add remaining text after last ANSI code
    remaining_text = content[match.end() :] if match else content
    for char in remaining_text:
        if current_len >= max_len:
            break
        parts.append(char)
        current_len += 1

    # Close open ANSI codes
    while ansi_stack:
        parts.append(ENDC)

    truncated = "".join(parts)
    if len(truncated) < len(content):
        truncated += f"{ENDC}...[TRUNCATED {len(content)-len(truncated):,} chars]"
    return truncated


def buffered_print(content: Union[str, StructuredMessage], log_level: str = "INFO"):
    """Atomic large message handler using proper logging"""
    
    logger = logging.getLogger()
    level = getattr(logging, log_level.upper(), logging.INFO)

    try:
        if isinstance(content, StructuredMessage):
            # Log directly without processing
            logger.log(level, content)
            return

        # Handle string content
        content_str = str(content)
        content_str = content_str[:MAX_SINGLE_PRINT]  # Remove ANSI-stripping line

        terminal_width = min(shutil.get_terminal_size().columns, 200)
        chunks = [
            content_str[i : i + terminal_width]
            for i in range(0, len(content_str), terminal_width)
        ]

        with cursor_manager():
            for chunk in chunks:
                logger.log(level, chunk)

    except Exception as e:
        error_msg = f"{RED}Print Error: {str(e)[:200]}{ENDC}"
        logger.error(error_msg)


def dynamic_print(content: str):
    """Safe dynamic printing through direct stdout"""
    
    try:
        content_str = str(content)
        content_str = _enforce_color_reset(content_str)

        # Ensure cursor management
        if not content_str.startswith("\r"):
            content_str = f"\r\033[K{content_str}"
        else:
            content_str = f"\033[K{content_str}"

        sys.stdout.write(content_str)
        sys.stdout.flush()

    except Exception as e:
        error_msg = f"{Colors.ERROR}Dynamic Print Error: {str(e)}{Colors.ENDC}"
        logging.error(error_msg)


def _get_terminal_width():
    """Get terminal width with fallback"""
    
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80


def print_header(message: str):
    """Header with dynamic width using StructuredMessage"""
    
    term_width = _get_terminal_width()
    border = f"{GREY}╔{'═' * (term_width-2)}╗{ENDC}"
    header_line = (
        f"{GREY}║{ENDC} {GREEN}{message.center(term_width-4)}{ENDC} {GREY}║{ENDC}"
    )
    footer = f"{GREY}╚{'═' * (term_width-2)}╝{ENDC}"

    # Use StructuredMessage with grey color for borders
    full_output = StructuredMessage(
        f"\n{border}\n{header_line}\n{footer}\n", color_code=GREY
    )
    logging.info(full_output)


def print_section(title: str, color: str = GREEN):
    """Section header with proper color wrapping"""
    
    term_width = _get_terminal_width()
    section = (
        f"{color}┌{'─'*(term_width-2)}┐\n"
        f"│ {title.center(term_width-4)} │\n"
        f"└{'─'*(term_width-2)}┘"
    )
    logging.info(StructuredMessage(section, color))


# def print_centered(title: str, color: str = GREEN):
#     """Section header with proper color wrapping"""
#     term_width = _get_terminal_width()
#     section = f"{title.center(term_width-4)}\n"
#     logging.info(StructuredMessage(section, color))

def print_centered(title: str, color: str = GREEN):
    """Section header with proper color wrapping"""

    term_width = _get_terminal_width()
    # Ensure the title doesn't exceed terminal width
    title = title[:term_width]
    # Center the text and pad with spaces
    centered = title.center(term_width)
    # Apply color and print
    logging.info(StructuredMessage(f"{color}{centered}{ENDC}", color))


# def print_boxed(text: str, color: str = CYAN, width: int = None):
#     """Large text box handler with consistent borders"""
#     if width is None:
#         width = min(MAX_LINE_WIDTH, shutil.get_terminal_size().columns - 4)

#     if len(text) > MAX_SINGLE_PRINT:
#         text = _safe_truncate(text, MAX_SINGLE_PRINT)

#     # Smart word wrapping
#     wrapper = textwrap.TextWrapper(
#         width=width - 4,  # Account for borders and padding
#         break_long_words=False,
#         replace_whitespace=False,
#     )

#     lines = []
#     for line in text.split("\n"):
#         lines.extend(wrapper.wrap(line))

#     # Chunked processing
#     MAX_BOX_LINES = 50  # Prevent terminal flooding
#     for line_chunk in chunked(lines, MAX_BOX_LINES):
#         box_lines = [
#             f"{color}│{ENDC} {line.ljust(width-4)} {color}│{ENDC}"
#             for line in line_chunk
#         ]
#         full_box = (
#             f"{color}┌{'─'*(width-2)}┐{ENDC}\n"
#             + "\n".join(box_lines)
#             + f"\n{color}└{'─'*(width-2)}┘{ENDC}"
#         )
#         buffered_print(full_box)


def print_boxed(text: str, color: str = CYAN, width: int = None):
    """Boxed text with proper color handling"""
    wrapper = textwrap.TextWrapper(width=MAX_LINE_WIDTH - 4)
    lines = wrapper.wrap(text)
    box_top = StructuredMessage(f"{color}┌{'─'*(MAX_LINE_WIDTH-2)}┐", color)
    box_bottom = StructuredMessage(f"{color}└{'─'*(MAX_LINE_WIDTH-2)}┘", color)

    buffered_print(box_top)
    for line in lines:
        buffered_print(
            StructuredMessage(
                f"{color}│ {line.ljust(MAX_LINE_WIDTH-4)} {color}│", color
            )
        )
    buffered_print(box_bottom)


# def print_status(message: str, status: str = "INFO"):
#     """Status messages with automatic color cleanup"""

#     color_map = {
#         "INFO": Colors.OKBLUE,
#         "SUCCESS": Colors.OKGREEN,
#         "WARNING": Colors.WARNING,
#         "ERROR": Colors.FAIL,
#         "CACHE": Colors.LABEL,
#         "CONFIG": Colors.CYAN,
#     }
#     color = color_map.get(status.upper(), ENDC)
#     timestamp = datetime.now().strftime("%H:%M:%S")
#     output = f"{color}[{timestamp}] [{status.ljust(7)}] {message}"
#     logging.info(StructuredMessage(output, color))


def print_status(message: str, status: str = "INFO"):
    """Status messages with structured formatting"""
    color_map = {
        "INFO": BLUE,
        "SUCCESS": GREEN,
        "WARNING": YELLOW,
        "ERROR": RED,
    }
    color = color_map.get(status.upper(), ENDC)
    formatted = f"[{status}] {message}"
    logging.info(StructuredMessage(formatted, color))


def print_trade(
    trade_type: str,
    pair: tuple,
    date: datetime,
    prices: tuple,
    z_score: float,
    tcs: float,
    size: float,
    weight: float = 0.0,
):
    """Print trade information with dynamic width and color-coded components"""
    term_width = _get_terminal_width()
    is_long = trade_type == "LONG"

    # Color coding
    direction_color = GREEN if is_long else RED
    opposite_color = RED if is_long else GREEN
    grey = "\033[38;5;240m"  # Grey color for metrics

    # Symbols
    symbol = "▲" if is_long else "▼"

    # Format components with colors
    type_part = f"{direction_color}{symbol} {trade_type.ljust(5)}{ENDC}"
    date_part = f"{date.strftime('%Y-%m-%d')}"
    pair_part = f"{ORANGE}{pair[0]}/{pair[1]}{ENDC}"

    # Price formatting with opposite colors for each leg
    if is_long:
        price_part = f"Price: {GREEN}{prices[0]:.2f}{ENDC}/{RED}{prices[1]:.2f}{ENDC}"
    else:
        price_part = f"Price: {RED}{prices[0]:.2f}{ENDC}/{GREEN}{prices[1]:.2f}{ENDC}"

    # Metrics in grey
    z_part = f"Z: {z_score:+.2f}"
    size_part = f"{grey}Size: {size}{ENDC}"
    weight_part = f"{grey}Weight: {weight:.2%}{ENDC}"
    tcs_part = f"{grey}TCS: {tcs:.1%}{ENDC}"

    # Build output with proper spacing
    parts = [
        type_part,
        date_part,
        pair_part,
        price_part,
        z_part,
        size_part,
        weight_part,
        tcs_part,
    ]

    output = " │ ".join(parts)
    logging.info(StructuredMessage(output))

    # Add separator
    print_centered(f"{YELLOW}{'─'*(term_width-2)}{ENDC}", YELLOW)


def print_trade_exit(
    pair: tuple, date: datetime, pnl: float, duration: int, reason: str
):
    """Print trade exit with dynamic width and color-coded components"""
    term_width = _get_terminal_width()
    color = GREEN if pnl >= 0 else RED
    grey = "\033[38;5;240m"  # Grey color for metrics
    symbol = "✅" if pnl >= 0 else "❌"

    # Format components with colors
    pair_part = f"{symbol} {ORANGE}{pair[0]}/{pair[1]}{ENDC}"
    date_part = f"Date: {date.strftime('%Y-%m-%d')}"
    pnl_part = f"PnL: {color}{pnl:+.2f}{ENDC}"
    duration_part = f"{grey}Duration: {duration}d{ENDC}"

    # Truncate reason if needed
    max_reason_len = (
        term_width
        - len(ansi_escape.sub("", pair_part))
        - len(date_part)
        - len(str(pnl))
        - len(str(duration))
        - 30
    )
    reason_truncated = (
        reason
        if len(reason) <= max_reason_len
        else reason[: max_reason_len - 3] + "..."
    )
    reason_part = f"{grey}Reason: {reason_truncated}{ENDC}"

    # Build output
    parts = [pair_part, date_part, pnl_part, duration_part, reason_part]

    output = " │ ".join(parts)
    logging.info(StructuredMessage(output))

    # Add separator
    print_centered(f"{YELLOW}{'─'*(term_width-2)}{ENDC}")


# def flush_queue():
#     """Atomic queue flushing"""

#     with _print_lock:
#         items = []
#         while not PRINT_QUEUE.empty():
#             try:
#                 items.append(PRINT_QUEUE.get_nowait())
#             except Empty:
#                 break
#         if items:
#             sys.stdout.write("\n".join(items) + "\n")  # Atomic write
#             sys.stdout.flush()


def shutdown_printing():
    """Graceful shutdown of logging system"""
    global queue_listener
    if queue_listener:
        queue_listener.stop()
    logging.shutdown()
    sys.stdout.write("\033[0m\033[?25h\n")  # Reset colors, show cursor
    sys.stdout.flush()

    # def shutdown_printing():
    # global PRINTER_ACTIVE
    # PRINTER_ACTIVE.clear()
    # with _print_lock:  # Wait for printer to finish
    # sys.stdout.write("\n")
    # sys.stdout.flush()


# ───────────────────────────────────────────────
# Thread Initialization
# ───────────────────────────────────────────────
# print_thread = threading.Thread(
#     target=_printer_daemon, daemon=True, name="PrinterSystem"
# )
# PRINTER_ACTIVE.set()
# print_thread.start()

setup_logging()

# Configure formatters after setup
# root_logger = logging.getLogger()
# for handler in root_logger.handlers:
#     if isinstance(handler, logging.StreamHandler):
#         handler.setFormatter(
#             ColorFormatter("%(asctime)s [%(threadName)s] %(levelname)s %(message)s")
#         )
