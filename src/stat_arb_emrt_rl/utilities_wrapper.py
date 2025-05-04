# from .printing_system import (
#     Colors,
#     buffered_print,
#     print_header as _print_header,
#     print_section,
#     print_boxed,
# )

# def _print_header(message: str):
#     """Uniform header formatting"""
#     print(f"\n{'#' * 80}")
#     print(f"# {message.center(76)} #")
#     print(f"{'#' * 80}\n")


# def _print_status(message: str, status: str = "INFO"):
#     """Standardized status messages"""
#     timestamp = datetime.now().strftime("%H:%M:%S")
#     print(f"[{timestamp}] [{status.ljust(5)}] {message}")


# def _print_status(message: str, status: str = "INFO"):
#     """Bloomberg-style status updates"""
#     status_color = {
#         "INFO": Colors.LABEL,
#         "WARN": Colors.WARNING,
#         "ERROR": Colors.ERROR,
#         "SUCCESS": Colors.VALUE,
#     }.get(status.upper(), Colors.ENDC)

#     formatted = (
#         f"{Colors.BORDER}│{Colors.ENDC} "
#         f"{status_color}{status[:4]:<4}{Colors.ENDC} "
#         f"{Colors.BORDER}│{Colors.ENDC} "
#         f"{message}"
#     )
#     print(formatted)
