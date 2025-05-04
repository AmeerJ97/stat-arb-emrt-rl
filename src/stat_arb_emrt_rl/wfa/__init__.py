__all__ = [
    "WalkForwardEngine",
    "WFAConfig",
    "WindowResult",
    "WFAReport",
    "WFAReporter",
]


def __getattr__(name):
    if name in {"WalkForwardEngine", "WFAConfig", "WindowResult", "WFAReport"}:
        from .wfa_engine import WalkForwardEngine, WFAConfig, WindowResult, WFAReport

        return {
            "WalkForwardEngine": WalkForwardEngine,
            "WFAConfig": WFAConfig,
            "WindowResult": WindowResult,
            "WFAReport": WFAReport,
        }[name]
    if name == "WFAReporter":
        from .wfa_report import WFAReporter

        return WFAReporter
    raise AttributeError(name)
