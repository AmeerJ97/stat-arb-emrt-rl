import subprocess
import sys


def test_cli_help_exposes_reboot_workflows():
    completed = subprocess.run(
        [sys.executable, "-m", "stat_arb_emrt_rl.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "paper-pairs" in completed.stdout
    assert "discover" in completed.stdout
    assert "backtest" in completed.stdout
    assert "streamlit" in completed.stdout


def test_streamlit_command_accepts_native_streamlit_flags():
    from stat_arb_emrt_rl.cli import build_parser

    args, unknown = build_parser().parse_known_args(
        ["streamlit", "--server.headless=true", "--server.port=8502"]
    )

    assert args.command == "streamlit"
    assert unknown == ["--server.headless=true", "--server.port=8502"]
