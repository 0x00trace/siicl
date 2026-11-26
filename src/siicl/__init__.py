"""SIICL: research tools toward a Switch 2 compatibility layer for Linux."""

__version__ = "0.1.0"


def main() -> int:
    from siicl.cli import main as cli_main

    return cli_main()
