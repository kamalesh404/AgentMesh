"""Entry point for the AgentMesh CLI."""
from __future__ import annotations

import importlib.metadata
import pathlib
import sys

import click

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logging import configure_logging  # noqa: E402


def _package_version() -> str:
    try:
        from src import __version__

        return __version__
    except Exception:
        try:
            return importlib.metadata.version("agentmesh")
        except importlib.metadata.PackageNotFoundError:
            return "0.0.0+dev"


@click.group()
@click.version_option(version=_package_version(), prog_name="agentmesh")
def main() -> None:
    """AgentMesh - build, connect, and orchestrate LLM agents."""


def register_commands() -> None:
    from cli.agent import agent
    from cli.session import session

    main.add_command(agent)
    main.add_command(session)


register_commands()


if __name__ == "__main__":
    configure_logging()
    main()
