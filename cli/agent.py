"""Agent lifecycle commands: create, list, run, remove."""
from __future__ import annotations

import asyncio
import datetime as _dt
import pathlib
from typing import Any, Dict

import click
import yaml

from src.providers.base import MockProvider, ProviderError

AGENTS_DIR = pathlib.Path.home() / ".agentmesh" / "agents"

ROLE_MAP = {
    "chat": "src.agents.conversational:ChatAgent",
    "coder": "src.agents.coding:CodingAgent",
    "researcher": "src.agents.research:ResearchAgent",
    "planner": "src.agents.planner:PlannerAgent",
    "critic": "src.agents.critic:CriticAgent",
}


@click.group()
def agent() -> None:
    """Create, inspect, and run AgentMesh agents."""


def _agent_path(name: str) -> pathlib.Path:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_")
    if not safe:
        raise click.ClickException("agent name must contain alphanumeric characters")
    return AGENTS_DIR / f"{safe}.yaml"


@agent.command("create")
@click.option("--name", required=True, help="Unique agent name")
@click.option("--role", default="chat", show_default=True, type=click.Choice(sorted(ROLE_MAP)))
@click.option("--provider", default="mock", show_default=True, help="mock | openai | anthropic | ollama")
def create(name: str, role: str, provider: str) -> None:
    """Persist a new agent definition under ~/.agentmesh/agents."""
    path = _agent_path(name)
    if path.exists():
        raise click.ClickException(f"agent '{name}' already exists at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "role": role,
        "provider": provider,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, default_flow_style=False)
    click.echo(f"created agent '{name}' ({role}/{provider}) at {path}")


@agent.command("list")
def listing() -> None:
    """Show all saved agent definitions."""
    if not AGENTS_DIR.exists():
        click.echo("(no agents defined yet; try 'agentmesh agent create')")
        return
    for path in sorted(AGENTS_DIR.glob("*.yaml")):
        record = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        click.echo(
            f"{record.get('name', path.stem):<16} {record.get('role', '?'):<12} "
            f"{record.get('provider', '?'):<10} {record.get('created_at', '')}"
        )


@agent.command("remove")
@click.argument("name")
def remove(name: str) -> None:
    """Delete a saved agent definition."""
    path = _agent_path(name)
    if not path.exists():
        raise click.ClickException(f"no agent named '{name}'")
    path.unlink()
    click.echo(f"removed '{name}'")


def _build_provider(provider_name: str) -> Any:
    if provider_name == "openai":
        from src.providers.openai import OpenAIProvider

        return OpenAIProvider()
    if provider_name == "anthropic":
        from src.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if provider_name == "ollama":
        from src.providers.ollama import OllamaProvider

        return OllamaProvider()
    return MockProvider()


async def _run_agent(record: Dict[str, Any], prompt: str) -> str:
    provider = _build_provider(str(record.get("provider", "mock")))
    from src.agents.conversational import ChatAgent
    from src.protocols.message import Message

    instance = ChatAgent(str(record.get("name", "cli-agent")), provider=provider)
    await instance.initialize()
    reply = await instance.deliver(
        Message(sender="cli", recipient=instance.agent_id, content=prompt)
    )
    await instance.stop()
    return str(reply.content) if reply else "(no reply)"


@agent.command("run")
@click.option("--name", required=True, help="Saved agent name to run")
@click.option("--prompt", required=True, help="Text to send to the agent")
def run(name: str, prompt: str) -> None:
    """Send one prompt to an agent and print the response."""
    path = _agent_path(name)
    if not path.exists():
        raise click.ClickException(f"no agent named '{name}' (create it first)")
    record = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        output = asyncio.run(_run_agent(record, prompt))
    except ProviderError as exc:
        raise click.ClickException(f"provider error: {exc}") from exc
    click.echo(output)
