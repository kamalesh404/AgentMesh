"""Session commands: start, send, resume, and inspect conversation history."""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import uuid
from typing import Any, Dict, List

import click

SESSIONS_FILE = pathlib.Path.home() / ".agentmesh" / "sessions.json"


def _load_store() -> Dict[str, Any]:
    if not SESSIONS_FILE.exists():
        return {"sessions": {}}
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"corrupt session store at {SESSIONS_FILE}") from exc
    if not isinstance(data.get("sessions"), dict):
        raise click.ClickException(f"invalid session store at {SESSIONS_FILE}")
    return data


def _save_store(store: Dict[str, Any]) -> None:
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _require_session(store: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    record = store["sessions"].get(session_id)
    if record is None:
        raise click.ClickException(f"unknown session '{session_id}'")
    return record


@click.group()
def session() -> None:
    """Persist and replay agent sessions."""


@session.command("start")
@click.option("--name", default="session", show_default=True, help="Human-readable label")
def start(name: str) -> None:
    """Open a new persisted session."""
    store = _load_store()
    session_id = uuid.uuid4().hex[:12]
    store["sessions"][session_id] = {
        "name": name,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "messages": [],
    }
    _save_store(store)
    click.echo(f"started session {session_id} ({name})")


@session.command("send")
@click.argument("session_id")
@click.argument("text", nargs=-1, required=True)
def send(session_id: str, text: tuple) -> None:
    """Append a user message plus an offline assistant echo to a session."""
    store = _load_store()
    record = _require_session(store, session_id)
    user_text = " ".join(text)
    reply = f"[offline assistant] acknowledged: {user_text[:120]}"
    messages: List[Dict[str, str]] = record.setdefault("messages", [])
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": reply})
    _save_store(store)
    click.echo(reply)


@session.command("resume")
@click.argument("session_id")
def resume(session_id: str) -> None:
    """Print the full transcript of a stored session."""
    store = _load_store()
    record = _require_session(store, session_id)
    click.echo(f"session {session_id} - {record.get('name', '')}")
    messages: List[Dict[str, str]] = record.get("messages", [])
    if not messages:
        click.echo("(empty transcript)")
        return
    for message in messages:
        speaker = message.get("role", "?").upper()
        click.echo(f"{speaker}: {message.get('content', '')}")


@session.command("history")
def history() -> None:
    """List every stored session with its message count."""
    store = _load_store()
    sessions: Dict[str, Any] = store.get("sessions", {})
    if not sessions:
        click.echo("(no sessions recorded)")
        return
    for session_id in sorted(sessions):
        record = sessions[session_id]
        count = len(record.get("messages", []))
        created = record.get("created_at", "")[:19].replace("T", " ")
        click.echo(f"{session_id}  {record.get('name', '?'):<20} {count:>3} msgs  {created}")


@session.command("drop")
@click.argument("session_id")
def drop(session_id: str) -> None:
    """Delete a stored session."""
    store = _load_store()
    _require_session(store, session_id)
    del store["sessions"][session_id]
    _save_store(store)
    click.echo(f"dropped session {session_id}")
