"""SQL database query tool with a read-only safety mode."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from src.tools.base import Parameter, Tool, ToolError

_ALLOWED_READ_KEYWORDS = {"SELECT", "WITH", "EXPLAIN", "PRAGMA"}
_MAX_FETCH_ROWS = 1000
_Param = Union[str, int, float, None]


class DatabaseTool(Tool):
    """Execute SQL against a SQLite database file.

    The ``engine`` parameter reserves room for future drivers (e.g. Postgres
    via psycopg) while SQLite keeps the default install dependency-free.
    """

    name = "database"
    description = "Run a SQL statement against the configured database and return rows or counts."
    parameters = [
        Parameter(name="query", type="string", description="SQL statement to execute"),
        Parameter(name="params", type="array", description="Positional bind parameters", required=False, default=[]),
        Parameter(
            name="fetch",
            type="string",
            description="Result expectation",
            required=False,
            default="auto",
            enum=["auto", "rows", "none"],
        ),
    ]

    def __init__(
        self,
        path: str | Path = "agentmesh.db",
        read_only: bool = False,
        engine: str = "sqlite",
    ) -> None:
        self.path = str(path)
        self.read_only = read_only
        if engine != "sqlite":
            raise ToolError(f"engine '{engine}' is not supported yet; use 'sqlite'")
        self._connection: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    @staticmethod
    def _leading_keyword(query: str) -> str:
        for token in query.strip().split():
            return token.upper().strip("(")
        return ""

    def _guard(self, query: str) -> None:
        keyword = self._leading_keyword(query)
        if self.read_only and keyword not in _ALLOWED_READ_KEYWORDS:
            raise ToolError(
                f"read_only mode permits only {_ALLOWED_READ_KEYWORDS}; got '{keyword or '(empty)'}'"
            )

    async def _run(
        self,
        query: str,
        params: Sequence[_Param] | None = None,
        fetch: str = "auto",
    ) -> Dict[str, Any]:
        self._guard(query)
        connection = self._get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(query, tuple(params or []))
        except sqlite3.Error as exc:
            connection.rollback()
            raise ToolError(f"SQL error: {exc}") from exc
        returns_rows = cursor.description is not None and fetch != "none"
        if returns_rows:
            rows: List[Dict[str, Any]] = []
            for row in cursor.fetchmany(_MAX_FETCH_ROWS):
                rows.append({key: row[key] for key in row.keys()})
            truncated = len(rows) == _MAX_FETCH_ROWS
            result = {
                "columns": [description[0] for description in cursor.description] if cursor.description else [],
                "rows": rows,
                "rowcount": len(rows),
                "truncated": truncated,
            }
        else:
            connection.commit()
            result = {
                "rowcount": max(0, cursor.rowcount),
                "lastrowid": cursor.lastrowid,
                "committed": True,
            }
        cursor.close()
        return result

    def close(self) -> None:
        """Close the underlying connection if one was opened."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "DatabaseTool":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()
