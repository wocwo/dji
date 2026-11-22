"""长期记忆：SQLite 存储运行历史、对话轮次与笔记。

Agent 的「记忆」分三层：
- 工作记忆：运行时的 messages（在 runtime 中维护，不落库）；
- 情节记忆：每次运行的 task/answer/steps/citations（episodes 表）；
- 语义记忆：用户偏好、结论、待办（notes 表）。

SQLite 足够轻量，单文件即可随项目分发；写入失败时降级为「无记忆」，
不影响主流程。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .knowledge import tokenize

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class AgentMemory:
    """基于 SQLite 的 Agent 长期记忆。"""

    def __init__(self, db_path: str = ".agent/memory.db", *, enabled: bool = True):
        self.db_path = str(db_path)
        self.enabled = enabled
        self.available = False
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        if not enabled:
            return
        try:
            path = Path(self.db_path).expanduser()
            if path.parent and str(path.parent) not in ("", "."):
                path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._init_schema()
                self._conn.commit()
            self.available = True
        except Exception as exc:  # 磁盘只读 / 路径非法等
            logger.warning("长期记忆初始化失败，将降级为无记忆: %s", exc)
            self.available = False
            self._conn = None

    # -- schema -----------------------------------------------------------
    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                answer TEXT,
                status TEXT,
                steps INTEGER,
                citations TEXT,
                usage TEXT,
                trace_id TEXT,
                session_id TEXT,
                created_at TEXT,
                duration_ms REAL,
                metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at DESC);

            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                tags TEXT,
                session_id TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at DESC);

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);

            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                last_turn_id INTEGER,
                updated_at TEXT
            );
            """
        )

    # -- 情节记忆 ---------------------------------------------------------
    def save_episode(self, result: Any, *, session_id: Optional[str] = None) -> Optional[str]:
        if not self.available or self._conn is None:
            return None
        episode_id = f"ep_{result.trace_id or ''}_{int(datetime.now().timestamp() * 1000)}"
        citations = [c.to_dict() if hasattr(c, "to_dict") else c for c in result.citations]
        usage = result.usage.to_dict() if hasattr(result.usage, "to_dict") else result.usage
        metadata = dict(result.metadata or {})
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO episodes "
                    "(id, task, answer, status, steps, citations, usage, trace_id, session_id, created_at, duration_ms, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        episode_id,
                        result.task,
                        result.answer,
                        result.status,
                        len(result.steps),
                        json.dumps(citations, ensure_ascii=False),
                        json.dumps(usage, ensure_ascii=False),
                        result.trace_id,
                        session_id,
                        _now(),
                        float(result.duration_ms),
                        json.dumps(metadata, ensure_ascii=False, default=str),
                    ),
                )
                self._conn.commit()
            return episode_id
        except Exception as exc:
            logger.warning("保存运行记录失败: %s", exc)
            return None

    def recent_episodes(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.available or self._conn is None:
            return []
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (max(1, limit),)
                ).fetchall()
            return [self._row_to_episode(row) for row in rows]
        except Exception as exc:
            logger.warning("读取运行记录失败: %s", exc)
            return []

    def search_episodes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """按词重叠对最近 500 条运行记录排序。"""
        episodes = self.recent_episodes(500)
        if not episodes:
            return []
        q_tokens = set(tokenize(query or ""))
        if not q_tokens:
            return episodes[:limit]
        scored = []
        for episode in episodes:
            tokens = set(tokenize(f"{episode.get('task', '')} {episode.get('answer', '')}"))
            if not tokens:
                continue
            overlap = len(q_tokens & tokens)
            if overlap:
                scored.append((overlap, episode))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [episode for _, episode in scored[: max(1, limit)]]

    # -- 笔记 -------------------------------------------------------------
    def add_note(
        self,
        text: str,
        tags: Optional[Sequence[str]] = None,
        *,
        session_id: Optional[str] = None,
    ) -> Optional[int]:
        if not self.available or self._conn is None:
            return None
        text = str(text).strip()
        if not text:
            return None
        clean_tags = self._clean_tags(tags)
        try:
            with self._lock:
                cursor = self._conn.execute(
                    "INSERT INTO notes (text, tags, session_id, created_at) VALUES (?, ?, ?, ?)",
                    (text, json.dumps(clean_tags, ensure_ascii=False), session_id, _now()),
                )
                self._conn.commit()
                return int(cursor.lastrowid)
        except Exception as exc:
            logger.warning("保存笔记失败: %s", exc)
            return None

    def search_notes(
        self,
        query: str = "",
        tags: Optional[Sequence[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.available or self._conn is None:
            return []
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM notes ORDER BY id DESC LIMIT 500"
                ).fetchall()
        except Exception as exc:
            logger.warning("读取笔记失败: %s", exc)
            return []

        tag_filter = {t.lower() for t in (tags or []) if t}
        q_tokens = set(tokenize(query or ""))
        scored: List[tuple] = []
        for row in rows:
            note_tags = self._parse_tags(row["tags"])
            if tag_filter and not tag_filter.intersection({t.lower() for t in note_tags}):
                continue
            score = 0
            if q_tokens:
                note_tokens = set(tokenize(row["text"]))
                score = len(q_tokens & note_tokens)
                if score == 0 and not tag_filter:
                    continue
            scored.append((score, self._row_to_note(row)))
        scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
        return [note for _, note in scored[: max(1, limit)]]

    # -- 对话轮次 ---------------------------------------------------------
    def append_turn(self, session_id: str, role: str, content: str) -> None:
        if not self.available or self._conn is None or not content:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (session_id, role, content, _now()),
                )
                self._conn.commit()
        except Exception as exc:
            logger.warning("保存对话轮次失败: %s", exc)

    def get_turns(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.available or self._conn is None:
            return []
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, role, content, created_at FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, max(1, limit)),
                ).fetchall()
            return [dict(row) for row in reversed(rows)]
        except Exception as exc:
            logger.warning("读取对话轮次失败: %s", exc)
            return []

    def get_summary(self, session_id: str) -> str:
        if not self.available or self._conn is None:
            return ""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT summary FROM session_summaries WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            return str(row["summary"]) if row else ""
        except Exception as exc:
            logger.warning("读取会话摘要失败: %s", exc)
            return ""

    def get_summary_last_turn_id(self, session_id: str) -> int:
        if not self.available or self._conn is None:
            return 0
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT last_turn_id FROM session_summaries WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            return int(row["last_turn_id"] or 0) if row else 0
        except Exception:
            return 0

    def save_summary(self, session_id: str, summary: str, last_turn_id: int = 0) -> None:
        if not self.available or self._conn is None or not summary:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO session_summaries (session_id, summary, last_turn_id, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                    "summary=excluded.summary, last_turn_id=excluded.last_turn_id, updated_at=excluded.updated_at",
                    (session_id, summary, int(last_turn_id), _now()),
                )
                self._conn.commit()
        except Exception as exc:
            logger.warning("保存会话摘要失败: %s", exc)

    def search_turns(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 4,
        exclude_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """按词面重叠检索同一会话的历史轮次。"""
        if not self.available or self._conn is None:
            return []
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, role, content, created_at FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT 500",
                    (session_id,),
                ).fetchall()
        except Exception as exc:
            logger.warning("检索历史轮次失败: %s", exc)
            return []
        q_tokens = set(tokenize(query or ""))
        if not q_tokens:
            return []
        scored = []
        excluded = exclude_ids or set()
        for row in rows:
            if row["id"] in excluded:
                continue
            tokens = set(tokenize(row["content"] or ""))
            overlap = len(q_tokens & tokens)
            if overlap:
                scored.append((overlap / max(1, len(q_tokens)), row))
        scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
        return [dict(row) for _, row in scored[: max(1, limit)]]

    def count_turns(self, session_id: str) -> int:
        if not self.available or self._conn is None:
            return 0
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def list_sessions(self, limit: int = 20, prefix: str = "") -> List[Dict[str, Any]]:
        if not self.available or self._conn is None:
            return []
        try:
            where = "WHERE t.session_id LIKE ?" if prefix else ""
            params: List[Any] = [f"{prefix}%"] if prefix else []
            params.append(max(1, limit))
            with self._lock:
                rows = self._conn.execute(
                    f"""
                    SELECT t.session_id,
                           COUNT(*) AS turns,
                           MAX(t.created_at) AS last_at,
                           (SELECT content FROM turns x
                             WHERE x.session_id = t.session_id AND x.role = 'user'
                             ORDER BY x.id DESC LIMIT 1) AS preview
                      FROM turns t
                      {where}
                     GROUP BY t.session_id
                     ORDER BY last_at DESC
                     LIMIT ?
                    """,
                    params,
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("读取会话列表失败: %s", exc)
            return []

    # -- 统计与维护 -------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        if not self.available or self._conn is None:
            return {"available": False, "db_path": self.db_path}
        try:
            with self._lock:
                episodes = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                notes = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
                turns = self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            return {
                "available": True,
                "db_path": self.db_path,
                "episodes": episodes,
                "notes": notes,
                "turns": turns,
            }
        except Exception as exc:
            return {"available": False, "db_path": self.db_path, "error": str(exc)}

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self.available = False

    # -- 辅助 -------------------------------------------------------------
    @staticmethod
    def _clean_tags(tags: Optional[Sequence[str]]) -> List[str]:
        if not tags:
            return []
        if isinstance(tags, str):
            tags = [x.strip() for x in tags.split(",") if x.strip()]
        clean: List[str] = []
        for tag in tags:
            value = str(tag).strip()
            if value and value not in clean:
                clean.append(value)
        return clean[:10]

    @staticmethod
    def _parse_tags(raw: Any) -> List[str]:
        if not raw:
            return []
        try:
            value = json.loads(raw)
            return [str(x) for x in value] if isinstance(value, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Dict[str, Any]:
        def load(raw: Any, default: Any) -> Any:
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        return {
            "id": row["id"],
            "task": row["task"],
            "answer": row["answer"],
            "status": row["status"],
            "steps": row["steps"],
            "citations": load(row["citations"], []),
            "usage": load(row["usage"], {}),
            "trace_id": row["trace_id"],
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "duration_ms": row["duration_ms"],
            "metadata": load(row["metadata"], {}),
        }

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "text": row["text"],
            "tags": AgentMemory._parse_tags(row["tags"]),
            "created_at": row["created_at"],
        }


__all__ = ["AgentMemory"]
