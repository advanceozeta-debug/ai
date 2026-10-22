"""오미크론의 저장소: 장기 기억과 대화 기록.

DATABASE_URL 이 있으면 PostgreSQL(예: Neon)에, 없으면 data/ 폴더의 JSON 파일에 저장합니다.
대화 기록은 사람이 읽는 글(질문과 답)만 저장하고, 다시 열 때 그 글로 대화를 이어갑니다.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .memory import Memory


class FileStore(Memory):
    """data/memory.json + data/conversations/<id>.json"""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir / "memory.json")
        self.conv_dir = data_dir / "conversations"

    def new_conversation(self, persona: str) -> str:
        conv_id = uuid.uuid4().hex
        self._write(conv_id, {"persona": persona, "created": _now(), "messages": []})
        return conv_id

    def latest_conversation(self, persona: str) -> str | None:
        if not self.conv_dir.exists():
            return None
        convs = [(json.loads(p.read_text(encoding="utf-8")), p.stem) for p in self.conv_dir.glob("*.json")]
        convs = [(c["created"], cid) for c, cid in convs if c["persona"] == persona]
        return max(convs)[1] if convs else None

    def load_messages(self, conv_id: str) -> list[dict]:
        path = self.conv_dir / f"{conv_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))["messages"] if path.exists() else []

    def append_messages(self, conv_id: str, messages: list[dict]) -> None:
        path = self.conv_dir / f"{conv_id}.json"
        conv = json.loads(path.read_text(encoding="utf-8"))
        conv["messages"].extend(messages)
        self._write(conv_id, conv)

    def _write(self, conv_id: str, conv: dict) -> None:
        self.conv_dir.mkdir(parents=True, exist_ok=True)
        (self.conv_dir / f"{conv_id}.json").write_text(json.dumps(conv, ensure_ascii=False), encoding="utf-8")


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id BIGSERIAL PRIMARY KEY,
    fact TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    persona TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, id);
CREATE INDEX IF NOT EXISTS conversations_persona_idx ON conversations (persona, created_at DESC);
"""


class PostgresStore:
    """같은 기능을 PostgreSQL 에 저장합니다. 요청이 적으니 호출마다 연결합니다."""

    def __init__(self, url: str):
        import psycopg

        self._connect = lambda: psycopg.connect(url, autocommit=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @property
    def facts(self) -> list[str]:
        with self._connect() as conn:
            return [r[0] for r in conn.execute("SELECT fact FROM memories ORDER BY id")]

    def add(self, fact: str) -> str:
        fact = fact.strip()
        if not fact:
            return "저장할 내용이 비어 있습니다."
        with self._connect() as conn:
            cur = conn.execute("INSERT INTO memories (fact) VALUES (%s) ON CONFLICT (fact) DO NOTHING", (fact,))
        return f"기억했습니다: {fact}" if cur.rowcount else "이미 기억하고 있는 내용입니다."

    def forget(self, keyword: str) -> str:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM memories WHERE strpos(fact, %s) > 0", (keyword,))
        return f"{cur.rowcount}개의 기억을 지웠습니다."

    def as_text(self) -> str:
        facts = self.facts
        return "\n".join(f"- {f}" for f in facts) if facts else "(아직 기억하는 내용 없음)"

    def new_conversation(self, persona: str) -> str:
        conv_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute("INSERT INTO conversations (id, persona) VALUES (%s, %s)", (conv_id, persona))
        return conv_id

    def latest_conversation(self, persona: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE persona = %s ORDER BY created_at DESC LIMIT 1", (persona,)
            ).fetchone()
        return row[0] if row else None

    def load_messages(self, conv_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT role, text FROM messages WHERE conversation_id = %s ORDER BY id", (conv_id,))
            return [{"role": role, "text": text} for role, text in rows]

    def append_messages(self, conv_id: str, messages: list[dict]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO messages (conversation_id, role, text) VALUES (%s, %s, %s)",
                    [(conv_id, m["role"], m["text"]) for m in messages],
                )


def open_store(data_dir: Path, database_url: str | None):
    return PostgresStore(database_url) if database_url else FileStore(data_dir)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
