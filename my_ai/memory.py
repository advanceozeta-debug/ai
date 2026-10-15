"""대화가 끝나도 남는 장기 기억. JSON 파일에 사실 목록으로 저장합니다."""

import json
from pathlib import Path


class Memory:
    def __init__(self, path: Path):
        self.path = path
        self.facts: list[str] = []
        if path.exists():
            self.facts = json.loads(path.read_text(encoding="utf-8"))

    def add(self, fact: str) -> str:
        fact = fact.strip()
        if not fact:
            return "저장할 내용이 비어 있습니다."
        if fact in self.facts:
            return "이미 기억하고 있는 내용입니다."
        self.facts.append(fact)
        self._save()
        return f"기억했습니다: {fact}"

    def forget(self, keyword: str) -> str:
        before = len(self.facts)
        self.facts = [f for f in self.facts if keyword not in f]
        removed = before - len(self.facts)
        self._save()
        return f"{removed}개의 기억을 지웠습니다."

    def as_text(self) -> str:
        if not self.facts:
            return "(아직 기억하는 내용 없음)"
        return "\n".join(f"- {f}" for f in self.facts)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.facts, ensure_ascii=False, indent=2), encoding="utf-8")
