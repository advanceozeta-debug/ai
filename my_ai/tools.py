"""AI가 스스로 쓸 수 있는 도구들. 새 도구는 TOOLS에 스키마를, run_tool에 실행 코드를 추가하세요."""

import ast
import math
import operator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .memory import Memory

# Anthropic 서버에서 실행되는 웹 검색 도구 (구현 불필요)
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS = [
    {
        "name": "calculator",
        "description": "수식을 정확히 계산합니다. + - * / // % ** 와 sqrt, sin, cos, tan, log, pi, e 를 지원합니다.",
        "input_schema": _schema({"expression": {"type": "string", "description": "예: (3+4)*2**3"}}, ["expression"]),
        "strict": True,
    },
    {
        "name": "current_time",
        "description": "지정한 시간대의 현재 날짜와 시각을 알려줍니다.",
        "input_schema": _schema(
            {"timezone": {"type": "string", "description": "IANA 시간대, 예: Asia/Seoul"}}, ["timezone"]
        ),
        "strict": True,
    },
    {
        "name": "remember",
        "description": "사용자에 대해 다음 대화에서도 기억할 사실을 장기 기억에 저장합니다.",
        "input_schema": _schema({"fact": {"type": "string"}}, ["fact"]),
        "strict": True,
    },
    {
        "name": "forget",
        "description": "키워드가 포함된 장기 기억을 지웁니다. 사용자가 잊어달라고 할 때 씁니다.",
        "input_schema": _schema({"keyword": {"type": "string"}}, ["keyword"]),
        "strict": True,
    },
    {
        "name": "list_files",
        "description": "workspace 폴더 안의 파일 목록을 보여줍니다.",
        "input_schema": _schema({}, []),
        "strict": True,
    },
    {
        "name": "read_file",
        "description": "workspace 폴더 안의 텍스트 파일을 읽습니다.",
        "input_schema": _schema({"path": {"type": "string", "description": "workspace 기준 상대 경로"}}, ["path"]),
        "strict": True,
    },
]

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "abs": abs}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float:
    """eval() 없이 산술식만 계산합니다."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 1000:
                raise ValueError("지수가 너무 큽니다")
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"지원하지 않는 식: {ast.dump(node)[:60]}")

    return ev(ast.parse(expression, mode="eval"))


def _resolve_in(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("workspace 밖의 파일은 읽을 수 없습니다")
    return path


def run_tool(name: str, args: dict, memory: Memory, workspace: Path) -> tuple[str, bool]:
    """도구를 실행하고 (결과 문자열, 오류 여부)를 돌려줍니다."""
    try:
        if name == "calculator":
            return str(safe_eval(args["expression"])), False
        if name == "current_time":
            now = datetime.now(ZoneInfo(args["timezone"]))
            return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)"), False
        if name == "remember":
            return memory.add(args["fact"]), False
        if name == "forget":
            return memory.forget(args["keyword"]), False
        if name == "list_files":
            workspace.mkdir(parents=True, exist_ok=True)
            files = sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file())
            return "\n".join(files) or "(workspace가 비어 있음)", False
        if name == "read_file":
            text = _resolve_in(workspace, args["path"]).read_text(encoding="utf-8")
            if len(text) > 50_000:
                return text[:50_000] + "\n...(50,000자 이후 생략 — 파일이 너무 깁니다)", False
            return text, False
        return f"알 수 없는 도구: {name}", True
    except Exception as e:  # 도구 오류는 AI에게 돌려줘서 스스로 고치게 합니다
        return f"오류: {e}", True
