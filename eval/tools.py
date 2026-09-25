"""Инструменты агента и расширяемый реестр.

Новый инструмент добавляется без правки цикла агента::

    @REGISTRY.tool("my_tool", "описание", {"x": {"type": "string"}})
    def my_tool(ctx: ToolContext, x: str) -> str: ...

    MODES["my_mode"] = ("list_files", "my_tool")

Все инструменты работают внутри ``ctx.workdir`` — копии случая без truth.json.
Защита эталона двойная: truth.json в рабочую папку не копируется (см.
eval.workspace), а инструменты отказываются открывать файлы с именем
truth.json и пути вне рабочей папки. ``run_python`` запускает код в
подпроцессе с audit-хуком, который запрещает открывать truth.json, читать
каталог случаев и логов и запускать внешние процессы. Это защита от
случайного или любопытного доступа, а не песочница против целенаправленного
обхода: код через ctypes может вызвать libc напрямую, минуя audit-хук
(запретить ctypes нельзя — его импортирует numpy). Для недоверенных моделей
запускайте стенд в контейнере без доступа к папке случаев.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

TRUTH_NAME = "truth.json"
# корень репозитория: чтобы ring_toolkit импортировался и без pip install
REPO_ROOT = Path(__file__).resolve().parents[1]


class ToolError(Exception):
    """Ошибка, которую агент видит как результат вызова инструмента."""


@dataclass
class ToolContext:
    workdir: Path
    timeout_s: float = 60.0
    # каталоги, которые агенту нельзя читать даже из run_python (исходные случаи)
    forbidden_roots: tuple[Path, ...] = ()


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema properties
    func: Callable[..., str]
    required: tuple[str, ...] = ()

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.required),
                },
            },
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self.tools:
            raise ValueError(f"инструмент {tool.name!r} уже зарегистрирован")
        self.tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: dict | None = None):
        """Декоратор: обязательные аргументы берутся из сигнатуры функции."""

        def deco(func: Callable[..., str]) -> Callable[..., str]:
            sig = inspect.signature(func)
            required = tuple(
                p.name
                for p in list(sig.parameters.values())[1:]
                if p.default is inspect.Parameter.empty
            )
            self.register(Tool(name, description, parameters or {}, func, required))
            return func

        return deco

    def subset(self, names: tuple[str, ...] | list[str]) -> ToolRegistry:
        missing = [n for n in names if n not in self.tools]
        if missing:
            raise KeyError(f"нет инструментов: {missing}")
        return ToolRegistry({n: self.tools[n] for n in names})

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: dict, ctx: ToolContext) -> str:
        if name not in self.tools:
            raise ToolError(f"неизвестный инструмент {name!r}; доступны: {sorted(self.tools)}")
        tool = self.tools[name]
        unknown = set(arguments) - set(tool.parameters)
        if unknown:
            raise ToolError(f"{name}: неизвестные аргументы {sorted(unknown)}")
        missing = set(tool.required) - set(arguments)
        if missing:
            raise ToolError(f"{name}: не хватает аргументов {sorted(missing)}")
        return tool.func(ctx, **arguments)


REGISTRY = ToolRegistry()

MODES: dict[str, tuple[str, ...]] = {
    "naive": ("list_files", "read_file", "run_python"),
    "operator": ("list_files", "run_analysis"),
}


# ----------------------------------------------------------------------
# Доступ к файлам
# ----------------------------------------------------------------------
def resolve_in_workdir(ctx: ToolContext, rel: str) -> Path:
    """Путь внутри рабочей папки; всё остальное — ToolError."""
    root = ctx.workdir.resolve()
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise ToolError(f"путь {rel!r} вне рабочей папки")
    if path.name == TRUTH_NAME:
        raise ToolError(f"доступ к {TRUTH_NAME} запрещён")
    return path


@REGISTRY.tool(
    "list_files",
    "List files in the working directory (recursively) with sizes in bytes.",
    {"path": {"type": "string", "description": "subdirectory, default '.'"}},
)
def list_files(ctx: ToolContext, path: str = ".") -> str:
    base = resolve_in_workdir(ctx, path)
    if not base.is_dir():
        raise ToolError(f"{path!r} — не каталог")
    root = ctx.workdir.resolve()
    lines = [
        f"{p.relative_to(root).as_posix()}\t{p.stat().st_size}"
        for p in sorted(base.rglob("*"))
        if p.is_file() and p.name != TRUTH_NAME
    ]
    return "\n".join(lines) or "(пусто)"


@REGISTRY.tool(
    "read_file",
    "Read a text file from the working directory (JSON, CSV, Markdown...).",
    {"path": {"type": "string", "description": "file path relative to the working directory"}},
)
def read_file(ctx: ToolContext, path: str) -> str:
    p = resolve_in_workdir(ctx, path)
    if not p.is_file():
        raise ToolError(f"файла {path!r} нет")
    data = p.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError(
            f"{path!r} — двоичный файл ({len(data)} байт); используйте run_python"
        ) from None


# ----------------------------------------------------------------------
# Выполнение кода
# ----------------------------------------------------------------------
_GUARD = textwrap.dedent(
    """
    import os as _os, sys as _sys
    _FORBIDDEN = {forbidden!r}
    _WORK = {work!r}
    _TRUTH = {truth!r}
    _DENY_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                    "os.spawn", "os.fork", "os.forkpty", "pty.spawn")

    def _blocked(path):
        try:
            p = _os.path.realpath(_os.fsdecode(path))
        except (TypeError, ValueError):
            return False
        if _os.path.basename(p) == _TRUTH:
            return True
        if p == _WORK or p.startswith(_WORK + _os.sep):
            return False
        return any(p == r or p.startswith(r + _os.sep) for r in _FORBIDDEN)

    def _guard(event, args):
        if event == "open" and args and not isinstance(args[0], int) and _blocked(args[0]):
            raise PermissionError("access denied: " + str(args[0]))
        if event in ("os.listdir", "os.scandir") and args and args[0] is not None \\
                and _blocked(args[0]):
            raise PermissionError("access denied: " + str(args[0]))
        if event.startswith(_DENY_EVENTS):
            raise PermissionError("starting processes is not allowed")

    _sys.addaudithook(_guard)
    del _guard
    _code = open(_sys.argv[1], encoding="utf-8").read()
    exec(compile(_code, "<agent>", "exec"), {{"__name__": "__main__"}})
    """
)


def run_subprocess(cmd: list[str], ctx: ToolContext) -> str:
    # минимальное окружение: ключи API и прочие секреты в подпроцесс не попадают
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": tempfile.gettempdir(),
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=ctx.workdir,
            capture_output=True,
            text=True,
            timeout=ctx.timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"превышен таймаут {ctx.timeout_s:g} с") from None
    out = proc.stdout
    if proc.stderr:
        out += ("\n[stderr]\n" if out else "[stderr]\n") + proc.stderr
    if proc.returncode:
        out += f"\n[exit code {proc.returncode}]"
    return out or "(нет вывода)"


@REGISTRY.tool(
    "run_python",
    "Run a Python script (numpy, scipy available) in the working directory. "
    "Returns stdout and stderr. Print what you need to see.",
    {"code": {"type": "string", "description": "Python source code"}},
)
def run_python(ctx: ToolContext, code: str) -> str:
    forbidden = [str(p.resolve()) for p in ctx.forbidden_roots]
    guard = _GUARD.format(
        forbidden=forbidden, work=str(ctx.workdir.resolve()), truth=TRUTH_NAME
    )
    with tempfile.TemporaryDirectory(prefix="eval_code_") as tmp:
        guard_py, code_py = Path(tmp) / "guard.py", Path(tmp) / "code.py"
        guard_py.write_text(guard, encoding="utf-8")
        code_py.write_text(code, encoding="utf-8")
        return run_subprocess([sys.executable, str(guard_py), str(code_py)], ctx)


@REGISTRY.tool(
    "run_analysis",
    "Run `python -m ring_toolkit.analyze` on the working directory and return analysis.json.",
    {},
)
def run_analysis(ctx: ToolContext) -> str:
    out = run_subprocess([sys.executable, "-m", "ring_toolkit.analyze", "."], ctx)
    result = ctx.workdir / "analysis.json"
    if not result.is_file():
        raise ToolError(f"analysis.json не создан. Вывод анализа:\n{out}")
    return json.dumps(json.loads(result.read_text(encoding="utf-8")), indent=1, ensure_ascii=False)
