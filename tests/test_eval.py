"""Стенд eval: модель-заглушка, изоляция эталона, оценка и сводка.

Реальные LLM не вызываются: ScriptedClient выдаёт заранее заданные ответы,
HTTP-клиент проверяется на локальном сервере-заглушке.
"""

import csv
import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from eval.agent import run_agent
from eval.client import ChatResponse, OpenAICompatClient, ScriptedClient
from eval.config import EvalConfig, ModelConfig, RunConfig, load_config
from eval.runner import (
    WORKDIR_NAME,
    build_prompts,
    list_cases,
    load_logs,
    make_workspace,
    run_benchmark,
    run_case,
    summarize,
)
from eval.scoring import AMBIGUOUS, Answer, expected_regime, parse_answer, score
from eval.tools import MODES, REGISTRY, ToolContext, ToolError, ToolRegistry
from ring_toolkit.benchmark import generate_benchmark

SENTINEL = "TRUTH-SENTINEL-7f3a"
MODEL = ModelConfig(name="stub", model="stub-model")


# ----------------------------------------------------------------------
# Наблюдение за открытием truth.json в этом процессе
# ----------------------------------------------------------------------
_TRUTH_OPENS: list[str] = []
_WATCH = {"on": False}


def _audit(event, args):
    if _WATCH["on"] and event == "open" and args and isinstance(args[0], str | bytes):
        path = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if Path(path).name == "truth.json":
            _TRUTH_OPENS.append(path)


sys.addaudithook(_audit)


@contextmanager
def watch_truth():
    _TRUTH_OPENS.clear()
    _WATCH["on"] = True
    try:
        yield _TRUTH_OPENS
    finally:
        _WATCH["on"] = False


# ----------------------------------------------------------------------
# Фикстуры
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("bench")
    generate_benchmark(root, seed=0, traps=("overcoupled", "undersampled", "doublet"))
    for case in list_cases(root):  # метка, по которой видна утечка эталона
        truth = json.loads((case / "truth.json").read_text(encoding="utf-8"))
        truth["notes"] += f" {SENTINEL}"
        (case / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    return root


@pytest.fixture
def case(bench) -> Path:
    return list_cases(bench)[0]


@pytest.fixture
def ctx(case, tmp_path) -> ToolContext:
    workdir = make_workspace(case, tmp_path)
    return ToolContext(workdir=workdir, timeout_s=60, forbidden_roots=(case.parent,))


def _truth(case: Path) -> dict:
    return json.loads((case / "truth.json").read_text(encoding="utf-8"))


def _good_answer(truth: dict) -> dict:
    return {
        "regime": truth["regime"],
        "q_i": truth["q_i"] * 1.1,
        "anomalies": list(truth["expected_anomalies"]),
        "confidence": 0.8,
    }


def _cfg(**run) -> EvalConfig:
    return EvalConfig(models=(MODEL,), run=RunConfig(**run))


# ----------------------------------------------------------------------
# Изоляция эталона
# ----------------------------------------------------------------------
def test_workspace_has_no_truth(case, tmp_path):
    workdir = make_workspace(case, tmp_path)
    assert (case / "truth.json").is_file()
    assert not list(workdir.rglob("truth.json"))
    assert workdir.name == WORKDIR_NAME  # исходный case_id агенту не виден
    assert (workdir / "run_000" / "spectrum.npz").is_file()


def _adversarial_args(tool, case: Path) -> list[dict]:
    """Попытки добраться до эталона через каждый параметр инструмента."""
    truth = (case / "truth.json").resolve()
    paths = [
        "truth.json", "../truth.json", "./run_000/../truth.json", str(truth),
        str(case.resolve()), str(case.parent.resolve()), "..", "../..",
    ]
    code = [
        f"print(open({str(truth)!r}).read())",
        f"import glob; print(glob.glob({str(case.parents[1])!r} + '/**/truth.json', "
        "recursive=True))",
        f"import os; print(os.listdir({str(case.resolve())!r}))",
        f"import pathlib; print(pathlib.Path({str(truth)!r}).read_text())",
        f"import subprocess; print(subprocess.run(['cat', {str(truth)!r}], "
        "capture_output=True, text=True).stdout)",
        f"import os; os.system('cat {truth}')",
        f"import numpy as np; print(np.loadtxt({str(truth)!r}, dtype=str, delimiter='|'))",
    ]
    if not tool.parameters:
        return [{}]
    variants = []
    for name, spec in tool.parameters.items():
        values = code if name == "code" else paths
        if spec.get("type") == "string":
            variants += [{name: v} for v in values]
    return variants or [{}]


@pytest.mark.parametrize("tool_name", sorted(REGISTRY.tools))
def test_no_tool_reads_truth(tool_name, case, ctx):
    """Каждый зарегистрированный инструмент (и будущие тоже) не отдаёт эталон."""
    tool = REGISTRY.tools[tool_name]
    for args in _adversarial_args(tool, case):
        with watch_truth() as opens:
            try:
                out = REGISTRY.call(tool_name, args, ctx)
            except ToolError as e:
                out = str(e)
        assert SENTINEL not in out, (tool_name, args)
        assert '"q_i"' not in out and "q_intrinsic" not in out, (tool_name, args)
        assert opens == [], (tool_name, args, opens)


def test_run_python_guard_reports_denial(case, ctx):
    out = REGISTRY.call(
        "run_python", {"code": f"open({str((case / 'truth.json').resolve())!r})"}, ctx
    )
    assert "PermissionError" in out and "access denied" in out
    # обычная работа с данными разрешена
    out = REGISTRY.call(
        "run_python",
        {"code": "import numpy as np\nprint(len(np.load('run_000/spectrum.npz')['lam_nm']))"},
        ctx,
    )
    assert out.strip().isdigit()


@pytest.mark.parametrize("mode", sorted(MODES))
def test_full_run_never_touches_truth(mode, bench, case):
    """Прогон целиком: заглушка пробует всё, эталон не утекает ни в один ответ."""
    probes = [
        ScriptedClient.tool_call("list_files", {"path": ".."}),
        ScriptedClient.tool_call("list_files", {}),
    ]
    if mode == "naive":
        probes += [
            ScriptedClient.tool_call("read_file", {"path": "../truth.json"}),
            ScriptedClient.tool_call("read_file", {"path": str(case / "truth.json")}),
            ScriptedClient.tool_call(
                "run_python", {"code": f"print(open({str(case / 'truth.json')!r}).read())"}
            ),
        ]
    else:
        probes.append(ScriptedClient.tool_call("run_analysis", {}))
    client = ScriptedClient([*probes, ScriptedClient.final(_good_answer(_truth(case)))])

    import eval.runner as runner

    original = runner.run_agent

    def watched(*a, **kw):
        with watch_truth() as opens:
            res = original(*a, **kw)
            assert opens == []
        return res

    runner.run_agent = watched
    try:
        rec = run_case(client, MODEL, mode, case, _cfg())
    finally:
        runner.run_agent = original

    seen = json.dumps(client.calls, ensure_ascii=False)
    assert SENTINEL not in seen
    # исходный путь случая модели не показывали (сама заглушка его знает — не в счёт)
    assert case.name not in json.dumps(client.calls[0], ensure_ascii=False)
    for tc in rec["tool_calls"]:
        assert SENTINEL not in tc["output"]
    assert rec["score"]["valid"] and rec["score"]["regime_correct"]


def test_prompts_do_not_contain_truth(case):
    for mode in MODES:
        system, user = build_prompts(mode)
        assert SENTINEL not in system + user
        assert _truth(case)["trap"] not in user.lower().replace("_", " ")


# ----------------------------------------------------------------------
# Агентный цикл
# ----------------------------------------------------------------------
def test_agent_loop_records_calls_and_usage(ctx):
    client = ScriptedClient([
        ScriptedClient.tool_call("list_files", {}),
        ScriptedClient.tool_call("read_file", {"path": "run_000/params.json"}),
        ScriptedClient.final('{"regime": "critical", "q_i": 1, "anomalies": [], "confidence": 1}'),
    ])
    tools = REGISTRY.subset(MODES["naive"])
    res = run_agent(client, tools, ctx, "sys", "user", max_steps=5)
    assert res.stop_reason == "final" and res.steps == 3
    assert [tc.name for tc in res.tool_calls] == ["list_files", "read_file"]
    assert "run_000/spectrum.npz" in res.tool_calls[0].output
    assert "kappa2_design" in res.tool_calls[1].output
    assert res.usage == {"prompt_tokens": 30, "completion_tokens": 15}
    # модель видела вывод инструментов на следующем шаге
    assert client.calls[1][-1]["role"] == "tool"


def test_agent_step_limit(ctx):
    client = ScriptedClient([ScriptedClient.tool_call("list_files", {})] * 10)
    res = run_agent(client, REGISTRY.subset(["list_files"]), ctx, "s", "u", max_steps=3)
    assert res.stop_reason == "max_steps" and res.final_text is None and res.steps == 3


def test_agent_time_limit(ctx):
    client = ScriptedClient([ScriptedClient.final("{}")])
    res = run_agent(client, REGISTRY.subset(["list_files"]), ctx, "s", "u", time_limit_s=0)
    assert res.stop_reason == "time_limit" and client.calls == []


def test_agent_tool_errors_go_back_to_model(ctx):
    client = ScriptedClient([
        ScriptedClient.tool_call("no_such_tool", {}),
        ScriptedClient.tool_call("read_file", "{not json"),
        ScriptedClient.tool_call("read_file", {"path": "run_000/spectrum.npz"}),
        ScriptedClient.tool_call("read_file", {"wrong": 1}),
        ScriptedClient.final("done"),
    ])
    res = run_agent(client, REGISTRY.subset(MODES["naive"]), ctx, "s", "u")
    assert res.stop_reason == "final"
    assert all(tc.error for tc in res.tool_calls)
    assert "двоичный" in res.tool_calls[2].output


def test_run_python_timeout(ctx):
    ctx.timeout_s = 1
    with pytest.raises(ToolError, match="таймаут"):
        REGISTRY.call("run_python", {"code": "import time; time.sleep(5)"}, ctx)


def test_run_analysis_returns_analysis_json(ctx):
    data = json.loads(REGISTRY.call("run_analysis", {}, ctx))
    assert data["n_runs"] == 1 and "anomalies" in data["runs"][0]


def test_registry_is_extensible(ctx):
    """Новый инструмент подключается без правки цикла агента."""
    reg = ToolRegistry(dict(REGISTRY.tools))

    @reg.tool("count_runs", "Count run_* folders.", {})
    def count_runs(c: ToolContext) -> str:
        return str(len(list(c.workdir.glob("run_*"))))

    assert reg.tools["count_runs"].schema()["function"]["name"] == "count_runs"
    client = ScriptedClient([ScriptedClient.tool_call("count_runs", {}), ScriptedClient.final("x")])
    res = run_agent(client, reg.subset(["list_files", "count_runs"]), ctx, "s", "u")
    assert res.tool_calls[0].output == "1"
    with pytest.raises(ValueError):
        reg.register(reg.tools["count_runs"])


# ----------------------------------------------------------------------
# Разбор ответа и оценка
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "The answer is overcoupled",
        '```json\n{"regime": "critical", "q_i": 1, "anomalies": [], "confidence": 1}\n```',
        '{"regime": "critical", "q_i": 1, "anomalies": []}',
        '{"regime": "critical", "q_i": 1, "anomalies": [], "confidence": 1, "extra": 0}',
        '{"regime": "maybe", "q_i": 1, "anomalies": [], "confidence": 1}',
        '{"regime": "critical", "q_i": "1e5", "anomalies": [], "confidence": 1}',
        '{"regime": "critical", "q_i": true, "anomalies": [], "confidence": 1}',
        '{"regime": "critical", "q_i": -5, "anomalies": [], "confidence": 1}',
        '{"regime": "critical", "q_i": 1, "anomalies": "SMALL_FSR", "confidence": 1}',
        '{"regime": "critical", "q_i": 1, "anomalies": [], "confidence": 1.5}',
        '[1, 2]',
    ],
)
def test_parse_answer_rejects_invalid(text):
    answer, err = parse_answer(text)
    assert answer is None and err


def test_parse_answer_accepts_valid():
    answer, err = parse_answer(
        ' {"regime": "undercoupled", "q_i": null, "anomalies": ["x"], "confidence": 0} '
    )
    assert err is None and answer == Answer("undercoupled", None, ["x"], 0.0)


def test_score_metrics():
    truth = {"regime": "overcoupled", "q_i": 1e5,
             "expected_anomalies": ["UNDERSAMPLED", "SMALL_FSR"]}
    ans = Answer("overcoupled", 1.2e5, ["undersampled_fwhm", "BASELINE_TILT"], 0.5)
    s = score(ans, truth)
    assert s.valid and s.regime_correct
    assert s.q_i_rel_error == pytest.approx(0.2)
    assert s.anomaly_recall == 0.5  # алиас анализатора засчитан
    assert s.false_anomalies == ["BASELINE_TILT"]

    bad = score(None, truth, "не JSON")
    assert not bad.valid and not bad.regime_correct and bad.anomaly_recall == 0.0
    assert bad.parse_error == "не JSON"
    assert score(None, {**truth, "expected_anomalies": []}).anomaly_recall is None


def test_parse_answer_accepts_ambiguous():
    answer, err = parse_answer(
        '{"regime": "ambiguous", "q_i": 1e5, "anomalies": [], "confidence": 0.3}'
    )
    assert err is None and answer.regime == AMBIGUOUS


@pytest.mark.parametrize(
    ("identifiable", "said", "correct"),
    [
        (True, "overcoupled", True),
        (True, "undercoupled", False),
        (True, "ambiguous", False),  # восстановимый режим: уклониться — промах
        (False, "ambiguous", True),
        (False, "overcoupled", False),  # верная догадка без данных — всё равно ошибка
        (False, "undercoupled", False),
        (None, "overcoupled", True),  # старая истина без поля — считается восстановимой
    ],
)
def test_regime_scoring_with_identifiability(identifiable, said, correct):
    truth = {"regime": "overcoupled", "q_i": 1e5, "expected_anomalies": []}
    if identifiable is not None:
        truth["regime_identifiable"] = identifiable
    assert score(Answer(said, 1e5, [], 0.9), truth).regime_correct is correct
    assert expected_regime(truth) == ("overcoupled" if identifiable is not False else AMBIGUOUS)


def test_prompt_allows_ambiguous():
    _, user = build_prompts("naive")
    assert '"ambiguous"' in user and "if available" in user


def test_no_hint_benchmark_end_to_end(tmp_path):
    bench = tmp_path / "bench"
    generate_benchmark(bench, seed=1, traps=("overcoupled", "kappa_dispersion"),
                       design_hint=False)
    cases = {_truth(c)["trap"]: c for c in list_cases(bench)}
    for trap, case in cases.items():
        rec = run_case(
            ScriptedClient([
                ScriptedClient.tool_call("read_file", {"path": "run_000/params.json"}),
                ScriptedClient.final(
                    '{"regime": "ambiguous", "q_i": null, "anomalies": [], "confidence": 0.5}'
                ),
            ]),
            MODEL, "naive", case, _cfg(),
        )
        assert "kappa2_design" not in rec["tool_calls"][0]["output"]
        # без подсказки "ambiguous" верно для обычного случая и неверно для дисперсии kappa^2
        assert rec["score"]["regime_correct"] is (trap == "overcoupled")
        assert rec["truth"]["expected_regime"] == (
            AMBIGUOUS if trap == "overcoupled" else _truth(case)["regime"]
        )


# ----------------------------------------------------------------------
# Прогон, логи, сводка
# ----------------------------------------------------------------------
def test_run_benchmark_logs_and_summary(bench, tmp_path):
    cases = list_cases(bench)
    truths = {c.name: _truth(c) for c in cases}

    def factory(model: ModelConfig):
        # по случаю на прогон: в naive — верный ответ, в operator — невалидный
        def reply(messages):
            return ScriptedClient.final(_good_answer(truths[order.pop(0)]))

        order = [c.name for c in cases]
        script = [reply for _ in cases] + [ScriptedClient.final("not json")] * len(cases)
        return ScriptedClient(script)

    out = tmp_path / "runs"
    recs = run_benchmark(_cfg(), bench, out, client_factory=factory, progress=None)
    assert len(recs) == 2 * len(cases)

    logs = load_logs(out)
    assert len(logs) == len(recs)
    log = next(r for r in logs if r["mode"] == "naive")
    for key in ("model", "mode", "case_id", "tool_calls", "final_text", "answer", "score",
                "usage", "elapsed_s", "stop_reason", "messages"):
        assert key in log
    assert log["model"] == MODEL.public()

    rows = {r["mode"]: r for r in summarize(logs)}
    assert rows["naive"]["valid_rate"] == 1 and rows["naive"]["regime_accuracy"] == 1
    assert rows["naive"]["q_i_median_rel_error"] == pytest.approx(0.1)
    assert rows["operator"]["valid_rate"] == 0 and rows["operator"]["regime_accuracy"] == 0
    assert rows["operator"]["n_cases"] == len(cases)  # невалидные не отброшены

    with (out / "summary.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "| stub | naive |" in md and "| stub | operator |" in md


def test_cli_summarize(bench, tmp_path):
    from eval.__main__ import main

    out = tmp_path / "runs"
    run_benchmark(
        _cfg(modes=("operator",)), bench, out,
        client_factory=lambda m: ScriptedClient([ScriptedClient.final("x")] * 5),
        progress=None,
    )
    (out / "summary.md").unlink()
    assert main(["summarize", str(out)]) == 0
    assert (out / "summary.md").is_file()


# ----------------------------------------------------------------------
# Конфиг и HTTP-клиент
# ----------------------------------------------------------------------
def test_example_config_loads():
    cfg = load_config(Path(__file__).parents[1] / "eval" / "config.example.toml")
    assert cfg.models[0].base_url == "http://localhost:11434/v1"
    assert cfg.run.modes == ("naive", "operator")


class _FakeServer(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):  # noqa: N802 — имя задаёт http.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeServer.requests.append({"path": self.path, "body": body,
                                     "auth": self.headers.get("Authorization")})
        payload = {
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "list_files", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_openai_compat_client_request_format():
    server = HTTPServer(("127.0.0.1", 0), _FakeServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = ModelConfig(name="m", model="qwen2.5:7b", temperature=0.3, seed=11,
                          base_url=f"http://127.0.0.1:{server.server_port}/v1/")
        tools = REGISTRY.subset(MODES["operator"]).schemas()
        resp = OpenAICompatClient(cfg).chat([{"role": "user", "content": "hi"}], tools)
    finally:
        server.shutdown()
    req = _FakeServer.requests[-1]
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["model"] == "qwen2.5:7b"
    assert req["body"]["temperature"] == 0.3 and req["body"]["seed"] == 11
    assert [t["function"]["name"] for t in req["body"]["tools"]] == ["list_files", "run_analysis"]
    assert isinstance(resp, ChatResponse)
    assert resp.message["tool_calls"][0]["function"]["name"] == "list_files"
    assert resp.usage["prompt_tokens"] == 7
