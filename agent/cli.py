"""命令行入口：run / chat / demo / eval / tools / health / trace / config / serve。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent
from .config import AgentConfig, load_config
from .errors import AgentError
from .llm import create_llm
from .policy import AgentPolicy
from .prompt import truncate_text
from .trace import list_traces, read_trace, render_trace
from .types import AgentEvent, AgentResult, Message


# ---------------------------------------------------------------------------
# 输出辅助
# ---------------------------------------------------------------------------

def _setup_console() -> None:
    """避免 Windows 控制台遇到不可编码字符时崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _print_json(data: Any) -> None:
    # 控制台编码不稳定时，ASCII 转义保证输出仍是合法 JSON
    print(json.dumps(data, ensure_ascii=True, indent=2, default=str))


def _print_yaml(data: Any) -> None:
    try:
        import yaml

        print(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    except Exception:
        _print_json(data)


def _event_printer(*, quiet: bool = False) -> Callable[[AgentEvent], None]:
    def on_event(event: AgentEvent) -> None:
        if quiet:
            return
        payload = event.payload
        if event.type == "tool_call":
            args = json.dumps(payload.get("arguments", {}), ensure_ascii=False)
            print(f"\n  -> 调用 {payload.get('name')}({truncate_text(args, 180)})", flush=True)
        elif event.type == "tool_result":
            mark = "OK" if payload.get("ok") else "FAIL"
            content = str(payload.get("content", "")).replace("\n", " ")
            print(f"  <- [{mark}] {truncate_text(content, 200)}", flush=True)
        elif event.type == "warning":
            print(f"  [!] {payload.get('message', '')}", flush=True)
        elif event.type == "budget":
            print(f"  [停止] 预算触发：{payload.get('reason')}", flush=True)

    return on_event


# ---------------------------------------------------------------------------
# 配置与构建
# ---------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Agent YAML 配置路径（默认 configs/agent.yaml）")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "vllm", "deepseek", "offline"],
        default=None,
        help="覆盖 LLM 后端",
    )
    parser.add_argument("--model", default=None, help="覆盖模型名")
    parser.add_argument("--base-url", default=None, help="覆盖 LLM 服务地址")
    parser.add_argument("--kb", choices=["auto", "chroma", "jsonl", "none"], default=None, help="覆盖知识库后端")
    parser.add_argument("--jsonl", default=None, help="覆盖 JSONL 语料路径")
    parser.add_argument("--chroma", default=None, help="覆盖 Chroma 目录")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖最大步数")
    parser.add_argument("--no-trace", action="store_true", help="关闭 trace 写入")
    parser.add_argument("--trace-dir", default=None, help="覆盖 trace 目录")
    parser.add_argument("--no-memory", action="store_true", help="关闭长期记忆")
    parser.add_argument("--allow-risky", action="store_true", help="允许高风险工具（危险）")
    parser.add_argument(
        "--no-fallback-offline",
        dest="fallback_offline",
        action="store_false",
        default=True,
        help="LLM 不可用时不自动降级到离线策略",
    )
    parser.add_argument("--verbose", action="store_true", help="输出完整异常堆栈")


def _config_from_args(args: argparse.Namespace) -> AgentConfig:
    config = load_config(getattr(args, "config", None))
    if getattr(args, "provider", None):
        config.llm.provider = args.provider.strip().lower()
        if config.llm.provider == "offline":
            config.llm.model = "offline-policy"
    if getattr(args, "model", None):
        config.llm.model = args.model
    if getattr(args, "base_url", None):
        config.llm.base_url = args.base_url
    if getattr(args, "kb", None):
        config.knowledge.backend = args.kb
    if getattr(args, "jsonl", None):
        config.knowledge.jsonl_path = args.jsonl
    if getattr(args, "chroma", None):
        config.knowledge.chroma_path = args.chroma
    if getattr(args, "max_steps", None):
        config.agent.max_steps = max(1, args.max_steps)
    if getattr(args, "no_trace", False):
        config.trace.enabled = False
    if getattr(args, "trace_dir", None):
        config.trace.dir = args.trace_dir
    if getattr(args, "no_memory", False):
        config.memory.enabled = False
    if getattr(args, "allow_risky", False):
        config.tools.allow_risky = True
    config.validate()
    return config


def _materialize_agent(args: argparse.Namespace) -> Agent:
    """从参数构建 Agent；真实 LLM 不可用时默认降级到离线策略。"""
    config = _config_from_args(args)
    provider = config.llm.provider
    if getattr(args, "fallback_offline", True) and provider not in ("offline",):
        try:
            health = create_llm(config.llm).health()
        except Exception as exc:  # pragma: no cover - 防御
            health = {"healthy": False, "error": str(exc)}
        if not health.get("healthy"):
            reason = str(health.get("error") or "健康检查未通过").replace("\n", " ")
            print(
                f"[提示] LLM 后端 {provider} 不可用（{reason[:160]}），"
                "已自动降级为离线策略。使用 --no-fallback-offline 可关闭。",
                file=sys.stderr,
            )
            config.llm.provider = "offline"
            config.llm.model = "offline-policy"

    policy = AgentPolicy.from_config(config)
    return Agent(config, policy=policy)


def _require_kb(agent: Agent) -> bool:
    if agent.kb is not None:
        return True
    print(
        "错误：没有可用的知识库。请确认 chroma_final/ 存在，或先生成演示语料：\n"
        "  python scripts/build_demo_corpus.py\n"
        "也可用 --kb jsonl --jsonl data/agent_demo_corpus.jsonl 指定语料。",
        file=sys.stderr,
    )
    return False


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    task = " ".join(args.task).strip()
    if not task:
        print("错误：任务不能为空", file=sys.stderr)
        return 2
    agent = _materialize_agent(args)
    try:
        session = args.session
        result = agent.ask(
            task,
            session_id=session,
            on_event=_event_printer(quiet=args.json or args.quiet),
        )
        if args.json:
            _print_json(result.to_dict(include_steps=True))
        else:
            print()
            print(result.to_markdown())
            trace_path = result.metadata.get("trace_path")
            if trace_path:
                print(f"\nTrace: {trace_path}")
        return 0 if result.status == "ok" and result.answer else 1
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

def cmd_chat(args: argparse.Namespace) -> int:
    agent = _materialize_agent(args)
    session = args.session or f"chat-{time.strftime('%Y%m%d-%H%M%S')}"
    history: List[Message] = []
    printer = _event_printer(quiet=args.quiet)
    last_result: Optional[AgentResult] = None

    print("大疆手册 Agent 交互模式。输入 /help 查看命令，/exit 退出。")
    print(f"会话 ID：{session}")
    try:
        while True:
            try:
                line = input("\n你> ").strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            if line in ("/exit", "/quit", "exit", "quit"):
                break
            if line == "/help":
                print(
                    "/tools   查看工具\n"
                    "/health  健康检查\n"
                    "/plan    查看上一次运行的计划\n"
                    "/clear   清空当前对话历史（不影响长期记忆）\n"
                    "/session 显示会话 ID\n"
                    "/exit    退出"
                )
                continue
            if line == "/tools":
                for spec in agent.tools.describe(agent.runtime.enabled_tools):
                    print(f"- {spec['name']}（{spec['risk']}）：{spec['description']}")
                continue
            if line == "/health":
                _print_json(agent.health())
                continue
            if line == "/plan":
                if last_result is None:
                    print("还没有运行记录。")
                else:
                    plan = last_result.metadata.get("plan") or []
                    print(json.dumps(plan, ensure_ascii=False, indent=2) if plan else "本轮没有显式计划。")
                continue
            if line == "/clear":
                history.clear()
                print("已清空当前对话历史。")
                continue
            if line == "/session":
                print(session)
                continue

            try:
                result = agent.ask(line, history=history, session_id=session, on_event=printer)
            except Exception as exc:
                print(f"\n运行失败：{exc}", file=sys.stderr)
                continue
            last_result = result
            print("\n" + result.answer)
            if result.citations:
                print("\n引用：")
                for citation in result.citations:
                    print(f"  [{citation.ref}] {citation.label()}")
            if result.status != "ok":
                print(f"\n[状态] {result.status}；警告：{'; '.join(result.warnings) or '无'}")
            history.append(Message("user", line))
            history.append(Message("assistant", result.answer))
    finally:
        agent.close()
    return 0


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

DEMO_TASK = "智能飞行电池在低温环境下如何保养？"


def cmd_demo(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    config.llm.provider = "offline"
    config.llm.model = "offline-policy"
    # 演示默认使用自带的 JSONL 语料，保证秒级、可复现；--kb chroma 可切到真实向量库
    if not getattr(args, "kb", None) and os.path.exists(config.knowledge.jsonl_path):
        config.knowledge.backend = "jsonl"
    agent = Agent(config, policy=AgentPolicy.from_config(config))
    if not _require_kb(agent):
        agent.close()
        return 2
    task = " ".join(args.task).strip() if args.task else DEMO_TASK

    if not args.json:
        print("=" * 66)
        print("DJI 手册 Agent · 离线演示")
        print("=" * 66)
        print("流程：LLM 决策 -> 调用手册检索工具 -> 读取 observation -> 再决策 -> 带引用作答")
        print(f"任务：{task}")
        print(f"知识库：{agent.kb.description}")
        print("-" * 66)

    try:
        result = agent.ask(task, on_event=_event_printer(quiet=args.json))
    finally:
        agent.close()

    if args.json:
        _print_json(result.to_dict(include_steps=True))
        return 0 if result.answer else 1

    print("\n" + "=" * 66)
    print("最终回答")
    print("=" * 66)
    print(result.answer)
    if result.citations:
        print("\n引用来源：")
        for citation in result.citations:
            print(f"  [{citation.ref}] {citation.label()}")
    print("\n运行摘要：")
    print(f"  状态      : {result.status}")
    print(f"  步数      : {len(result.steps)}")
    print(f"  工具调用  : {result.usage.tool_calls}")
    print(f"  耗时      : {result.duration_ms / 1000:.2f}s")
    print(f"  Trace     : {result.metadata.get('trace_path', '(未写入)')}")
    if result.warnings:
        print(f"  警告      : {'; '.join(result.warnings)}")
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> int:
    from .eval import DEFAULT_CASES, format_report, run_eval, save_report

    agent = _materialize_agent(args)
    if not _require_kb(agent):
        agent.close()
        return 2
    cases = DEFAULT_CASES
    if args.case:
        wanted = set(args.case)
        cases = [case for case in DEFAULT_CASES if case.id in wanted]
    if not args.json:
        print(f"评测后端：{agent.llm.name} / 知识库：{agent.kb.description}")
    try:
        report = run_eval(agent, cases, verbose=not args.json)
    finally:
        agent.close()
    if args.output:
        save_report(report, args.output)
        print(f"报告已保存：{args.output}")
    if args.json:
        _print_json(report)
    else:
        print("\n" + format_report(report))
    return 0 if report["passed"] == report["total"] else 1


# ---------------------------------------------------------------------------
# tools / health / trace / config / serve
# ---------------------------------------------------------------------------

def cmd_tools(args: argparse.Namespace) -> int:
    agent = _materialize_agent(args)
    try:
        specs = agent.tools.describe(agent.runtime.enabled_tools)
    finally:
        agent.close()
    if args.json:
        _print_json(specs)
        return 0
    if not specs:
        print("没有启用任何工具。请检查 tools.enabled 配置。")
        return 0
    for spec in specs:
        print(f"{spec['name']}  [{spec['risk']}]")
        print(f"  {spec['description']}")
        params = spec.get("parameters", {}).get("properties", {})
        if params:
            print(f"  参数: {', '.join(params.keys())}")
        print()
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    agent = _materialize_agent(args)
    try:
        health = agent.health()
    finally:
        agent.close()
    _print_json(health)
    return 0 if health.get("llm", {}).get("healthy") else 1


def cmd_trace(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    directory = config.trace.dir
    if args.show:
        target = args.show
        path = target
        if not os.path.exists(path):
            for record in list_traces(directory, limit=500):
                if record.get("trace_id") == target:
                    path = record.get("path", "")
                    break
        if not path or not os.path.exists(path):
            print(f"未找到 trace：{target}", file=sys.stderr)
            return 1
        events = read_trace(path)
        print(render_trace(events))
        return 0

    records = list_traces(directory, limit=args.limit)
    if not records:
        print(f"没有 trace 记录（目录：{directory}）。")
        return 0
    for record in records:
        print(
            f"{record.get('ts', '')}  {record.get('trace_id', '')}  "
            f"{record.get('status', ''):10}  {record.get('duration_ms', 0):8.0f}ms  "
            f"{record.get('task', '')[:60]}"
        )
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    data = config.to_dict()
    if isinstance(data.get("llm"), dict) and data["llm"].get("api_key"):
        data["llm"]["api_key"] = "***"
    if args.yaml:
        _print_yaml(data)
    else:
        _print_json(data)
    print(f"\n配置文件: {config.config_path or '(默认值)'}", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(
        config_path=args.config,
        host=args.host,
        port=args.port,
        open_browser=getattr(args, "open", False),
        fallback_offline=getattr(args, "fallback_offline", True),
    )
    return 0


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent",
        description="DJI 手册 Agent：多步工具调用、检索、记忆与可观测性",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    _add_common(common)

    p_run = sub.add_parser("run", parents=[common], help="执行单次任务")
    p_run.add_argument("task", nargs="+", help="用户任务/问题")
    p_run.add_argument("--session", default=None, help="会话 ID（用于恢复历史）")
    p_run.add_argument("--json", action="store_true", help="输出 JSON 结果")
    p_run.add_argument("--quiet", action="store_true", help="不打印中间过程")
    p_run.set_defaults(func=cmd_run)

    p_chat = sub.add_parser("chat", parents=[common], help="交互式多轮对话")
    p_chat.add_argument("--session", default=None, help="会话 ID")
    p_chat.add_argument("--quiet", action="store_true", help="不打印工具过程")
    p_chat.set_defaults(func=cmd_chat)

    p_demo = sub.add_parser("demo", parents=[common], help="离线演示（无需 LLM）")
    p_demo.add_argument("task", nargs="*", help="演示问题")
    p_demo.add_argument("--json", action="store_true", help="输出 JSON 结果")
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("eval", parents=[common], help="运行 Agent 评测集")
    p_eval.add_argument("--case", action="append", default=None, help="只运行指定用例（可重复）")
    p_eval.add_argument("--output", default=None, help="保存 JSON 报告路径")
    p_eval.add_argument("--json", action="store_true", help="输出 JSON 报告")
    p_eval.set_defaults(func=cmd_eval)

    p_tools = sub.add_parser("tools", parents=[common], help="列出可用工具")
    p_tools.add_argument("--json", action="store_true", help="输出 JSON")
    p_tools.set_defaults(func=cmd_tools)

    p_health = sub.add_parser("health", parents=[common], help="组件健康检查")
    p_health.set_defaults(func=cmd_health)

    p_trace = sub.add_parser("trace", parents=[common], help="查看运行轨迹")
    p_trace.add_argument("--list", action="store_true", help="列出最近的 trace（默认行为）")
    p_trace.add_argument("--show", default=None, help="显示指定 trace_id 或文件")
    p_trace.add_argument("--limit", type=int, default=20, help="列出条数")
    p_trace.set_defaults(func=cmd_trace)

    p_config = sub.add_parser("config", parents=[common], help="打印生效配置")
    p_config.add_argument("--yaml", action="store_true", help="以 YAML 输出")
    p_config.set_defaults(func=cmd_config)

    p_serve = sub.add_parser("serve", parents=[common], help="启动 HTTP API 与 Web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8077)
    p_serve.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    except AgentError as exc:
        print(f"Agent 错误：{exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1
    except Exception as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
