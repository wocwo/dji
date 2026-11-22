"""长期记忆测试：运行记录、笔记、对话轮次与持久化。"""

from __future__ import annotations

from agent.memory import AgentMemory
from agent.types import AgentResult, Citation, Usage


def _result(task="电池怎么保养", answer="应避免低温充电。[S1]") -> AgentResult:
    return AgentResult(
        task=task,
        answer=answer,
        status="ok",
        citations=[Citation(ref="S1", chunk_id="c1", title="手册")],
        usage=Usage(tool_calls=1),
        trace_id="tr-test",
        duration_ms=12.5,
    )


def test_episode_roundtrip(tmp_path):
    memory = AgentMemory(str(tmp_path / "memory.db"))
    try:
        episode_id = memory.save_episode(_result(), session_id="s1")
        assert episode_id
        episodes = memory.recent_episodes(limit=5)
        assert len(episodes) == 1
        assert episodes[0]["task"] == "电池怎么保养"
        assert episodes[0]["citations"][0]["ref"] == "S1"
        assert episodes[0]["session_id"] == "s1"
    finally:
        memory.close()


def test_search_episodes(tmp_path):
    memory = AgentMemory(str(tmp_path / "memory.db"))
    try:
        memory.save_episode(_result(task="固件升级失败", answer="重新下载固件"))
        memory.save_episode(_result(task="电池保养", answer="低温下不要充电"))
        hits = memory.search_episodes("电池", limit=5)
        assert hits
        assert "电池" in hits[0]["task"]
    finally:
        memory.close()


def test_notes_and_tags(tmp_path):
    memory = AgentMemory(str(tmp_path / "memory.db"))
    try:
        note_id = memory.add_note("用户常用 Mavic 4 Pro", ["机型", "偏好"])
        memory.add_note("固件升级前备份素材", ["流程"])
        assert note_id is not None

        by_query = memory.search_notes("Mavic")
        assert by_query and by_query[0]["id"] == note_id

        by_tag = memory.search_notes(tags=["流程"])
        assert by_tag and "备份" in by_tag[0]["text"]

        assert memory.search_notes(tags=["不存在"]) == []
    finally:
        memory.close()


def test_turns(tmp_path):
    memory = AgentMemory(str(tmp_path / "memory.db"))
    try:
        memory.append_turn("s1", "user", "问题一")
        memory.append_turn("s1", "assistant", "回答一")
        memory.append_turn("s1", "user", "问题二")
        turns = memory.get_turns("s1", limit=10)
        assert [turn["content"] for turn in turns] == ["问题一", "回答一", "问题二"]
        assert memory.get_turns("other") == []
    finally:
        memory.close()


def test_persistence_across_instances(tmp_path):
    path = str(tmp_path / "memory.db")
    first = AgentMemory(path)
    first.add_note("持久化笔记", ["test"])
    first.close()

    second = AgentMemory(path)
    try:
        notes = second.search_notes("持久化")
        assert notes and notes[0]["text"] == "持久化笔记"
        assert second.stats()["notes"] == 1
    finally:
        second.close()


def test_disabled_memory_is_safe(tmp_path):
    memory = AgentMemory(str(tmp_path / "memory.db"), enabled=False)
    assert memory.available is False
    assert memory.save_episode(_result()) is None
    assert memory.add_note("x") is None
    assert memory.search_notes("x") == []
    assert memory.stats()["available"] is False
    memory.append_turn("s", "user", "x")
    memory.close()
