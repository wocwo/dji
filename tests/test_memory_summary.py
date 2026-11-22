"""会话摘要与历史检索测试。"""

from __future__ import annotations

from tests.helpers import final


def test_summary_and_history_context(make_agent):
    replies = []
    for index in range(9):
        replies.append(final(f"回答 {index}"))
    agent = make_agent(replies, memory=True)
    session = "memory-test"
    for index in range(9):
        agent.ask(f"问题 {index}：电池保养", session_id=session)
    summary = agent.memory.get_summary(session)
    assert summary
    assert "电池保养" in summary
    result = agent.ask("之前问过什么电池问题？", session_id=session)
    context = "\n".join(
        message.content for message in result.transcript if message.role == "system"
    )
    assert "长期会话摘要" in context
    assert "相关历史对话" in context
