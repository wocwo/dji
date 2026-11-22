"""附件上下文测试。"""

from types import SimpleNamespace

from agent.server import _task_with_attachment


def test_attachment_text_is_labeled():
    request = SimpleNamespace(task="这张截图说明什么？", attachment_text="Battery Low")
    text = _task_with_attachment(request)
    assert "OCR" in text
    assert "Battery Low" in text
    assert "不能声称看过图片" in text


def test_no_attachment_returns_task():
    request = SimpleNamespace(task="电池怎么保养", attachment_text=None)
    assert _task_with_attachment(request) == "电池怎么保养"
