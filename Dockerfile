FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_KB_BACKEND=jsonl \
    AGENT_CONFIG=configs/agent.deepseek.yaml \
    AGENT_RATE_LIMIT_PER_MINUTE=30

WORKDIR /app

COPY requirements-demo.txt ./
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY agent ./agent
COPY web ./web
COPY configs ./configs
COPY data/agent_manual_kb.jsonl ./data/agent_manual_kb.jsonl
COPY data/agent_demo_corpus.jsonl ./data/agent_demo_corpus.jsonl

EXPOSE 8077

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8077/healthz', timeout=3)"

CMD ["python", "-m", "agent", "serve", "--host", "0.0.0.0", "--port", "8077", "--config", "configs/agent.deepseek.yaml"]
