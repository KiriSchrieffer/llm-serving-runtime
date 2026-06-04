FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]"

COPY benchmarks/ benchmarks/
COPY docs/ docs/
COPY .env.example .env.benchmark ./

EXPOSE 8000

ENV LLM_RUNTIME_BACKEND=mock
ENV LLM_RUNTIME_ENABLE_BATCHING=true
ENV LLM_RUNTIME_MAX_BATCH_SIZE=8
ENV LLM_RUNTIME_BATCH_TIMEOUT_MS=10

CMD ["uvicorn", "llm_runtime.main:app", "--host", "0.0.0.0", "--port", "8000"]
