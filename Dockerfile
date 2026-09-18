FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml README.md ./
COPY spyno_service ./spyno_service
COPY spyno_decision ./spyno_decision

# A política PPO puxa torch (~800 MB). Fora por padrão: a recomendada pela
# avaliação é ts_hybrid. Construa com --build-arg COM_PPO=1 se precisar.
ARG COM_PPO=0
RUN uv pip install --system --no-cache . "psycopg[binary]>=3.2" \
 && if [ "$COM_PPO" = "1" ]; then uv pip install --system --no-cache ".[ppo]"; fi

EXPOSE 8000
CMD ["uvicorn", "spyno_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
