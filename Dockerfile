FROM python:3.12.16-slim-trixie

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

# A política PPO puxa torch (~800 MB). Fora por padrão: a recomendada pela
# avaliação é ts_hybrid. Construa com --build-arg COM_PPO=1 se precisar.
ARG COM_PPO=0

# Dependências primeiro (camada em cache), exatamente as versões do uv.lock.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra postgres \
      $(if [ "$COM_PPO" = "1" ]; then echo --extra ppo; fi)

COPY spyno_service ./spyno_service
COPY spyno_decision ./spyno_decision
# Artefatos de warm-start (training/build_warmstart.py); o uso é opt-in por
# SPYNO_WARMSTART_SNAPSHOT.
COPY warmstart ./warmstart
RUN uv sync --frozen --no-dev --no-editable --extra postgres \
      $(if [ "$COM_PPO" = "1" ]; then echo --extra ppo; fi)

# Usuário sem privilégios. O código fica com dono root (só leitura); /app é do
# usuário para que o SQLite padrão (./spyno_service.db) ainda possa ser criado.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin spyno \
 && chown spyno:spyno /app
USER spyno

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "spyno_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
