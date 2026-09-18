FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY spyno_service ./spyno_service

EXPOSE 8000
CMD ["uvicorn", "spyno_service.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
