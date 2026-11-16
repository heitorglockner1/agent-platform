FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
# Extras opcionais: --build-arg EXTRAS=langchain,chroma (vazio instala só o núcleo).
ARG EXTRAS=langchain
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}"
RUN useradd -m app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "aiplatform.main:app", "--host", "0.0.0.0", "--port", "8000"]
