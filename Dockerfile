# BlueprintAI backend - FastAPI + extraction pipeline
FROM python:3.12-slim

WORKDIR /app

# ezdxf/matplotlib need freetype AND actual font files: with no fonts
# installed, DXF text renders as empty outline rectangles
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 fonts-dejavu-core curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY db ./db

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=12 \
  CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
