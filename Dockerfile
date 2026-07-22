# BlueprintAI backend - FastAPI + extraction pipeline

# Build stage: compile LibreDWG (GPL, https://github.com/LibreDWG/libredwg)
# for DWG -> DXF conversion. Only the dwg2dxf binary is carried into the
# final image. Static so no shared libs need copying.
FROM python:3.12-slim AS libredwg-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl xz-utils pkg-config && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/libredwg.tar.xz \
      https://github.com/LibreDWG/libredwg/releases/download/0.14/libredwg-0.14.tar.xz \
    && mkdir /tmp/libredwg && tar -xJf /tmp/libredwg.tar.xz -C /tmp/libredwg --strip-components=1 \
    && cd /tmp/libredwg \
    && ./configure --disable-shared --disable-bindings --disable-docs --program-prefix= \
    && make -j"$(nproc)" -C src && make -j"$(nproc)" -C programs dwg2dxf \
    && cp programs/dwg2dxf /usr/local/bin/dwg2dxf

FROM python:3.12-slim

WORKDIR /app

COPY --from=libredwg-build /usr/local/bin/dwg2dxf /usr/local/bin/dwg2dxf

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
