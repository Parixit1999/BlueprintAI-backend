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
ARG TARGETARCH

WORKDIR /app

COPY --from=libredwg-build /usr/local/bin/dwg2dxf /usr/local/bin/dwg2dxf

# ezdxf/matplotlib need freetype AND actual font files: with no fonts
# installed, DXF text renders as empty outline rectangles
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 fonts-dejavu-core curl && rm -rf /var/lib/apt/lists/*

# ODA File Converter (free, closed-source, x64-only): reads every DWG
# version including AutoCAD 2018+ (AC1032), which LibreDWG cannot. Its Qt
# only ships the xcb platform plugin, so it runs under xvfb via the
# oda-convert wrapper. amd64 (production) only - arm64 dev machines fall
# back to the bundled LibreDWG dwg2dxf automatically.
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      apt-get update && \
      curl -fsSL -o /tmp/oda.deb \
        'https://www.opendesign.com/guestfiles/get?filename=ODAFileConverter_QT6_lnxX64_8.3dll_27.1.deb' && \
      apt-get install -y --no-install-recommends /tmp/oda.deb \
        xvfb xauth libgl1 libglib2.0-0 libfontconfig1 libxkbcommon0 libdbus-1-3 \
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-render-util0 libxcb-shape0 libxcb-xkb1 libxcb-cursor0 && \
      rm /tmp/oda.deb && rm -rf /var/lib/apt/lists/* && \
      printf '#!/bin/sh\nexec xvfb-run -a ODAFileConverter "$@"\n' \
        > /usr/local/bin/oda-convert && \
      chmod +x /usr/local/bin/oda-convert ; \
    fi
# harmless when the wrapper does not exist (arm64): the converter check
# does shutil.which() and falls through to dwg2dxf
ENV ODA_CONVERTER_PATH=/usr/local/bin/oda-convert

# uv resolves and installs the requirements several times faster than pip,
# and the BuildKit cache mount keeps downloaded wheels across builds - a
# requirements bump only fetches what changed. --compile-bytecode trades a
# little build time for faster container starts.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --compile-bytecode -r requirements.txt

COPY app ./app
COPY db ./db

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=12 \
  CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
