# app/Dockerfile

FROM python:3.13.7-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    musescore \
    fluidsynth \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/generated && chmod 777 /tmp/generated

ENV QT_QPA_PLATFORM=offscreen

COPY . .

RUN uv sync

EXPOSE 8888

HEALTHCHECK CMD curl --fail http://localhost:8888/_stcore/health

ENTRYPOINT [ "uv", "run"]

CMD [ "streamlit", "run", "app.py", "--server.port=8888", "--server.address=0.0.0.0" ]