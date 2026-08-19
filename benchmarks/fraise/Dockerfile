# Build from the repo root (docker compose here does this for you):
#   docker build -f benchmarks/fraise/Dockerfile -t amb-letta .
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /amb
COPY . .
RUN uv sync --package fraise-benchmark --no-dev

# same invocation as local usage; --no-sync keeps the env baked at build time
ENTRYPOINT ["uv", "run", "--no-sync", "amb"]
CMD ["--help"]
