# Build from the repo root (docker compose here does this for you):
#   docker build -f benchmarks/cognee/Dockerfile -t amb-cognee .
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
# the floating tag lags behind the repo's required uv (>0.10,<0.13);
# pin the binary itself so builds don't depend on the tag's age
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /usr/local/bin/

WORKDIR /amb
COPY . .
RUN uv sync --package cognee-benchmark --no-dev

# same invocation as local usage; --no-sync keeps the env baked at build time
ENTRYPOINT ["uv", "run", "--no-sync", "amb"]
CMD ["--help"]
