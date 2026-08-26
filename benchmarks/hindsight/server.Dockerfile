# The Hindsight server itself — the system under test. Built from the
# published `hindsight-api` distribution, not the project's own images:
# those stop at 0.6.2, which predates the pg_search backend the project
# itself recommends, while PyPI is at 0.9.x.
FROM python:3.13-slim

ARG HINDSIGHT_API_VERSION=0.9.2

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "hindsight-api==${HINDSIGHT_API_VERSION}"

EXPOSE 8888

# Hindsight wants its LLM key under its own name; `env_file` cannot rename
# a variable and compose interpolation only sees exported shell vars, so the
# mapping happens here — from ../../.env or CI's environment alike.
ENTRYPOINT ["/bin/sh", "-c", "exec env HINDSIGHT_API_LLM_API_KEY=\"${HINDSIGHT_API_LLM_API_KEY:-$OPENAI_API_KEY}\" hindsight-api"]
