# The Hindsight server itself — the system under test.
#
# Built from the published `hindsight-api` distribution rather than the
# project's own container image: those images stop at 0.6.2 while the
# source and the PyPI distribution are both at 0.9.x, and 0.6.2 predates
# the pg_search text-search backend the project itself recommends. This
# measures the current line instead of a year-old one.
FROM python:3.13-slim

ARG HINDSIGHT_API_VERSION=0.9.2

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "hindsight-api==${HINDSIGHT_API_VERSION}"

EXPOSE 8888

# Hindsight wants its LLM key under its own name. `env_file` can inject
# OPENAI_API_KEY but cannot rename it, and compose interpolation only
# sees an exported shell variable — so the mapping happens here, which
# works whether the key arrives from ../../.env or from CI's environment.
ENTRYPOINT ["/bin/sh", "-c", "exec env HINDSIGHT_API_LLM_API_KEY=\"${HINDSIGHT_API_LLM_API_KEY:-$OPENAI_API_KEY}\" hindsight-api"]
