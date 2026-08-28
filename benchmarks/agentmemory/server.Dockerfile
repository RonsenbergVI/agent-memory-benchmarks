# The agentmemory server — the system under test. No published image, so
# adapted from the project's own deploy template (deploy/fly/Dockerfile in
# rohitg00/agentmemory): node + the iii engine binary + the npm package.
ARG III_VERSION=0.11.2

FROM iiidev/iii:${III_VERSION} AS iii-image

FROM node:22-slim

ARG AGENTMEMORY_VERSION=0.9.29
ARG III_VERSION=0.11.2
ARG III_SDK_VERSION=0.11.2

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=iii-image /app/iii /usr/local/bin/iii

# A dedicated prefix so package.json `overrides` can pin iii-sdk to the
# engine's version (`npm install -g` ignores overrides): the caret range
# resolves to 0.11.6, whose sandboxed-worker model agentmemory is not
# refactored for — EPIPE reconnect loops, empty search after save.
# Upstream's own note.
WORKDIR /opt/agentmemory
RUN printf '{"name":"agentmemory-benchmark","version":"1.0.0","private":true,"overrides":{"iii-sdk":"%s"}}\n' "${III_SDK_VERSION}" > package.json \
 && npm install "@agentmemory/agentmemory@${AGENTMEMORY_VERSION}" --omit=optional --no-fund --no-audit \
 && ln -s /opt/agentmemory/node_modules/.bin/agentmemory /usr/local/bin/agentmemory

# gpt-5-mini rejects the `max_tokens` agentmemory sends; see the script.
COPY benchmarks/agentmemory/patch-openai-params.js /tmp/patch-openai-params.js
RUN node /tmp/patch-openai-params.js && rm /tmp/patch-openai-params.js

ENV AGENTMEMORY_III_VERSION=${III_VERSION} \
    AGENTMEMORY_DATA_DIR=/data \
    TINI_SUBREAPER=1

COPY --chmod=0755 benchmarks/agentmemory/server-entrypoint.sh /usr/local/bin/agentmemory-entrypoint.sh

EXPOSE 3111

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/agentmemory-entrypoint.sh"]
