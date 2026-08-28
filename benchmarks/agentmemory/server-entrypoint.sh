#!/bin/sh
# agentmemory server entrypoint, adapted from upstream's
# deploy/fly/entrypoint.sh with two deliberate differences: the bundled
# iii config binds 127.0.0.1 with relative ./data paths (unreachable from
# another compose service), so this writes one binding 0.0.0.0 with
# absolute /data paths; and the HMAC secret comes from AGENTMEMORY_SECRET
# instead of upstream's random first-boot print, because the adapter must
# know it before the server starts. Unset means an open API — which the
# container's private network already is.
set -eu

DATA_DIR="${AGENTMEMORY_DATA_DIR:-/data}"
III_CONFIG="/opt/agentmemory/node_modules/@agentmemory/agentmemory/dist/iii-config.yaml"

mkdir -p "$DATA_DIR"

cat > "$III_CONFIG" <<'EOF'
workers:
  - name: iii-http
    config:
      port: 3111
      host: 0.0.0.0
      default_timeout: 180000
  - name: iii-state
    config:
      adapter:
        name: kv
        config:
          store_method: file_based
          file_path: /data/state_store.db
  - name: iii-queue
    config:
      adapter:
        name: builtin
  - name: iii-pubsub
    config:
      adapter:
        name: local
  - name: iii-cron
    config:
      adapter:
        name: kv
  - name: iii-stream
    config:
      port: 3112
      host: 0.0.0.0
      adapter:
        name: kv
        config:
          store_method: file_based
          file_path: /data/stream_store
  - name: iii-observability
    config:
      enabled: true
      service_name: agentmemory
      exporter: memory
      sampling_ratio: 1.0
      metrics_enabled: true
      logs_enabled: true
      logs_console_output: true
EOF

# The bare OPENAI_API_KEY drives the LLM (detectProvider); the embedding
# client resolves OPENAI_EMBEDDING_API_KEY || OPENAI_API_KEY. Both are
# wanted, so the key is mirrored and the bare one left in place.
if [ -n "${OPENAI_API_KEY:-}" ] && [ -z "${OPENAI_EMBEDDING_API_KEY:-}" ]; then
  OPENAI_EMBEDDING_API_KEY="$OPENAI_API_KEY"
  export OPENAI_EMBEDDING_API_KEY
fi

exec agentmemory "$@"
