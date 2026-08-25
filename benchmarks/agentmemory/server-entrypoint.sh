#!/bin/sh
# agentmemory server entrypoint, adapted from the project's own deploy
# template (deploy/fly/entrypoint.sh in rohitg00/agentmemory).
#
# Two deliberate differences from upstream's:
#
#   1. The bundled iii config binds 127.0.0.1 and uses relative ./data
#      paths, which is unreachable from another compose service. This
#      writes a config that binds 0.0.0.0 and uses absolute /data paths
#      — the same substitution upstream makes for every managed host.
#   2. Upstream generates a random HMAC secret on first boot and prints
#      it once. A benchmark needs the adapter to know the secret before
#      the server starts, so AGENTMEMORY_SECRET is taken from the
#      environment instead (compose sets the same value on both
#      services). Unset means an open API, which is what the container's
#      private network already is.
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

exec agentmemory "$@"
