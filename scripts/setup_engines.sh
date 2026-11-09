#!/usr/bin/env bash
# Installs the systems under test (Sec. V-A1) on Ubuntu 24.04, x86-64:
#   qdrant       official release binaries 1.19.2, 1.19.0 and 1.18.3
#   clickhouse   official static binaries 26.9.13.15 and 26.3.45.3
#   chroma       chromadb 1.5.9 from PyPI, in its own virtual environment
#   pgvector     PostgreSQL 16 (Ubuntu package) with pgvector 0.8.6 built from source,
#                and the unprivileged user pguser that runs the PostgreSQL servers
# Binaries go to engines/ (or $PLIMSOLL_ENGINES), in the layout plimsoll/paths.py expects.
# Usage: sudo scripts/setup_engines.sh [qdrant] [clickhouse] [chroma] [pgvector]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
ENGINES=${PLIMSOLL_ENGINES:-$PWD/engines}
mkdir -p "$ENGINES"

fetch() {  # fetch <url> <file>
  curl -fsSL --retry 3 -o "$2" "$1"
}

qdrant() {
  local v d
  for v in 1.19.2 1.19.0 1.18.3; do
    d="$ENGINES/qdrant-$v"
    [ -x "$d/qdrant" ] && continue
    mkdir -p "$d"
    fetch "https://github.com/qdrant/qdrant/releases/download/v$v/qdrant-x86_64-unknown-linux-gnu.tar.gz" "$d/qdrant.tar.gz"
    if [ "$v" = 1.19.2 ]; then
      echo "34a788a09a4cb278b5c6d2af96d9a6db30c8c7e88ae3550455f7a4d425bb8d4b  $d/qdrant.tar.gz" | sha256sum -c -
    fi
    tar xzf "$d/qdrant.tar.gz" -C "$d"
    rm "$d/qdrant.tar.gz"
  done
}

clickhouse() {
  local tag v d
  for tag in 26.9.13.15-stable 26.3.45.3-lts; do
    v=${tag%-*}
    d="$ENGINES/clickhouse-$v"
    [ -x "$d/clickhouse-common-static-$v/usr/bin/clickhouse" ] && continue
    mkdir -p "$d"
    fetch "https://github.com/ClickHouse/ClickHouse/releases/download/v$tag/clickhouse-common-static-$v-amd64.tgz" \
      "$d/clickhouse.tgz"
    tar xzf "$d/clickhouse.tgz" -C "$d"
    rm "$d/clickhouse.tgz"
  done
}

chroma() {
  local d="$ENGINES/chroma-1.5.9"
  [ -x "$d/bin/chroma" ] && return
  python3 -m venv "$d"
  "$d/bin/pip" install -q chromadb==1.5.9
}

pgvector() {
  local tmp
  apt-get install -y -q postgresql-16 postgresql-server-dev-16 build-essential
  systemctl disable --now postgresql 2>/dev/null || true   # the harness starts its own servers
  id pguser >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin pguser
  tmp=$(mktemp -d)
  fetch https://github.com/pgvector/pgvector/archive/refs/tags/v0.8.6.tar.gz "$tmp/pgvector.tar.gz"
  echo "10bf9938906e5d643bbc4a7eea104b6f57ba4898e5b76b20e60484ea1d5a7f8f  $tmp/pgvector.tar.gz" | sha256sum -c -
  tar xzf "$tmp/pgvector.tar.gz" -C "$tmp"
  make -C "$tmp/pgvector-0.8.6" PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config
  make -C "$tmp/pgvector-0.8.6" PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config install
  rm -rf "$tmp"
}

for target in "${@:-qdrant clickhouse chroma pgvector}"; do
  for t in $target; do
    case "$t" in
      qdrant | clickhouse | chroma | pgvector) "$t" ;;
      *) echo "unknown engine: $t" >&2; exit 2 ;;
    esac
  done
done
echo "engines ready in $ENGINES"
