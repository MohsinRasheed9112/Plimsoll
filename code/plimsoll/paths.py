"""Locations of the data, the results, scratch space and the engine binaries.

The repository layout is fixed: data/ and results/ sit next to code/. Engine binaries
are found under PLIMSOLL_ENGINES (default: engines/ in the repository), in the layout
that scripts/setup_engines.sh creates. Every location can be overridden by an
environment variable.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
RESULTS = os.path.join(REPO, "results")
ENGINES = os.environ.get("PLIMSOLL_ENGINES", os.path.join(REPO, "engines"))

# Scratch space for engine data directories (local disk, deleted after each run).
WORK = os.environ.get("PLIMSOLL_WORK", "/var/tmp/plimsoll")

# PostgreSQL 16 server binaries (Debian/Ubuntu package layout) with pgvector installed,
# and the unprivileged OS user that runs the server (the harness itself runs as root).
PG_BIN = os.environ.get("PLIMSOLL_PG_BIN", "/usr/lib/postgresql/16/bin")
PG_USER = os.environ.get("PLIMSOLL_PG_USER", "pguser")

QDRANT_VERSION = "1.19.2"
CLICKHOUSE_VERSION = "26.9.13.15"
CHROMA_VERSION = "1.5.9"


def qdrant_binary(version=QDRANT_VERSION):
    """Official Qdrant release binary (x86_64-unknown-linux-gnu)."""
    return os.path.join(ENGINES, "qdrant-%s" % version, "qdrant")


def clickhouse_binary(version=CLICKHOUSE_VERSION):
    """Official ClickHouse static binary, as unpacked from clickhouse-common-static."""
    return os.path.join(ENGINES, "clickhouse-%s" % version, "clickhouse-common-static-%s" % version,
                        "usr", "bin", "clickhouse")


def chroma_binary(version=CHROMA_VERSION):
    """The chroma command of a virtual environment with chromadb installed."""
    return os.environ.get("PLIMSOLL_CHROMA", os.path.join(ENGINES, "chroma-%s" % version, "bin", "chroma"))


def work_dir(name):
    """A scratch directory under WORK."""
    return os.path.join(WORK, name)
