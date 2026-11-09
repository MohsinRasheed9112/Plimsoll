# Plimsoll

Code for the paper *Plimsoll: Testing Vector Databases for Wrong Answers Hidden by
Approximation* (under review).

Plimsoll decides whether an answer of a vector database is wrong or merely approximate.
It runs workloads with writes, maintenance operations and process kills against the
database and judges every answer with exact invariants, witness queries and
state-equivalence tests.

## Setup

Ubuntu 24.04 (x86-64) and Python 3.13.

```bash
pip install -r requirements.txt
sudo scripts/setup_engines.sh     # Qdrant, ClickHouse, Chroma, PostgreSQL 16 with pgvector
python scripts/get_data.py        # SIFT-128 and Fashion-MNIST from ANN-Benchmarks
(cd code && python -m pytest -q tests)
```

## Run

Run the experiments as root: the harness starts PostgreSQL as the unprivileged user pguser.

```bash
sudo PYTHON="$(command -v python3)" scripts/run_all.sh local   # 2 vCPUs, 8 GB
sudo PYTHON="$(command -v python3)" scripts/run_all.sh cloud   # 32 vCPUs: SIFT 1M, ground truth at scale
scripts/run_all.sh report                                      # every reported number
```

Results are written to `results/`.

## License

Apache-2.0
