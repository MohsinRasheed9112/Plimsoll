#!/usr/bin/env bash
# Runs the experiments of Sec. V with the seeds and settings of the paper.
#
#   scripts/run_all.sh local    2 vCPUs, 8 GB: Experiments 1, 2, 3, 4 (SIFT 100K), 5, 6, 7, 8,
#                               and the directed reproductions of Sec. V-C
#   scripts/run_all.sh cloud    32 vCPUs, 128 GiB: Experiment 4 on SIFT 1M and ground truth at
#                               scale (Experiment 8)
#   scripts/run_all.sh report   collects every reported number into results/report.json
#
# Run as root (the pgvector adapter runs PostgreSQL as the user pguser), after
# scripts/setup_engines.sh and scripts/get_data.py. Every driver keeps finished runs, so
# an interrupted group can be started again. PROCS (default 12) sets the number of
# parallel Experiment 4 processes in the cloud group.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PYTHON:-python3}
LOGS=$ROOT/results/logs
mkdir -p "$LOGS"

camp() {  # camp <engine> <first_seed> <runs> <tag> [options]
  local e=$1 s=$2 n=$3 tag=$4
  shift 4
  (cd code && "$PY" experiments/campaign.py "$e" "$s" "$n" --tag "$tag" "$@") >> "$LOGS/campaigns.log" 2>&1
}

exp1() {      # Sec. V-B, Table I
  (cd code && PLIMSOLL_WORK=/var/tmp/plimsoll/exact "$PY" experiments/exp1.py exact) > "$LOGS/exp1_exact.log" 2>&1 &
  (cd code && PLIMSOLL_WORK=/var/tmp/plimsoll/hnsw "$PY" experiments/exp1.py hnsw) > "$LOGS/exp1_hnsw.log" 2>&1
  wait
  (cd code && "$PY" experiments/exp1_summarize.py && "$PY" experiments/exp1_report.py) > "$LOGS/exp1_summary.log"
  for e in pgvector qdrant; do
    (cd code && "$PY" experiments/exp1r.py "$e" && "$PY" experiments/exp1r_summarize.py "$e") > "$LOGS/exp1r_$e.log" 2>&1
  done
}

exp2() {      # Sec. V-C3 and V-C4: campaigns on older releases
  camp qdrant-1.18.3 100 9 exp2 --qdrant-full-scan
  camp clickhouse-26.3.45.3 550 3 exp2
}

exp3() {      # Sec. V-C6, Table III: campaigns on recent releases
  (cd code && "$PY" experiments/chroma_calib.py) > "$LOGS/chroma_calib.log" 2>&1
  camp pgvector 400 20 exp3
  camp qdrant 300 28 exp3 --qdrant-full-scan
  camp qdrant 360 20 exp3h
  camp clickhouse 500 20 exp3
  camp chroma 900 20 exp3
  camp pgvector 750 5 exp3o --op-concurrent 5
  camp qdrant 650 5 exp3o --op-concurrent 5
  camp clickhouse 850 5 exp3o --op-concurrent 5
  camp chroma 970 5 exp3o --op-concurrent 5
  camp pgvector 700 5 exp3c --concurrent 5
  camp qdrant 600 5 exp3c --concurrent 5
  camp clickhouse 800 5 exp3c --concurrent 5
  camp chroma 950 5 exp3c --concurrent 5
}

exp4_local() {   # Sec. V-D, Table IV (SIFT 100K)
  (cd code && "$PY" experiments/exp4.py A calib && "$PY" experiments/exp4.py A envelopes \
    && "$PY" experiments/exp4.py A trials && "$PY" experiments/exp4_summarize.py A) > "$LOGS/exp4_A.log" 2>&1
}

exp5() {      # Sec. V-C5, Table II
  (cd code && "$PY" experiments/exp5.py clickhouse 26.9.13.15 5700 10 \
    && "$PY" experiments/exp5.py clickhouse 26.3.45.3 5800 10 \
    && "$PY" experiments/exp5.py qdrant 1.19.2 5600 20 \
    && "$PY" experiments/exp5.py qdrant 1.18.3 5500 20 \
    && "$PY" experiments/exp5.py summarize) > "$LOGS/exp5.log" 2>&1
}

repro() {     # Sec. V-C: directed runs
  (
    cd code
    for v in 1.18.3 1.19.0 1.19.2; do "$PY" experiments/repro_qdrant_lost_points.py "$v" 10; done
    for v in 26.9.13.15 26.3.45.3; do "$PY" experiments/repro_clickhouse_117286.py "$v"; done
    "$PY" experiments/repro_clickhouse_concurrent_update.py attr_index 10
    "$PY" experiments/repro_clickhouse_concurrent_update.py none 5
    "$PY" experiments/repro_clickhouse_concurrent_update.py add_index 5
    "$PY" experiments/repro_clickhouse_concurrent_update.py materialize_cidx 5
    "$PY" experiments/repro_clickhouse_concurrent_update.py materialize 2
    "$PY" experiments/repro_clickhouse_concurrent_update.py optimize 2
    for s in 1 2 3 4 5; do "$PY" experiments/repro_qdrant_filter.py --data sift --n 15000 --full-scan-kb 1000 --seed "$s"; done
    for s in 1 2 3; do "$PY" experiments/repro_qdrant_filter.py --data sift --n 15000 --full-scan-kb 1300 --seed "$s"; done
    for s in 1 2; do "$PY" experiments/repro_qdrant_filter.py --data fmnist --n 59000 --full-scan-kb 0 --seed "$s"; done
  ) > "$LOGS/repro.log" 2>&1
}

exp7() {      # Sec. V-F1, Table VI: no kills, two datasets
  local e
  for e in pgvector qdrant clickhouse chroma; do
    camp "$e" 1000 5 exp7_sift --no-kill
    camp "$e" 1100 5 exp7_fmnist --data fmnist --no-kill
  done
  (cd code && "$PY" experiments/exp7_summarize.py) > "$LOGS/exp7_summary.log" 2>&1
}

exp6() {      # Sec. V-E, Table V: ablations over the runs above
  (cd code && "$PY" experiments/exp6_ablations.py) > "$LOGS/exp6.log" 2>&1
}

exp8_local() {   # Sec. V-F2: cost, on an otherwise idle machine
  (cd code && "$PY" experiments/exp8.py oracles gt capacity pgvector qdrant clickhouse) > "$LOGS/exp8.log" 2>&1
}

cloud() {     # Sec. V-D1 (SIFT 1M) and Sec. V-F3 (ground truth at scale)
  local procs=${PROCS:-12}
  "$PY" scripts/get_data.py sift gist
  cd code
  seq 5000 5019 | xargs -P "$procs" -I{} sh -c "\"$PY\" experiments/exp4.py B calib {} \$(({}+1))" >> "$LOGS/exp4_B.log" 2>&1
  "$PY" experiments/exp4.py B envelopes >> "$LOGS/exp4_B.log" 2>&1
  seq 5100 5199 | xargs -P "$procs" -I{} sh -c "\"$PY\" experiments/exp4.py B trials {} \$(({}+1))" >> "$LOGS/exp4_B.log" 2>&1
  "$PY" experiments/exp4_summarize.py B >> "$LOGS/exp4_B.log" 2>&1
  "$PY" experiments/exp8.py gt_scale > "$LOGS/exp8_gt_scale.log" 2>&1
}

case "${1:-}" in
  local)
    exp1; exp2; exp3; exp4_local; exp5; repro; exp7; exp6; exp8_local ;;
  cloud)
    cloud ;;
  report)
    "$PY" scripts/report.py ;;
  *)
    echo "usage: $0 {local|cloud|report}" >&2
    exit 2 ;;
esac
