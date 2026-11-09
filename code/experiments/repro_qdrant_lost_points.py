"""Directed reproduction of the Qdrant lost-points finding (triage and Experiment 2).

Scenario (found by campaign seed 108 on Qdrant 1.18.3, then minimized by hand):
  1. Load 15,000 SIFT points with an integer payload; wait until the collection is green.
  2. Delete 2,018 points, then 123 more (acknowledged).
  3. Trigger the optimizers (empty optimizer update).
  4. set_payload on 1,366 live points in one batch (acknowledged, wait = true).
  5. Wait until the collection is green again (the optimizer has finished), then SIGKILL.
  6. Restart and look up every point: an acknowledged live point that is absent is a
     violation of the EI lookup rule; any deleted point that is present is one too.

Usage: python experiments/repro_qdrant_lost_points.py <version> <trials>
Writes results/repro/qdrant_lost_points/<version>.json, with the server's warning and
error lines from the restart of every trial that lost points.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
from qdrant_client import QdrantClient, models

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from plimsoll import paths  # noqa: E402


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main(version, trials):
    binary = paths.qdrant_binary(version)
    X = np.load(os.path.join(ROOT, "data", "sift100k_base.npy"))[:15000]
    work = "/var/tmp/plim_qd/repro_%s" % version
    out = dict(version=version, binary=binary, trials=[])
    ver = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout.strip()
    out["binary_version"] = ver
    rng = np.random.default_rng(2026)

    def launch(env):
        p = subprocess.Popen([binary], cwd=work, env=env, stdout=open(work + "/log", "a"),
                             stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            try:
                if urllib.request.urlopen("http://127.0.0.1:%s/readyz" % env["QDRANT__SERVICE__HTTP_PORT"],
                                          timeout=1).status == 200:
                    break
            except Exception:
                time.sleep(0.02)
            if time.time() - t0 > 120:
                raise RuntimeError("not ready")
        return p, QdrantClient(host="127.0.0.1", port=int(env["QDRANT__SERVICE__HTTP_PORT"]),
                               grpc_port=int(env["QDRANT__SERVICE__GRPC_PORT"]), prefer_grpc=True, timeout=120)

    def wait_green(c, limit=60):
        t0 = time.time()
        while c.get_collection("items").status != models.CollectionStatus.GREEN and time.time() - t0 < limit:
            time.sleep(0.05)
        return str(c.get_collection("items").status)

    for t in range(trials):
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=work + "/storage", QDRANT__SERVICE__HOST="127.0.0.1",
                   QDRANT__SERVICE__HTTP_PORT=str(port()), QDRANT__SERVICE__GRPC_PORT=str(port()),
                   QDRANT__TELEMETRY_DISABLED="true", QDRANT__LOG_LEVEL="INFO")
        p, c = launch(env)
        c.create_collection("items", vectors_config=models.VectorParams(size=128, distance=models.Distance.EUCLID))
        c.update_collection("items", optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000))
        ids = list(range(15000))
        for b in range(0, 15000, 1000):
            c.upsert("items", points=models.Batch(ids=ids[b:b + 1000], vectors=X[b:b + 1000].tolist(),
                                                  payloads=[{"cat": int(a)} for a in rng.integers(0, 100, 1000)]),
                     wait=True)
        wait_green(c)
        live = set(ids)
        for n in (2018, 123):
            d = sorted(int(i) for i in rng.choice(sorted(live), n, replace=False))
            c.delete("items", points_selector=models.PointIdsList(points=d), wait=True)
            live -= set(d)
        c.update_collection("items", optimizers_config=models.OptimizersConfigDiff())
        tgt = sorted(int(i) for i in rng.choice(sorted(live), 1366, replace=False))
        groups = {}
        for i in tgt:
            groups.setdefault(int(rng.integers(0, 100)), []).append(i)
        c.batch_update_points("items", update_operations=[models.SetPayloadOperation(
            set_payload=models.SetPayload(payload={"cat": a}, points=g)) for a, g in sorted(groups.items())],
            wait=True)
        status = wait_green(c)
        log_before = len(open(work + "/log").read().splitlines())
        os.kill(p.pid, signal.SIGKILL)
        p.wait()
        c.close()
        t0 = time.time()
        p, c = launch(env)
        ready = time.time() - t0
        got = set()
        for b in range(0, 15000, 1000):
            got |= {int(r.id) for r in c.retrieve("items", ids=ids[b:b + 1000], with_vectors=False,
                                                  with_payload=False)}
        missing = sorted(live - got)
        extra = sorted(got - live)
        c.close()
        p.send_signal(signal.SIGTERM)
        p.wait()
        lines = open(work + "/log").read().splitlines()[log_before:]
        bad_lines = [l for l in lines if (" ERROR " in l or " WARN " in l) and "Config file not found" not in l
                     and "Static content" not in l]
        out["trials"].append(dict(trial=t, status_before_kill=status, restart_ready_secs=round(ready, 2),
                                  live=len(live), missing=len(missing), missing_among_updated=len(set(missing) & set(tgt)),
                                  present_but_deleted=len(extra), log_warn_error_lines=len(bad_lines),
                                  log_examples=bad_lines[:4]))
        print(json.dumps(out["trials"][-1])[:300], flush=True)
    out["summary"] = dict(trials=trials, trials_with_loss=sum(x["missing"] > 0 for x in out["trials"]),
                          max_missing=max(x["missing"] for x in out["trials"]))
    d = os.path.join(ROOT, "results", "repro", "qdrant_lost_points")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "%s.json" % version), "w") as f:
        json.dump(out, f, indent=1)
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
