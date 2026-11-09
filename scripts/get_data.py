"""Download the datasets from ANN-Benchmarks and write the .npy files the experiments use.

  sift     SIFT-128: data/sift100k_{base,queries}.npy (the first 100,000 base vectors
           and the first 1,000 test queries) and data/sift1m_{base_part0..3,queries}.npy
  fmnist   Fashion-MNIST-784: data/fmnist_{base,queries}.npy (train and test, float32)
  gist     GIST-960 (Experiment 8, ground truth at scale): data/gist-960-euclidean.hdf5,
           read as is; its SHA-256 and shapes go to data/gist_provenance.json

Every source file and every output is checked against data/datasets_provenance.json and
data/sift100k_provenance.json, so the files are byte-identical to those of the paper's runs.
Usage (from the repository root): python scripts/get_data.py [sift] [fmnist] [gist]
(default: sift fmnist)
"""
import hashlib
import json
import os
import sys
import urllib.request

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SOURCE = "http://ann-benchmarks.com/%s.hdf5"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def provenance():
    with open(os.path.join(DATA, "datasets_provenance.json")) as f:
        prov = json.load(f)
    with open(os.path.join(DATA, "sift100k_provenance.json")) as f:
        p100 = json.load(f)
    prov["data/sift100k_base.npy"] = p100["base_sha256"]
    prov["data/sift100k_queries.npy"] = p100["queries_sha256"]
    return prov


def fetch(name, expected_sha=None):
    path = os.path.join(DATA, name + ".hdf5")
    if not os.path.exists(path):
        print("downloading", SOURCE % name, flush=True)
        urllib.request.urlretrieve(SOURCE % name, path + ".part")
        os.replace(path + ".part", path)
    if expected_sha is not None and sha256(path) != expected_sha:
        sys.exit("checksum mismatch for %s" % path)
    return path


def write_checked(outputs, prov):
    for rel, arr in outputs.items():
        path = os.path.join(ROOT, rel)
        np.save(path, arr)
        if sha256(path) != prov[rel]:
            sys.exit("checksum mismatch for " + rel)
        print("ok", rel, flush=True)


def sift(prov):
    with h5py.File(fetch("sift-128-euclidean", prov["sift_source_sha256"]), "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
    assert train.shape == (1_000_000, 128) and test.shape == (10_000, 128)
    out = {"data/sift1m_base_part%d.npy" % i: np.ascontiguousarray(train[i * 250_000:(i + 1) * 250_000])
           for i in range(4)}
    out["data/sift1m_queries.npy"] = np.ascontiguousarray(test)
    out["data/sift100k_base.npy"] = np.ascontiguousarray(train[:100_000])
    out["data/sift100k_queries.npy"] = np.ascontiguousarray(test[:1_000])
    write_checked(out, prov)


def fmnist(prov):
    with h5py.File(fetch("fashion-mnist-784-euclidean", prov["fmnist_source_sha256"]), "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
    assert train.shape == (60_000, 784) and test.shape == (10_000, 784)
    write_checked({"data/fmnist_base.npy": np.ascontiguousarray(train),
                   "data/fmnist_queries.npy": np.ascontiguousarray(test)}, prov)


def gist(prov):
    path = fetch("gist-960-euclidean")
    with h5py.File(path, "r") as f:
        shapes = {k: list(f[k].shape) for k in ("train", "test")}
    assert shapes["train"] == [1_000_000, 960] and shapes["test"][1] == 960, shapes
    with open(os.path.join(DATA, "gist_provenance.json"), "w") as f:
        json.dump(dict(url=SOURCE % "gist-960-euclidean", sha256=sha256(path), shapes=shapes), f, indent=1)
    print("ok data/gist-960-euclidean.hdf5", flush=True)


def main(args):
    os.makedirs(DATA, exist_ok=True)
    prov = provenance()
    for name in args or ["sift", "fmnist"]:
        {"sift": sift, "fmnist": fmnist, "gist": gist}[name](prov)


if __name__ == "__main__":
    main(sys.argv[1:])
