"""Contract descriptors. Every rule records the sentence it rests on and its source.

The reference engine's documentation is the docstring of plimsoll/refengine/server.py.
"""

REF_DOC = "plimsoll/refengine/server.py, module docstring (reference engine contract)"

CONTRACTS = {
    "exact": {
        "metric": "squared_l2",
        "score": {"on": True, "source": "row", "formula": "expansion", "unit_roundoff": 2.0 ** -53,
                  "sentence": "scores are squared L2 distances from row data in float64 by the "
                              "expansion formula", "doc": REF_DOC},
        "order": {"follows": "reported", "sentence": "order follows the reported score", "doc": REF_DOC},
        "size": {"promised": True, "sentence": "result size min(k, n)", "doc": REF_DOC},
        "visibility": {"immediate": True, "sentence": "an acknowledged write is visible at once",
                       "doc": REF_DOC},
        "durability": {"acknowledged": True, "sentence": "an acknowledged write is durable",
                       "doc": REF_DOC},
        "errors_without_effect": [],
        "theta_max_ef": None,  # exact search: no search-effort parameter
    },
    "hnsw": {
        "metric": "squared_l2",
        "score": {"on": True, "source": "index", "formula": "direct", "unit_roundoff": 2.0 ** -24,
                  "sentence": "scores are squared L2 distances from index data in float32 by the "
                              "direct formula", "doc": REF_DOC},
        "order": {"follows": "reported", "sentence": "order follows the reported score", "doc": REF_DOC},
        "size": {"promised": False, "sentence": "result size not promised", "doc": REF_DOC},
        "visibility": {"immediate": True, "sentence": "an acknowledged write is visible at once",
                       "doc": REF_DOC},
        "durability": {"acknowledged": True, "sentence": "an acknowledged write is durable",
                       "doc": REF_DOC},
        "errors_without_effect": [],
        # No documented maximum: the default of Sec. III-D, ef = collection capacity.
        "theta_max_ef": 100_000,
    },
}

# ----------------------------------------------------------------------------- real engines
PGV_README = "pgvector 0.8.6 README.md (shipped with the release)"
PGV_SRC = "pgvector 0.8.6 src/vector.c and src/hnsw.c"
QD_POINTS = "https://qdrant.tech/documentation/manage-data/points/ (read 2026-10-12)"
QD_SEARCH = "https://qdrant.tech/documentation/search/search/ (read 2026-10-12)"
QD_UPSERT = "https://api.qdrant.tech/api-reference/points/upsert-points (read 2026-10-12)"

CONTRACTS["pgvector_hnsw"] = {
    "metric": "squared_l2",
    "score": {"on": True, "source": "row", "formula": "direct", "unit_roundoff": 2.0 ** -24,
              "reported": "sqrt", "sqrt_unit_roundoff": 2.0 ** -53,
              "sentence": "`<->` - L2 distance", "doc": PGV_README,
              "implementation": "squared differences summed in float, sqrt in double, returned as "
                                "float8 (VectorL2SquaredDistance, l2_distance)", "impl_source": PGV_SRC},
    "order": {"follows": "reported",
              "sentence": "SQL ORDER BY; 'Relaxed allows results to be slightly out of order by distance' "
                          "applies only to iterative scans, which are off", "doc": PGV_README},
    "size": {"promised": False,
             "sentence": "There may be even less results due to dead tuples or filtering conditions "
                         "in the query.", "doc": PGV_README},
    "visibility": {"immediate": True, "sentence": "ACID compliance (read committed: a committed write "
                                                   "is visible to later statements)", "doc": PGV_README},
    "durability": {"acknowledged": True,
                   "sentence": "ACID compliance; pgvector uses the write-ahead log (WAL)", "doc": PGV_README},
    "errors_without_effect": ["an error raised inside a transaction rolls it back"],
    "theta_max_ef": 1000,  # hnsw.ef_search: "Valid range is 1..1000." (GUC description, hnsw.c)
}

CONTRACTS["qdrant_hnsw"] = {
    "metric": "squared_l2",
    "score": {"on": True, "source": "storage", "formula": "direct", "unit_roundoff": 2.0 ** -24,
              "reported": "sqrt", "sqrt_unit_roundoff": 2.0 ** -24,
              "sentence": "Euclidean distance (`Euclid`); no formula is documented", "doc": QD_SEARCH,
              "implementation": "sum of squared differences in float32 (FMA on AVX), then sqrt(abs(score)) "
                                "in float32",
              "impl_source": "Qdrant v1.19.2 lib/segment/src/spaces/simple.rs (EuclidMetric, euclid_similarity, "
                             "postprocess) and simple_avx.rs (euclid_similarity_avx)"},
    "order": {"follows": "reported",
              "sentence": "The `result` contains ordered by `score` list of found point ids.", "doc": QD_SEARCH},
    "size": {"promised": False,
             "sentence": "'Returns `limit` vectors' is listed for dense queries without saying what happens "
                         "with fewer reachable points; treated as not promised (contract gap)", "doc": QD_SEARCH},
    "visibility": {"immediate": True, "sentence": "wait: If true, wait for changes to actually happen",
                   "doc": QD_UPSERT},
    "durability": {"acknowledged": True,
                   "sentence": "At the first stage, the operation is written to the Write-ahead-log. After "
                               "this moment, the service will not lose the data, even if the machine loses "
                               "power supply.", "doc": QD_POINTS},
    "repeatability": {"sentence": "Because HNSW search is approximate, the ranking of results can shift "
                                  "slightly between requests.", "doc": QD_SEARCH},
    "errors_without_effect": [],
    "theta_max_ef": None,  # no documented maximum: the default of Sec. III-D, set per configuration
}

CH_DOC = ("ClickHouse docs, docs/reference/engines/table-engines/mergetree-family/annindexes.mdx and "
          "docs/reference/statements/{delete,update}.mdx (ClickHouse repository, commit 360bb0de, 2026-10-12)")
CH_DOC_REL = ("ClickHouse docs at tag v26.9.13.15-stable (commit 4b131779): annindexes.mdx and "
              "statements/{delete,update}.mdx")

CONTRACTS["clickhouse_hnsw_f32"] = {
    "metric": "squared_l2",
    "score": {"on": True, "source": "index", "formula": "direct", "unit_roundoff": 2.0 ** -24,
              "reported": "sqrt", "sqrt_unit_roundoff": 2.0 ** -24,
              "sentence": "`L2Distance`, the Euclidean distance; without rescoring (the default) the index "
                          "returns the most similar vectors and their distances directly", "doc": CH_DOC_REL,
              "implementation": "EXPLAIN shows CAST(sqrt(_distance) AS Float32); quantization 'f32'; the "
                                "index's L2sq is the sum of squared differences accumulated in float32",
              "impl_source": "ClickHouse v26.9.13.15-stable links usearch 6bd13d4c with SimSIMD e81e1c2d on x86-64 "
                             "(contrib/usearch-cmake/CMakeLists.txt); usearch index_plugins.hpp configures SimSIMD "
                             "l2sq (fallback metric_l2sq_gt); SimSIMD spatial.h simsimd_l2sq_f32_{haswell,skylake}"},
    "order": {"follows": "reported", "sentence": "SQL ORDER BY ... ASC", "doc": CH_DOC_REL},
    "size": {"promised": "unfiltered",
             "sentence": "SQL ORDER BY ... LIMIT N; the only documented exception is post-filtering: "
                         "'Post-filtering has the general problem that it may return less than the number of "
                         "rows requested in the LIMIT <N> clause.' So min(k, n) rows are promised for "
                         "unfiltered queries only", "doc": CH_DOC_REL},
    "visibility": {"immediate": True,
                   "sentence": "By default, DELETE statements wait until marking the rows as deleted is completed "
                               "before returning; lightweight UPDATE makes updated values immediately visible",
                   "doc": CH_DOC_REL},
    # The exact-search guarantee for parts with deleted rows appears in the main-branch
    # documentation (commit 360bb0de) but not in the documentation of release 26.9.13.15
    # (tag commit 4b131779), so it is not part of this release's contract.
    "exact_after_delete": False,
    "durability": {"acknowledged": True,
                   "sentence": "not stated; an INSERT returns after its part is written (fsync_after_insert is off "
                               "by default), so it is expected to survive a process kill but not a power loss",
                   "doc": CH_DOC_REL},
    "errors_without_effect": [],
    "theta_max_ef": None,  # hnsw_candidate_list_size_for_search has no documented maximum
}

CHROMA_DOC = ("Chroma documentation at tag 1.5.9 (commit 11f3c743), docs/mintlify: docs/collections/"
              "{configure,add-data,update-data,delete-data}.mdx, docs/querying-collections/query-and-get.mdx, "
              "docs/overview/troubleshooting.mdx, reference/architecture/overview.mdx")
CHROMA_SRC = ("chroma-core/hnswlib commit 6868102b (pinned in Chroma 1.5.9 Cargo.lock), hnswlib/space_l2.h: "
              "L2Sqr sums (a_i - b_i)^2 in float")

CONTRACTS["chroma_hnsw"] = {
    "metric": "squared_l2",
    "score": {"on": True, "source": "index", "formula": "direct", "unit_roundoff": 2.0 ** -24,
              "sentence": "space l2: 'Squared L2', d = sum (A_i - B_i)^2", "doc": CHROMA_DOC,
              "implementation": "L2Sqr, float accumulation", "impl_source": CHROMA_SRC},
    "order": {"follows": "reported",
              "sentence": "'The Query API enables nearest-neighbor similarity search'; the order of the "
                          "returned list is not stated, so ascending distance is assumed (contract gap)",
              "doc": CHROMA_DOC},
    "size": {"promised": "unfiltered",
             "sentence": "'By default, Chroma will return 10 results per input query. You can modify this "
                         "number using the n_results argument'; the troubleshooting page says that failing to "
                         "retrieve the requested number of results is an error. Nothing is said for queries "
                         "with a where filter, so min(k, n) is promised for unfiltered queries only",
             "doc": CHROMA_DOC},
    "visibility": {"immediate": True,
                   "sentence": "not stated; the single-node server serves queries from the same process that "
                               "acknowledged the write (assumed immediate, contract gap)", "doc": CHROMA_DOC},
    "durability": {"acknowledged": True,
                   "sentence": "'Chroma delegates as much as possible to durable, well-understood subsystems "
                               "such as SQLite'; a process kill is assumed not to lose an acknowledged write "
                               "(contract gap: no explicit statement)", "doc": CHROMA_DOC},
    "errors_without_effect": ["update of an absent id: 'an error will be logged and the update will be ignored'"],
    "theta_max_ef": 20_000,   # no documented maximum; collection capacity (campaigns). See adapters/chroma_f.py
}
