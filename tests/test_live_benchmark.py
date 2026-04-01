"""Engram-Memory SDK — Live Benchmark Suite with Time Profiling & Publication Metrics.

Runs against a **real** Neo4j + LLM stack and captures every timing, accuracy,
and scalability metric needed for a Towards-Data-Science-grade writeup.

**Opt-in**:  ENGRAM_MEMORY_LIVE_TESTS=1  (env / .env / engram_memory/.env)
**Run**:     pytest tests/test_live_benchmark.py -m live -v -s
**Output**:  benchmarks/benchmark_report.json  (auto-created)

Metrics captured
────────────────
• Latency:  ingest p50/p95/p99/max, recall p50/p95/p99/max, search p50/p95/p99
• Throughput:  ingest ops/sec, recall ops/sec
• Accuracy:  Precision@K, Recall@K, MRR, F1 for K ∈ {1,3,5,10}
• Semantic discrimination:  cross-topic contamination rate
• Scaling:  ingest latency vs corpus size (1→N curve)
• Graph fidelity:  nodes/relationships created, node-per-ingest ratio
• User isolation:  cross-user leakage check (pass/fail)
• Health:  Neo4j, LLM, embedder, vector index status
• System:  Python version, platform, timestamp, SDK version
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from engram_memory.client import AsyncMemoryClient
from engram_memory.config import Config

# Per-1K-token pricing used for cost estimation. Adjust for your model.
TOKEN_PRICE_PER_1K: dict[str, float] = {
    "prompt": 0.06,       # $/1K tokens — Azure GPT-4-32k prompt
    "completion": 0.12,   # $/1K tokens — Azure GPT-4-32k completion
}

# ── gate ─────────────────────────────────────────────────────────────────────
LIVE_ENABLED = os.environ.get("ENGRAM_MEMORY_LIVE_TESTS", "").strip().lower() in (
    "1", "true", "yes", "on",
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_ENABLED,
        reason="Set ENGRAM_MEMORY_LIVE_TESTS=1 to run live benchmark",
    ),
]

# ── output dir ───────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
REPORT_PATH = BENCHMARK_DIR / "benchmark_report.json"

# ── corpus ───────────────────────────────────────────────────────────────────
# Realistic, diverse knowledge-graph material across 6 domains.
# Each entry: (text, reference_id_prefix, domain, expected_entities)
CORPUS = [
    # ── Engineering / DevOps ──
    (
        "Jordan Vega is a site-reliability engineer at NovaTech who migrated "
        "the payment microservice from PostgreSQL 14 to CockroachDB in Q1 2025, "
        "reducing p99 latency from 320 ms to 85 ms.",
        "eng", "engineering",
        ["Jordan Vega", "NovaTech", "CockroachDB", "PostgreSQL"],
    ),
    (
        "The Kubernetes cluster kube-prod-us-east runs 480 pods across 12 nodes "
        "with Istio service mesh and Prometheus monitoring, owned by the platform "
        "team lead Priya Sharma.",
        "eng", "engineering",
        ["kube-prod-us-east", "Priya Sharma", "Istio", "Prometheus"],
    ),
    (
        "CI/CD pipeline phoenix-deploy uses GitHub Actions with a 14-minute "
        "build time. The team targets sub-10-minute builds by Q3 2025 via "
        "Bazel remote caching.",
        "eng", "engineering",
        ["phoenix-deploy", "GitHub Actions", "Bazel"],
    ),
    # ── Healthcare ──
    (
        "Dr. Amara Okafor leads the cardiology department at Lakeside General "
        "Hospital. She published a landmark study on SGLT2-inhibitor efficacy "
        "in heart failure patients over age 65, cited 340 times.",
        "health", "healthcare",
        ["Dr. Amara Okafor", "Lakeside General Hospital", "SGLT2-inhibitor"],
    ),
    (
        "Patient intake records show a 23% increase in Type-2 diabetes referrals "
        "at Lakeside General since January 2025, prompting a new tele-health "
        "screening program led by nurse practitioner Luis Mendez.",
        "health", "healthcare",
        ["Luis Mendez", "Lakeside General", "Type-2 diabetes"],
    ),
    # ── Finance ──
    (
        "Quant analyst Mei-Lin Chang developed the Sigma-7 volatility model at "
        "Bridgewater Capital, achieving a 12.4% Sharpe ratio improvement over "
        "the prior GARCH-based approach on S&P 500 options data.",
        "fin", "finance",
        ["Mei-Lin Chang", "Sigma-7", "Bridgewater Capital", "GARCH"],
    ),
    (
        "The FX trading desk processed $4.2B in notional volume during March 2025 "
        "with a 0.3 bps average spread. Head of desk: Raj Patel.",
        "fin", "finance",
        ["Raj Patel", "FX trading desk"],
    ),
    # ── Education ──
    (
        "Professor Elena Vasquez teaches Advanced Machine Learning (CS-6140) at "
        "MIT, using a flipped-classroom model. Student satisfaction rose from "
        "3.8 to 4.6 / 5.0 after adopting interactive Jupyter notebooks in Fall 2024.",
        "edu", "education",
        ["Elena Vasquez", "CS-6140", "MIT"],
    ),
    # ── Facilities / Operations ──
    (
        "Building A visitors must use parking garage PARK-A on level 2 and "
        "collect a badge from reception desk R1 before 09:00. After-hours "
        "entry requires security escort from guard station GS-North.",
        "ops", "operations",
        ["PARK-A", "R1", "GS-North"],
    ),
    (
        "The cafeteria on floor 1 mezzanine serves a fresh salad bar on "
        "Tuesdays and Thursdays, with a rotating hot-food menu curated by "
        "head chef Marco Bianchi. Capacity: 120 seats.",
        "ops", "operations",
        ["cafeteria", "Marco Bianchi"],
    ),
    # ── Legal / Compliance ──
    (
        "Chief compliance officer Naomi Tanaka issued directive COMP-2025-003 "
        "requiring all customer PII to be encrypted at rest with AES-256 and "
        "rotated every 90 days, effective 1 April 2025.",
        "legal", "compliance",
        ["Naomi Tanaka", "COMP-2025-003", "AES-256"],
    ),
    (
        "The GDPR audit completed on 15 March 2025 by external firm DataShield "
        "found 2 minor non-conformities in the data-retention schedule; "
        "remediation deadline is 30 June 2025.",
        "legal", "compliance",
        ["DataShield", "GDPR"],
    ),
]

# ── recall test bank ─────────────────────────────────────────────────────────
# Each: (query, expected_domain, expected_entity_substrings)
RECALL_QUERIES = [
    ("Who migrated the payment service to CockroachDB?",
     "engineering", ["Jordan Vega", "CockroachDB"]),
    ("What monitoring does the Kubernetes cluster use?",
     "engineering", ["Prometheus", "kube-prod-us-east"]),
    ("How long does the phoenix-deploy CI pipeline take?",
     "engineering", ["phoenix-deploy", "14"]),
    ("Who leads cardiology at Lakeside General Hospital?",
     "healthcare", ["Amara Okafor", "Lakeside General"]),
    ("What caused the diabetes referral increase?",
     "healthcare", ["Type-2 diabetes", "Luis Mendez"]),
    ("What volatility model did Mei-Lin Chang develop?",
     "finance", ["Sigma-7", "Mei-Lin Chang"]),
    ("Who heads the FX trading desk?",
     "finance", ["Raj Patel"]),
    ("Which professor teaches Advanced Machine Learning at MIT?",
     "education", ["Elena Vasquez", "CS-6140"]),
    ("Where do visitors park for Building A?",
     "operations", ["PARK-A", "level 2"]),
    ("What food is available in the cafeteria on Thursdays?",
     "operations", ["salad bar", "Marco Bianchi"]),
    ("What does directive COMP-2025-003 require?",
     "compliance", ["Naomi Tanaka", "AES-256"]),
    ("What were the GDPR audit findings?",
     "compliance", ["DataShield", "non-conformit"]),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])


def _latency_stats(timings: list[float]) -> dict[str, float]:
    if not timings:
        return {}
    s = sorted(timings)
    return {
        "count": len(s),
        "mean_ms": round(statistics.mean(s) * 1000, 2),
        "median_ms": round(statistics.median(s) * 1000, 2),
        "p50_ms": round(_percentile(s, 50) * 1000, 2),
        "p95_ms": round(_percentile(s, 95) * 1000, 2),
        "p99_ms": round(_percentile(s, 99) * 1000, 2),
        "max_ms": round(max(s) * 1000, 2),
        "min_ms": round(min(s) * 1000, 2),
        "stddev_ms": round(statistics.stdev(s) * 1000, 2) if len(s) > 1 else 0.0,
        "total_sec": round(sum(s), 3),
    }


def _blob(nodes) -> str:
    parts = []
    for n in nodes:
        parts.append(n.summary or "")
        parts.append(str(n.properties or {}))
        parts.append(n.label or "")
    return " ".join(parts).lower()


def _entity_hit(blob: str, entity: str) -> bool:
    return entity.lower() in blob


_CLEANUP_CYPHER = "MATCH (n) WHERE n.userId = $userId DETACH DELETE n"


# ── THE BENCHMARK (single test, full pipeline) ──────────────────────────────

@pytest.mark.asyncio
async def test_full_benchmark():
    """End-to-end benchmark: health → ingest → recall → search → discrimination
    → isolation → scaling → graph snapshot → save report."""

    required = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
    missing = [k for k in required if not os.environ.get(k, "").strip()]
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")

    cfg = Config()
    bench_user = f"bench-{uuid.uuid4().hex[:12]}"

    metrics: dict[str, Any] = {
        "meta": {
            "suite": "engram_memory live benchmark",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sdk_version": None,
        },
        "health": {},
        "ingest": {},
        "recall": {},
        "search": {},
        "accuracy": {},
        "scaling": {},
        "graph_stats": {},
        "user_isolation": {},
        "discrimination": {},
    }

    client = AsyncMemoryClient(cfg)
    await client.__aenter__()

    try:
        from engram_memory._version import __version__
        metrics["meta"]["sdk_version"] = __version__
    except Exception:
        pass

    errors: list[str] = []

    try:
        # ════════════════════════════════════════════════════════════════
        # PHASE 1 — HEALTH CHECK
        # ════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  PHASE 1: HEALTH CHECK")
        print("=" * 70)

        t0 = time.perf_counter()
        status = await client.health_check()
        dt = time.perf_counter() - t0

        metrics["health"] = {
            "neo4j_connected": status.neo4j_connected,
            "llm_reachable": status.llm_reachable,
            "embedding_model_loaded": status.embedding_model_loaded,
            "vector_index_exists": status.vector_index_exists,
            "latency_ms": round(dt * 1000, 2),
        }

        print(f"  Neo4j:      {'OK' if status.neo4j_connected else 'FAIL'}")
        print(f"  LLM:        {'OK' if status.llm_reachable else 'FAIL'}")
        print(f"  Embedder:   {'OK' if status.embedding_model_loaded else 'FAIL'}")
        print(f"  VectorIdx:  {'OK' if status.vector_index_exists else 'FAIL'}")
        print(f"  Latency:    {dt*1000:.0f} ms")

        if not status.neo4j_connected:
            errors.append("Neo4j not reachable")
        if not status.llm_reachable:
            errors.append("LLM not reachable")

        # ════════════════════════════════════════════════════════════════
        # PHASE 2 — INGEST PROFILING
        # ════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  PHASE 2: INGEST PROFILING")
        print("=" * 70)

        ingest_timings = []
        nodes_created_total = 0
        nodes_updated_total = 0
        rels_created_total = 0
        per_doc = []

        for i, (text, ref_prefix, domain, _) in enumerate(CORPUS):
            ref_id = f"{ref_prefix}-{bench_user}-{i}"
            t0 = time.perf_counter()
            result = await client.ingest(
                user_id=bench_user, text=text, reference_id=ref_id,
            )
            dt = time.perf_counter() - t0
            ingest_timings.append(dt)

            nc = len(result.nodes_created)
            nu = len(result.nodes_updated)
            rc = result.relationships_created
            nodes_created_total += nc
            nodes_updated_total += nu
            rels_created_total += rc

            per_doc.append({
                "index": i, "domain": domain, "reference_id": ref_id,
                "skipped": result.skipped,
                "nodes_created": nc, "nodes_updated": nu,
                "relationships_created": rc,
                "latency_ms": round(dt * 1000, 2),
                "tokens_prompt": result.tokens_prompt,
                "tokens_completion": result.tokens_completion,
                "tokens_total": result.tokens_total,
            })

            status_str = "SKIP" if result.skipped else f"+{nc}n +{nu}u +{rc}r"
            tok_str = f"tok={result.tokens_total}" if not result.skipped else ""
            print(f"  [{i+1:02d}/{len(CORPUS)}] {dt*1000:7.0f} ms  {status_str}  "
                  f"{tok_str:<12}  ({domain})")

        total_tokens_prompt = sum(d["tokens_prompt"] for d in per_doc)
        total_tokens_completion = sum(d["tokens_completion"] for d in per_doc)
        total_tokens_all = sum(d["tokens_total"] for d in per_doc)
        avg_tokens = round(total_tokens_all / len(per_doc), 1) if per_doc else 0.0

        metrics["ingest"] = {
            "corpus_size": len(CORPUS),
            "latency": _latency_stats(ingest_timings),
            "throughput_ops_per_sec": round(
                len(ingest_timings) / sum(ingest_timings), 2
            ) if sum(ingest_timings) > 0 else 0,
            "nodes_created_total": nodes_created_total,
            "nodes_updated_total": nodes_updated_total,
            "relationships_created_total": rels_created_total,
            "avg_nodes_per_ingest": round(
                (nodes_created_total + nodes_updated_total) / len(CORPUS), 2
            ),
            "total_tokens_prompt": total_tokens_prompt,
            "total_tokens_completion": total_tokens_completion,
            "total_tokens_all": total_tokens_all,
            "avg_tokens_per_ingest": avg_tokens,
            "per_document": per_doc,
        }

        print(f"\n  SUMMARY: {len(CORPUS)} docs -> {nodes_created_total} nodes, "
              f"{rels_created_total} rels | tokens: {total_tokens_all} total "
              f"(avg {avg_tokens}/doc)")
        print(f"  p50={metrics['ingest']['latency']['p50_ms']} ms  "
              f"p95={metrics['ingest']['latency']['p95_ms']} ms  "
              f"throughput={metrics['ingest']['throughput_ops_per_sec']} ops/s")

        if any(d["skipped"] for d in per_doc):
            errors.append("Some documents were trivial-filtered (unexpected)")

        # ════════════════════════════════════════════════════════════════
        # PHASE 3 — RECALL PROFILING + ACCURACY
        # ════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  PHASE 3: RECALL PROFILING + ACCURACY")
        print("=" * 70)

        recall_timings = []
        per_query = []
        precision_at_k = {k: [] for k in (1, 3, 5, 10)}
        recall_at_k = {k: [] for k in (1, 3, 5, 10)}
        reciprocal_ranks = []

        for qi, (query, expected_domain, expected_entities) in enumerate(RECALL_QUERIES):
            t0 = time.perf_counter()
            result = await client.recall(user_id=bench_user, query=query, top_k=10)
            dt = time.perf_counter() - t0
            recall_timings.append(dt)

            # retry once if empty (vector index propagation delay)
            if not result.nodes:
                await asyncio.sleep(2.0)
                t0 = time.perf_counter()
                result = await client.recall(user_id=bench_user, query=query, top_k=10)
                dt = time.perf_counter() - t0
                recall_timings[-1] = dt

            blob_all = _blob(result.nodes)
            total_expected = len(expected_entities)

            # MRR
            first_relevant_rank = None
            for rank_idx, node in enumerate(result.nodes, 1):
                node_blob = _blob([node])
                if any(_entity_hit(node_blob, e) for e in expected_entities):
                    first_relevant_rank = rank_idx
                    break
            rr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            reciprocal_ranks.append(rr)

            # Precision@K and Recall@K
            for k in (1, 3, 5, 10):
                top_k_blob = _blob(result.nodes[:k])
                hits = sum(1 for e in expected_entities if _entity_hit(top_k_blob, e))
                precision_at_k[k].append(min(hits, k) / k)
                recall_at_k[k].append(
                    min(hits, total_expected) / total_expected if total_expected else 0
                )

            entity_hits = {e: _entity_hit(blob_all, e) for e in expected_entities}

            per_query.append({
                "index": qi, "query": query,
                "expected_domain": expected_domain,
                "expected_entities": expected_entities,
                "entity_hits": entity_hits,
                "hit_rate": sum(entity_hits.values()) / total_expected if total_expected else 0,
                "nodes_returned": len(result.nodes),
                "total_candidates": result.total_candidates,
                "from_cache": result.from_cache,
                "reciprocal_rank": round(rr, 4),
                "top_score": round(result.nodes[0].score, 4) if result.nodes else 0,
                "latency_ms": round(dt * 1000, 2),
            })

            hit_pct = per_query[-1]["hit_rate"] * 100
            print(f"  [{qi+1:02d}/{len(RECALL_QUERIES)}] {dt*1000:7.0f} ms  "
                  f"hits={hit_pct:5.1f}%  nodes={len(result.nodes)}  "
                  f"RR={rr:.2f}  ({expected_domain})")

        avg_precision = {k: round(statistics.mean(v), 4) for k, v in precision_at_k.items()}
        avg_recall = {k: round(statistics.mean(v), 4) for k, v in recall_at_k.items()}
        mrr = round(statistics.mean(reciprocal_ranks), 4)
        f1_at_k = {}
        for k in (1, 3, 5, 10):
            p, r = avg_precision[k], avg_recall[k]
            f1_at_k[k] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

        metrics["recall"] = {
            "query_count": len(RECALL_QUERIES),
            "latency": _latency_stats(recall_timings),
            "throughput_ops_per_sec": round(
                len(recall_timings) / sum(recall_timings), 2
            ) if sum(recall_timings) > 0 else 0,
            "per_query": per_query,
        }
        metrics["accuracy"] = {
            "precision_at_k": avg_precision,
            "recall_at_k": avg_recall,
            "f1_at_k": f1_at_k,
            "mrr": mrr,
            "mean_hit_rate": round(statistics.mean([q["hit_rate"] for q in per_query]), 4),
            "queries_with_zero_results": sum(1 for q in per_query if q["nodes_returned"] == 0),
        }

        print(f"\n  SUMMARY: {len(RECALL_QUERIES)} queries")
        print(f"  Latency  p50={metrics['recall']['latency']['p50_ms']} ms  "
              f"p95={metrics['recall']['latency']['p95_ms']} ms")
        print(f"  MRR={mrr}  P@3={avg_precision[3]}  R@3={avg_recall[3]}  "
              f"F1@3={f1_at_k[3]}")

        # ════════════════════════════════════════════════════════════════
        # PHASE 4 — SEARCH (vector-only, no BFS)
        # ════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  PHASE 4: SEARCH (vector-only)")
        print("=" * 70)

        search_queries = [q[0] for q in RECALL_QUERIES[:6]]
        search_timings = []
        for q in search_queries:
            t0 = time.perf_counter()
            await client.search(user_id=bench_user, query=q, top_k=5)
            dt = time.perf_counter() - t0
            search_timings.append(dt)

        metrics["search"] = {
            "query_count": len(search_queries),
            "latency": _latency_stats(search_timings),
            "throughput_ops_per_sec": round(
                len(search_timings) / sum(search_timings), 2
            ) if sum(search_timings) > 0 else 0,
        }
        print(f"  {len(search_queries)} queries  "
              f"p50={metrics['search']['latency']['p50_ms']} ms  "
              f"p95={metrics['search']['latency']['p95_ms']} ms")

        # ════════════════════════════════════════════════════════════════
        # PHASE 5 — SEMANTIC DISCRIMINATION
        # ════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  PHASE 5: SEMANTIC DISCRIMINATION")
        print("=" * 70)

        disc_tests = [
            ("Where do visitors park?", "operations", ["finance", "healthcare"]),
            ("Who developed the Sigma-7 volatility model?", "finance", ["operations", "education"]),
            ("What did the GDPR audit find?", "compliance", ["engineering", "operations"]),
            ("Who leads the cardiology department?", "healthcare", ["finance", "engineering"]),
        ]
        disc_results = []
        contaminations = 0
        for query, target, anti_domains in disc_tests:
            r = await client.recall(user_id=bench_user, query=query, top_k=5)
            top3_refs = [n.reference_id or "" for n in r.nodes[:3]]
            top3_domains = set()
            for ref in top3_refs:
                for _, ref_prefix, domain, _ in CORPUS:
                    if ref.startswith(ref_prefix):
                        top3_domains.add(domain)
            contaminated = bool(top3_domains & set(anti_domains))
            if contaminated:
                contaminations += 1
            disc_results.append({
                "query": query, "target_domain": target,
                "anti_domains": anti_domains,
                "top3_domains_found": list(top3_domains),
                "contaminated": contaminated,
            })
            print(f"  {'FAIL' if contaminated else 'OK  '}  {query[:50]}...")

        metrics["discrimination"] = {
            "tests": disc_results,
            "contamination_rate": round(contaminations / len(disc_tests), 4),
            "clean_rate": round(1 - contaminations / len(disc_tests), 4),
        }
        print(f"\n  Clean rate: {metrics['discrimination']['clean_rate']*100:.0f}%")

        # ════════════════════════════════════════════════════════════════
        # PHASE 6 — USER ISOLATION
        # ════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  PHASE 6: USER ISOLATION")
        print("=" * 70)

        user_b = f"bench-iso-{uuid.uuid4().hex[:12]}"
        secret = f"SECRET_{uuid.uuid4().hex[:10]}"
        await client.ingest(
            user_id=bench_user,
            text=f"Project {secret} is a top-secret initiative by Director Kim.",
        )
        r_b = await client.recall(
            user_id=user_b, query=f"Tell me about project {secret}", top_k=10,
        )
        blob_b = _blob(r_b.nodes)
        leaked = secret.lower() in blob_b

        metrics["user_isolation"] = {
            "secret_token": secret,
            "leaked": leaked,
            "user_b_nodes_returned": len(r_b.nodes),
            "verdict": "FAIL — data leaked" if leaked else "PASS — isolation holds",
        }
        try:
            await client._driver.execute(_CLEANUP_CYPHER, {"userId": user_b})
        except Exception:
            pass

        print(f"  {'FAIL' if leaked else 'PASS'}  user B got {len(r_b.nodes)} nodes, "
              f"secret {'LEAKED' if leaked else 'not found'}")

        if leaked:
            errors.append("User isolation FAILED")

        # ════════════════════════════════════════════════════════════════
        # PHASE 7 — SCALING CURVE
        # ════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  PHASE 7: SCALING CURVE")
        print("=" * 70)

        scaling_docs = [
            "Robotics engineer Yuki Tanaka designed the RX-500 arm with 6 DOF.",
            "The warehouse in Düsseldorf stores 2.4 million SKUs with RFID tracking.",
            "Quantum physicist Dr. Raj Gupta published on topological qubits in Nature.",
            "Marketing VP Sandra Osei launched the 'Go Green' campaign in 12 markets.",
            "DevOps intern Alex Novak wrote a Terraform module for multi-region RDS.",
        ]
        scale_user = f"bench-scale-{uuid.uuid4().hex[:8]}"
        scale_timings = []
        try:
            for i, text in enumerate(scaling_docs):
                t0 = time.perf_counter()
                await client.ingest(
                    user_id=scale_user, text=text, reference_id=f"scale-{i}",
                )
                dt = time.perf_counter() - t0
                scale_timings.append({"doc_index": i + 1, "latency_ms": round(dt * 1000, 2)})
                print(f"  [{i+1}/{len(scaling_docs)}] {dt*1000:.0f} ms")
        finally:
            try:
                await client._driver.execute(_CLEANUP_CYPHER, {"userId": scale_user})
            except Exception:
                pass

        latencies = [t["latency_ms"] for t in scale_timings]
        n = len(latencies)
        if n > 1:
            x_mean = (n - 1) / 2
            y_mean = statistics.mean(latencies)
            numer = sum((i - x_mean) * (latencies[i] - y_mean) for i in range(n))
            denom = sum((i - x_mean) ** 2 for i in range(n))
            slope = numer / denom if denom != 0 else 0
        else:
            slope = 0

        metrics["scaling"] = {
            "docs_added": len(scaling_docs),
            "per_doc": scale_timings,
            "slope_ms_per_doc": round(slope, 2),
            "interpretation": (
                "near-constant (O(1))" if abs(slope) < 500
                else "linear growth detected"
            ),
        }
        print(f"\n  Slope: {slope:.1f} ms/doc -> {metrics['scaling']['interpretation']}")

        # ════════════════════════════════════════════════════════════════
        # PHASE 8 — GRAPH SNAPSHOT
        # ════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  PHASE 8: GRAPH SNAPSHOT")
        print("=" * 70)

        snap = await client.get_graph(user_id=bench_user, page=1, page_size=200)
        metrics["graph_stats"] = {
            "total_nodes": snap.total_nodes,
            "total_relationships": snap.total_relationships,
            "nodes_per_ingest": round(snap.total_nodes / len(CORPUS), 2) if CORPUS else 0,
            "rels_per_ingest": round(
                snap.total_relationships / len(CORPUS), 2
            ) if CORPUS else 0,
            "sample_labels": list({
                n.get("label") or (n.get("labels", ["?"])[0] if n.get("labels") else "?")
                for n in (snap.nodes[:20] if snap.nodes else [])
            }),
        }
        print(f"  {snap.total_nodes} nodes, {snap.total_relationships} rels  "
              f"({metrics['graph_stats']['nodes_per_ingest']} nodes/ingest)")

    finally:
        # ════════════════════════════════════════════════════════════════
        # CLEANUP + SAVE REPORT
        # ════════════════════════════════════════════════════════════════
        try:
            await client._driver.execute(_CLEANUP_CYPHER, {"userId": bench_user})
        except Exception:
            pass
        try:
            await client.__aexit__(None, None, None)
        except Exception:
            pass

    # ── build summary ──
    metrics["summary"] = {
        "headline": "Engram-Memory SDK -- Live Benchmark Results",
        "corpus_size": len(CORPUS),
        "query_count": len(RECALL_QUERIES),
        "ingest_p50_ms": metrics.get("ingest", {}).get("latency", {}).get("p50_ms"),
        "ingest_p95_ms": metrics.get("ingest", {}).get("latency", {}).get("p95_ms"),
        "ingest_throughput": metrics.get("ingest", {}).get("throughput_ops_per_sec"),
        "recall_p50_ms": metrics.get("recall", {}).get("latency", {}).get("p50_ms"),
        "recall_p95_ms": metrics.get("recall", {}).get("latency", {}).get("p95_ms"),
        "recall_throughput": metrics.get("recall", {}).get("throughput_ops_per_sec"),
        "search_p50_ms": metrics.get("search", {}).get("latency", {}).get("p50_ms"),
        "mrr": metrics.get("accuracy", {}).get("mrr"),
        "precision_at_1": metrics.get("accuracy", {}).get("precision_at_k", {}).get(1),
        "precision_at_3": metrics.get("accuracy", {}).get("precision_at_k", {}).get(3),
        "precision_at_5": metrics.get("accuracy", {}).get("precision_at_k", {}).get(5),
        "recall_at_3": metrics.get("accuracy", {}).get("recall_at_k", {}).get(3),
        "f1_at_3": metrics.get("accuracy", {}).get("f1_at_k", {}).get(3),
        "discrimination_clean_rate": metrics.get("discrimination", {}).get("clean_rate"),
        "user_isolation": metrics.get("user_isolation", {}).get("verdict"),
        "total_nodes": metrics.get("graph_stats", {}).get("total_nodes"),
        "total_relationships": metrics.get("graph_stats", {}).get("total_relationships"),
        "scaling_slope_ms": metrics.get("scaling", {}).get("slope_ms_per_doc"),
        "tokens_total_ingest": metrics.get("ingest", {}).get("total_tokens_all"),
        "tokens_avg_per_ingest": metrics.get("ingest", {}).get("avg_tokens_per_ingest"),
        "tokens_prompt_total": metrics.get("ingest", {}).get("total_tokens_prompt"),
        "tokens_completion_total": metrics.get("ingest", {}).get("total_tokens_completion"),
    }

    # Cost estimation
    prompt_total = metrics["summary"].get("tokens_prompt_total") or 0
    completion_total = metrics["summary"].get("tokens_completion_total") or 0
    cost_prompt = prompt_total * TOKEN_PRICE_PER_1K["prompt"] / 1000
    cost_completion = completion_total * TOKEN_PRICE_PER_1K["completion"] / 1000
    estimated_cost = round(cost_prompt + cost_completion, 6)
    corpus_size = len(CORPUS) or 1
    metrics["summary"]["estimated_cost_usd"] = estimated_cost
    metrics["summary"]["cost_per_ingest_usd"] = round(estimated_cost / corpus_size, 6)
    metrics["summary"]["cost_per_1k_docs_usd"] = round(
        estimated_cost / corpus_size * 1000, 4
    )

    # ── save JSON ──
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # ── print final report ──
    s = metrics["summary"]
    print("\n" + "=" * 70)
    print("  ENGRAM-MEMORY SDK -- BENCHMARK REPORT")
    print("=" * 70)
    print(f"  Timestamp:  {metrics['meta']['timestamp_utc']}")
    print(f"  Python:     {metrics['meta']['python_version']}")
    print(f"  SDK:        {metrics['meta']['sdk_version']}")
    print(f"  Corpus:     {s['corpus_size']} documents across 6 domains")
    print(f"  Queries:    {s['query_count']} recall queries")
    print()
    print("  +-----------------------------------------------------------+")
    print("  |  LATENCY                                               |")
    print("  +-----------------------------------------------------------+")
    print(f"  |  Ingest   p50={s['ingest_p50_ms']:>8} ms   "
          f"p95={s['ingest_p95_ms']:>8} ms       |")
    print(f"  |  Recall   p50={s['recall_p50_ms']:>8} ms   "
          f"p95={s['recall_p95_ms']:>8} ms       |")
    print(f"  |  Search   p50={s['search_p50_ms']:>8} ms"
          f"                            |")
    print("  +-----------------------------------------------------------+")
    print("  |  THROUGHPUT                                            |")
    print("  +-----------------------------------------------------------+")
    print(f"  |  Ingest   {s['ingest_throughput']:>6} ops/s"
          f"                              |")
    print(f"  |  Recall   {s['recall_throughput']:>6} ops/s"
          f"                              |")
    print("  +-----------------------------------------------------------+")
    print("  |  RETRIEVAL QUALITY                                     |")
    print("  +-----------------------------------------------------------+")
    print(f"  |  MRR          = {s['mrr']:<8}"
          f"                              |")
    print(f"  |  Precision@1  = {s['precision_at_1']:<8}  "
          f"Precision@3 = {s['precision_at_3']:<8}     |")
    print(f"  |  Recall@3     = {s['recall_at_3']:<8}  "
          f"F1@3        = {s['f1_at_3']:<8}     |")
    print("  +-----------------------------------------------------------+")
    print("  |  SAFETY & SCALE                                        |")
    print("  +-----------------------------------------------------------+")
    print(f"  |  Discrimination clean rate: "
          f"{s['discrimination_clean_rate']*100:.0f}%"
          f"                     |")
    print(f"  |  User isolation: {s['user_isolation']:<39}|")
    print(f"  |  Scaling slope: {s['scaling_slope_ms']:>8} ms/doc"
          f"                       |")
    print("  +-----------------------------------------------------------+")
    print("  |  GRAPH                                                 |")
    print("  +-----------------------------------------------------------+")
    print(f"  |  Nodes: {s['total_nodes']}   Relationships: {s['total_relationships']}"
          f"                         |")
    print("  +-----------------------------------------------------------+")
    print("  |  TOKEN USAGE (ingest LLM calls)                        |")
    print("  +-----------------------------------------------------------+")
    print(f"  |  Total tokens:      {s['tokens_total_ingest']:<8}"
          f"  (prompt+completion)         |")
    print(f"  |    Prompt tokens:   {s['tokens_prompt_total']:<8}"
          f"                              |")
    print(f"  |    Completion tok:  {s['tokens_completion_total']:<8}"
          f"                              |")
    print(f"  |  Avg tokens/ingest: {s['tokens_avg_per_ingest']:<8}"
          f"                              |")
    print("  +-----------------------------------------------------------+")
    print("  |  COST ESTIMATION                                       |")
    print("  +-----------------------------------------------------------+")
    print(f"  |  Estimated total:   ${s['estimated_cost_usd']:<10}"
          f"                          |")
    print(f"  |  Per ingest:        ${s['cost_per_ingest_usd']:<10}"
          f"                          |")
    print(f"  |  Per 1K docs:       ${s['cost_per_1k_docs_usd']:<10}"
          f"                          |")
    print(f"  |  Pricing: prompt=${TOKEN_PRICE_PER_1K['prompt']}/1K  "
          f"completion=${TOKEN_PRICE_PER_1K['completion']}/1K      |")
    print("  +-----------------------------------------------------------+")
    print(f"\n  Report saved: {REPORT_PATH}")
    print("=" * 70)

    # ── assert no critical errors ──
    assert not errors, f"Benchmark had critical errors: {errors}"
