"""Production end-to-end validation: health, optional ingests (timed), recall/search suite.

Loads environment from the current working directory only:

  .env
  engram_memory/.env

(non-overriding). Set ``NEO4J_*``, ``LLM_*``, and embedding vars before running, or
place them in those files when executing from the repository root.

Skip ingestion (retrieval-only)::

  engram_memory-e2e --skip-seed --user-id <existing_user_id>

or::

  E2E_USER_ID=<id> engram_memory-e2e --skip-seed

Environment (optional)::

  E2E_LLM_TIMEOUT_SEC     HTTP timeout for LLM calls (default: 120)
  E2E_INGEST_TIMEOUT_SEC  asyncio wall timeout per ingest (default: llm + 45)
  E2E_USER_ID             default --user-id when --skip-seed is set
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

Mode = Literal["recall", "search"]


def _load_dotenv_cwd() -> None:
    if not load_dotenv:
        return
    cwd = Path.cwd()
    load_dotenv(cwd / ".env", override=False)
    load_dotenv(cwd / "engram" / ".env", override=False)


def _filter_nodes(
    nodes: list[Any],
    *,
    min_score: float | None = None,
    label_substr: str | None = None,
    summary_substr: str | None = None,
    require_reference: bool = False,
) -> list[Any]:
    out = []
    for n in nodes:
        if min_score is not None and getattr(n, "score", 0.0) < min_score:
            continue
        if label_substr and label_substr.lower() not in (n.label or "").lower():
            continue
        if summary_substr and summary_substr.lower() not in (n.summary or "").lower():
            continue
        if require_reference and not getattr(n, "reference_id", None):
            continue
        out.append(n)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--skip-seed",
        action="store_true",
        help="Run recall/search validation only; requires --user-id or E2E_USER_ID.",
    )
    p.add_argument(
        "--user-id",
        dest="user_id",
        default=None,
        metavar="ID",
        help="Graph user id. Required with --skip-seed. If omitted when seeding, a unique id is generated.",
    )
    p.add_argument(
        "--batch-seed",
        action="store_true",
        help="Single batched ingest (one LLM call). Default is one ingest per seed document with per-item timing.",
    )
    p.add_argument(
        "--full-health",
        action="store_true",
        help="Include LLM ping in health check (extra API round-trip before ingests).",
    )
    p.add_argument(
        "--no-llm-timeout",
        action="store_true",
        help="Do not set llm_request_timeout (use provider/LiteLLM default).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Lower llm_max_retries and cap max_tokens for quicker extraction (smaller graphs).",
    )
    return p


def _e2e_config(args: argparse.Namespace):
    from engram_memory.config import Config

    c = Config()
    updates: dict[str, Any] = {}
    if not args.no_llm_timeout:
        updates["llm_request_timeout"] = float(os.environ.get("E2E_LLM_TIMEOUT_SEC", "120"))
    if args.fast:
        updates["llm_max_retries"] = 1
        updates["llm_max_tokens"] = min(c.llm_max_tokens, 2048)
    return c.model_copy(update=updates) if updates else c


def _ingest_wall_timeout_sec(llm_timeout: float | None) -> float:
    base = float(llm_timeout) if llm_timeout is not None else 150.0
    return float(os.environ.get("E2E_INGEST_TIMEOUT_SEC", str(base + 45.0)))


def _scenarios() -> list[dict[str, Any]]:
    return [
        {
            "name": "recall top_k=3",
            "mode": "recall",
            "query": "Who owns the Phoenix analytics rollout?",
            "top_k": 3,
            "filters": {},
        },
        {
            "name": "recall top_k=10",
            "mode": "recall",
            "query": "database migration weekend",
            "top_k": 10,
            "filters": {},
        },
        {
            "name": "recall filter min_score",
            "mode": "recall",
            "query": "visitor parking and badges",
            "top_k": 12,
            "filters": {"min_score": 0.05},
        },
        {
            "name": "recall filter summary_substr",
            "mode": "recall",
            "query": "team leads and engineers",
            "top_k": 15,
            "filters": {"summary_substr": "on-call"},
        },
        {
            "name": "recall filter reference_id",
            "mode": "recall",
            "query": "salad bar Thursday cafeteria",
            "top_k": 10,
            "filters": {"require_reference": True},
        },
        {
            "name": "search detail_level=broad",
            "mode": "search",
            "query": "on-call rotation incident response",
            "top_k": 6,
            "detail_level": "broad",
        },
        {
            "name": "search detail_level=detailed",
            "mode": "search",
            "query": "on-call rotation incident response",
            "top_k": 6,
            "detail_level": "detailed",
        },
        {
            "name": "search detail_level=auto",
            "mode": "search",
            "query": "Kubernetes cluster upgrade timeline",
            "top_k": 8,
            "detail_level": "auto",
        },
        {
            "name": "recall top_k=1",
            "mode": "recall",
            "query": "Morgan Lee project owner",
            "top_k": 1,
            "filters": {},
        },
        {
            "name": "recall combined filters",
            "mode": "recall",
            "query": "engineering project Phoenix",
            "top_k": 15,
            "filters": {"min_score": 0.01, "summary_substr": "phoenix"},
        },
    ]


def _seed_documents() -> list[tuple[str, str]]:
    """(text, reference_id) pairs for validation corpus."""
    return [
        (
            "Morgan Lee is the project owner for the Phoenix analytics rollout; "
            "the go-live target is Q3. Jordan Vega handles on-call rotation for "
            "the data platform team.",
            "e2e-seed-phoenix",
        ),
        (
            "The team agreed to freeze schema changes during the database migration "
            "planned for the weekend of March 15; rollback playbook v2 is in Confluence.",
            "e2e-seed-migration",
        ),
        (
            "Visitors use garage VIS-2 on level 2 for badge pickup before entering "
            "the Seattle office; front desk validates government ID.",
            "e2e-seed-parking",
        ),
        (
            "The cafeteria runs a seasonal salad bar on Thursdays on the first-floor "
            "mezzanine; vegan options are labeled green.",
            "e2e-seed-cafe",
        ),
        (
            "Production Kubernetes cluster k8s-prod-01 will upgrade to 1.29 during "
            "the maintenance window Sunday 02:00-06:00 UTC; SRE signs off in #infra.",
            "e2e-seed-k8s",
        ),
    ]


def _log(line: str) -> None:
    print(line, flush=True)


async def _async_main(args: argparse.Namespace) -> int:
    from engram_memory.client import AsyncMemoryClient

    _load_dotenv_cwd()
    cfg = _e2e_config(args)

    uid = args.user_id or os.environ.get("E2E_USER_ID")
    if args.skip_seed:
        if not uid:
            print("error: --skip-seed requires --user-id or E2E_USER_ID", file=sys.stderr, flush=True)
            return 1
        user_id = uid
    else:
        user_id = uid or f"e2e-{uuid.uuid4().hex[:16]}"

    _log(f"user_id={user_id}")
    if cfg.llm_request_timeout is not None:
        _log(f"llm_request_timeout_sec={cfg.llm_request_timeout}")
    ingest_timeout = _ingest_wall_timeout_sec(cfg.llm_request_timeout)
    _log(f"ingest_wall_timeout_sec={ingest_timeout:.0f}")

    scenarios = _scenarios()
    seeds = _seed_documents()

    async with AsyncMemoryClient(cfg) as client:
        t0 = time.perf_counter()
        health = await client.health_check(ping_llm=args.full_health)
        _log(
            "health "
            f"neo4j={health.neo4j_connected} "
            f"llm_ping={health.llm_reachable} "
            f"embedder={health.embedding_model_loaded} "
            f"vector_index={health.vector_index_exists} "
            f"schema={health.schema_version_current} "
            f"elapsed_sec={time.perf_counter() - t0:.2f}",
        )

        core_ok = (
            health.neo4j_connected
            and health.embedding_model_loaded
            and health.vector_index_exists
            and health.schema_version_current
        )
        if args.full_health:
            core_ok = core_ok and health.llm_reachable
        if not core_ok:
            _log("error: health check failed (see flags above)")
            return 1

        if not args.skip_seed:
            if args.batch_seed:
                batched = "\n\n---\n\n".join(t for t, _ in seeds)
                _log(f"ingest batch: 1 LLM call reference_id=e2e-seed-batch (timeout {ingest_timeout:.0f}s)")
                t_ing = time.perf_counter()
                try:
                    r = await asyncio.wait_for(
                        client.ingest(
                            user_id=user_id,
                            text=batched,
                            reference_id="e2e-seed-batch",
                        ),
                        timeout=ingest_timeout,
                    )
                except asyncio.TimeoutError:
                    _log(f"error: ingest timed out after {ingest_timeout:.0f}s")
                    return 1
                elapsed = time.perf_counter() - t_ing
                _log(
                    f"ingest_done reference_id=e2e-seed-batch elapsed_sec={elapsed:.2f} "
                    f"skipped={r.skipped} nodes_created={len(r.nodes_created)} "
                    f"nodes_updated={len(r.nodes_updated)} rels={r.relationships_created}",
                )
            else:
                _log(f"ingest sequence: {len(seeds)} documents (one LLM call each)")
                for i, (text, ref) in enumerate(seeds, start=1):
                    preview = text.replace("\n", " ")[:100]
                    if len(text) > 100:
                        preview += "..."
                    _log(
                        f"ingest_start index={i}/{len(seeds)} reference_id={ref!r} "
                        f"preview={preview!r}",
                    )
                    t_ing = time.perf_counter()
                    try:
                        r = await asyncio.wait_for(
                            client.ingest(
                                user_id=user_id,
                                text=text,
                                reference_id=ref,
                            ),
                            timeout=ingest_timeout,
                        )
                    except asyncio.TimeoutError:
                        _log(
                            f"error: ingest index={i} reference_id={ref!r} "
                            f"timed out after {ingest_timeout:.0f}s",
                        )
                        return 1
                    elapsed = time.perf_counter() - t_ing
                    _log(
                        f"ingest_done index={i}/{len(seeds)} reference_id={ref!r} "
                        f"elapsed_sec={elapsed:.2f} "
                        f"skipped={r.skipped} nodes_created={len(r.nodes_created)} "
                        f"nodes_updated={len(r.nodes_updated)} rels={r.relationships_created}",
                    )
        else:
            _log("skip-seed: running retrieval validation only")

        _log("--- retrieval scenarios ---")
        for i, sc in enumerate(scenarios, start=1):
            name = sc["name"]
            mode: Mode = sc["mode"]
            query = sc["query"]
            top_k = sc["top_k"]
            flt = sc.get("filters") or {}

            t_q = time.perf_counter()
            if mode == "recall":
                raw = await client.recall(user_id=user_id, query=query, top_k=top_k)
            else:
                detail = sc.get("detail_level", "auto")
                raw = await client.search(
                    user_id=user_id,
                    query=query,
                    top_k=top_k,
                    detail_level=detail,
                )
            q_elapsed = time.perf_counter() - t_q

            filtered = _filter_nodes(
                raw.nodes,
                min_score=flt.get("min_score"),
                label_substr=flt.get("label_substr"),
                summary_substr=flt.get("summary_substr"),
                require_reference=flt.get("require_reference", False),
            )

            _log(
                f"scenario index={i:02d} name={name!r} elapsed_sec={q_elapsed:.2f} "
                f"hits={len(raw.nodes)} candidates={raw.total_candidates}",
            )
            if flt:
                _log(f"  post_filters={flt!r} filtered_count={len(filtered)}")
            display = filtered if flt else raw.nodes
            for j, n in enumerate(display[:3], start=1):
                summ = (n.summary or "")[:100]
                if len(n.summary or "") > 100:
                    summ += "..."
                lbl = getattr(n, "label", "") or ""
                _log(f"  rank={j} score={n.score:.4f} label={lbl!r} summary={summ!r}")
            if len(display) > 3:
                _log(f"  ... {len(display) - 3} more rows omitted")

        t_snap = time.perf_counter()
        snap = await client.get_graph(user_id=user_id, page=1, page_size=100)
        _log(
            f"graph_snapshot elapsed_sec={time.perf_counter() - t_snap:.2f} "
            f"total_nodes={snap.total_nodes} total_relationships={snap.total_relationships}",
        )
        _log(f"cleanup_cypher=MATCH (n) WHERE n.userId = '{user_id}' DETACH DELETE n")

    _log("e2e_validate: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console entry point for ``engram_memory-e2e`` and ``python -m engram.cli.e2e_validate``."""
    import traceback

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.skip_seed:
        args.user_id = args.user_id or os.environ.get("E2E_USER_ID")
        if not args.user_id:
            parser.error("--skip-seed requires --user-id or E2E_USER_ID")

    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr, flush=True)
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
