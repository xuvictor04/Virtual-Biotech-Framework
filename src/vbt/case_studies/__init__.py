"""Case studies from the paper and their CLI entry points.

  vbt case1 annotate|phase1|validate|features|stats   Case study 1 (Fig. 2-3)
  vbt scenario list|run|score                          Case studies 2-3 (Fig. 4-5) + agentic Case 1
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


def add_case_parsers(sub) -> None:
    c = sub.add_parser("case1", help="Case study 1: trial-outcome curation and target-feature associations")
    cs = c.add_subparsers(dest="step", required=True)

    a = cs.add_parser("annotate", help="one clinical-trialist agent per NCT ID (bulk, resumable)")
    a.add_argument("--out", default="results/case1/annotations.jsonl")
    a.add_argument("--sample", type=int, help="random sample of N trials (default: all Phase II/III)")
    a.add_argument("--ids", nargs="*", help="explicit NCT IDs")
    a.add_argument("--ids-file", help="file with one NCT ID per line")
    a.add_argument("--phases", default="2,3")
    a.add_argument("--concurrency", type=int, default=64)
    a.add_argument("--budget", type=float, help="stop when total spend reaches this many USD")
    a.add_argument("--seed", type=int, default=0)

    p1 = cs.add_parser("phase1", help="algorithmic Phase I -> II progression labels")
    p1.add_argument("--out", default="results/case1/phase1_progression.csv")

    v = cs.add_parser("validate", help="agreement of agent labels with a reference")
    v.add_argument("--pred", default="results/case1/annotations.jsonl", help="annotations JSONL or labels CSV")
    v.add_argument("--ref", default="released", help="'released', a manual-review CSV, or tdc:<path>")
    v.add_argument("--sample-manual", type=int, help="write a manual-review sample of N per phase and exit")

    f = cs.add_parser("features", help="tau / bimodality features from a Tabula Sapiens h5ad")
    f.add_argument("--h5ad", required=True)
    f.add_argument("--out", default="results/case1/target_features.csv")
    f.add_argument("--tissue-key", default="tissue_in_publication")
    f.add_argument("--celltype-key", default="cell_ontology_class")
    f.add_argument("--gene-id-column", default="ensembl_id")

    s = cs.add_parser("stats", help="association of target features with trial outcomes")
    s.add_argument("--features", default="results/case1/target_features.csv")
    s.add_argument("--labels", default="released", help="'released' or a labels CSV from `annotate`")
    s.add_argument("--genetic-pairs", help="CSV of targetId,diseaseId with direct genetic evidence")
    s.add_argument("--n-perm", type=int, default=1000, help="outcome permutations (paper)")
    s.add_argument("--gene-perm", type=int, default=0,
                   help="gene-label permutations: calibration null for aggregation confounding")
    s.add_argument("--out", default="results/case1/associations.csv")
    c.set_defaults(handler=_case1)

    sc = sub.add_parser("scenario", help="scripted case-study conversations (B7-H3, OSMR, curation)")
    ss = sc.add_subparsers(dest="action", required=True)
    ss.add_parser("list")
    r = ss.add_parser("run")
    r.add_argument("name")
    r.add_argument("--turns", type=int, help="only the first N turns")
    r.add_argument("--score", action="store_true", help="grade the run against the paper's findings")
    g = ss.add_parser("score")
    g.add_argument("name")
    g.add_argument("run_dir")
    sc.set_defaults(handler=_scenario)


# ---------------------------------------------------------------- case 1

def _case1(args, config: dict[str, Any]) -> int:
    import pandas as pd

    from ..config import resolve_path
    from .trial_outcomes import annotate as ann

    out_dir = resolve_path("results/case1")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.step == "annotate":
        mapping = pd.read_parquet(ann.mapping_path(config))
        ids = list(args.ids or [])
        if args.ids_file:
            ids += [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
        phases = tuple(float(p) for p in args.phases.split(","))
        trials = ann.select_trials(mapping, phases=phases, sample=args.sample, seed=args.seed, ids=ids or None)
        print(f"{len(trials)} trials selected")

        async def main() -> int:
            from ..orchestrator import open_session
            session = await open_session(config, start_mcp=not args.no_mcp)
            last = [0]

            def progress(st):
                n = st.done + st.failed
                if n - last[0] >= 25:
                    last[0] = n
                    print(f"  {n} done ({st.failed} failed), ${sum(st.costs):.2f}")
            try:
                summary = await ann.annotate(session.rt, trials, resolve_path(args.out),
                                             concurrency=args.concurrency, budget_usd=args.budget,
                                             on_progress=progress)
            finally:
                await session.close()
            print(json.dumps(summary, indent=2))
            labels = ann.results_to_labels(resolve_path(args.out))
            labels.to_csv(resolve_path(args.out).with_suffix(".labels.csv"), index=False)
            return 0
        return asyncio.run(main())

    if args.step == "phase1":
        from .trial_outcomes.phase1 import phase1_progression
        res = phase1_progression(pd.read_parquet(ann.mapping_path(config)))
        res.to_csv(resolve_path(args.out), index=False)
        print(res["phase2_progression"].value_counts().to_string())
        return 0

    if args.step == "validate":
        from .trial_outcomes import validation as val
        released = pd.read_csv(ann.labels_path(config))
        if args.sample_manual:
            path = out_dir / "manual_review_sample.csv"
            val.manual_review_sample(released, args.sample_manual).to_csv(path, index=False)
            print(f"wrote {path}: annotate these trials manually (same columns as the released labels)")
            return 0
        pred_path = resolve_path(args.pred)
        pred = ann.results_to_labels(pred_path) if pred_path.suffix == ".jsonl" else pd.read_csv(pred_path)
        if args.ref == "released":
            ref = released
        elif args.ref.startswith("tdc:"):
            ref = val.load_tdc(args.ref[4:])
        else:
            ref = pd.read_csv(args.ref)
        print(val.agreement_report(pred, ref).to_string(index=False))
        return 0

    if args.step == "features":
        from .trial_outcomes import features
        df = features.compute_features_from_h5ad(args.h5ad, tissue_key=args.tissue_key,
                                                 celltype_key=args.celltype_key,
                                                 gene_id_column=args.gene_id_column)
        df.to_csv(resolve_path(args.out))
        print(f"wrote {len(df)} genes to {args.out}")
        return 0

    if args.step == "stats":
        from .trial_outcomes.pipeline import run_stats
        table = run_stats(config, features_path=resolve_path(args.features), labels=args.labels,
                          genetic_pairs=args.genetic_pairs, n_perm=args.n_perm, gene_perm=args.gene_perm,
                          out_dir=resolve_path(args.out).parent)
        table.to_csv(resolve_path(args.out), index=False)
        print(table.to_string(index=False))
        return 0
    return 2


# ---------------------------------------------------------------- scenarios

def _scenario(args, config: dict[str, Any]) -> int:
    from ..config import load_config
    from . import scenarios as sc

    if args.action == "list":
        for name, s in sc.list_scenarios().items():
            print(f"{name:16s} {s['title']}  (profiles: {', '.join(s.get('profiles') or []) or '-'})")
        return 0
    scen = sc.load_scenario(args.name)
    config = load_config(list(dict.fromkeys(args.profile + list(scen.get("profiles") or []))),
                         {"web": {"enabled": False}} if args.no_web else None)

    async def main() -> int:
        from ..cli import _printer
        console, on_event = _printer(args.verbose)
        if args.action == "run":
            run, _ = await sc.run_scenario(config, scen, turns=args.turns, on_event=on_event,
                                           start_mcp=not args.no_mcp)
            console.print(f"\nRun: {run.dir}  cost ${run.cost.total_usd:.2f} "
                          f"(paper: ${scen.get('paper_cost_usd', '?')})")
            run_dir = run.dir
            if not args.score:
                return 0
        else:
            run_dir = Path(args.run_dir)
        from ..orchestrator import open_session
        session = await open_session(config, start_mcp=False)
        try:
            score = await sc.score_run(session.rt, run_dir, scen)
        finally:
            await session.close()
        console.print_json(json.dumps(score))
        return 0

    return asyncio.run(main())
