"""Case studies from the paper and their CLI entry points.

  vbt case1 annotate|phase1|validate|features|stats   Case study 1 (Fig. 2-3)
  vbt case1 replicate|benchmarks                       ... against the authors' Zenodo archive
  vbt data zenodo list|fetch|download|presets          the paper's Zenodo case-study archive
  vbt scenario list|run|score                          Case studies 2-3 (Fig. 4-5) + agentic Case 1
"""

from __future__ import annotations

import asyncio
import json
import sys
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
    from ..bulk import add_budget_arguments
    add_budget_arguments(a, what="trial")   # --concurrency --budget --budget-tokens --max-item-cost/-tokens
    a.add_argument("--prewarm", choices=["first_item", "none"], help="bulk.prewarm (default first_item)")
    a.add_argument("--protocol", help="trial_curation run directory or protocol .md designed by the clinical "
                   "trialist (default: the bundled annotator_prompt.md)")
    a.add_argument("--seed", type=int, default=0)

    p1 = cs.add_parser("phase1", help="algorithmic Phase I -> II progression labels")
    p1.add_argument("--out", default="results/case1/phase1_progression.csv")
    p1.add_argument("--statuses", default="Completed",
                    help="comma-separated registry statuses of Phase I trials to label (paper: Completed)")

    v = cs.add_parser("validate", help="agreement of agent labels with a reference")
    v.add_argument("--pred", default="results/case1/annotations.jsonl", help="annotations JSONL or labels CSV")
    v.add_argument("--ref", default="released", help="'released', a manual-review CSV, or tdc:<path>")
    v.add_argument("--sample-manual", type=int, help="write a manual-review sample of N per phase and exit")
    v.add_argument("--exclude-stopped", dest="exclude_stopped", action="store_true", default=True,
                   help="exclude stopped-early trials from endpoint agreement (default, as in the paper)")
    v.add_argument("--include-stopped", dest="exclude_stopped", action="store_false")

    f = cs.add_parser("features", help="tau / bimodality features from a Tabula Sapiens h5ad")
    f.add_argument("--h5ad", required=True)
    f.add_argument("--out", default="results/case1/target_features.csv")
    f.add_argument("--tissue-key", default="tissue_in_publication")
    f.add_argument("--celltype-key", default="cell_ontology_class")
    f.add_argument("--gene-id-column", default=None,
                   help="var column with Ensembl IDs (default: var_names when they are ENSG IDs, "
                        "else ensembl_id / feature_id)")
    f.add_argument("--layer", help="use this layer instead of X (must be log-normalised, or raw counts "
                   "with --normalize)")
    f.add_argument("--normalize", choices=["auto", "always", "never"], default="auto",
                   help="integer (raw-count) matrices: auto = normalize_total 1e4 + log1p; never = error")
    f.add_argument("--kurtosis", choices=["excess", "pearson"], default="excess",
                   help="kurtosis in the bimodality coefficient (paper text says Pearson; Pfister: excess)")
    f.add_argument("--no-bias-correction", action="store_true",
                   help="use biased (population) skewness/kurtosis in the bimodality coefficient")

    s = cs.add_parser("stats", help="association of target features with trial outcomes")
    s.add_argument("--features", default="results/case1/target_features.csv")
    s.add_argument("--labels", default="released", help="'released' or a labels CSV from `annotate`")
    s.add_argument("--genetic-pairs", help="CSV of targetId,diseaseId with direct genetic evidence")
    s.add_argument("--n-perm", type=int, default=1000, help="outcome permutations (paper)")
    s.add_argument("--gene-perm", type=int, default=0,
                   help="gene-label permutations: calibration null for aggregation confounding")
    s.add_argument("--out", default="results/case1/associations.csv")
    s.add_argument("--definitions", choices=["authors", "methods"], default="authors",
                   help="outcome definitions: the authors' archived code (default) or the Methods-text reading")

    rp = cs.add_parser("replicate", help="run our Case 1 statistics on the authors' Zenodo inputs and compare "
                       "with their result tables")
    rp.add_argument("--zenodo-dir", help="extracted virtualbiotech_submission folder "
                    "(default: $VBT_ZENODO_DIR or data/zenodo)")
    rp.add_argument("--n-perm", type=int, default=1000, help="outcome permutations (0 = skip)")
    rp.add_argument("--n-perm-beta", type=int, help="permutations for the beta (AE) models (default: --n-perm)")
    rp.add_argument("--gene-perm", type=int, default=200, help="gene-label permutations (harness check; 0 = skip)")
    rp.add_argument("--no-mixed", action="store_true", help="skip the GLMMs (slow without R)")
    rp.add_argument("--no-expr", action="store_true", help="skip the 2,604 cell-type expression models")
    rp.add_argument("--out", default="results/case1_replication")

    bp = cs.add_parser("benchmarks", help="table S2: competitor annotation agreement (Biomni, Kosmos, PantheonOS)")
    bp.add_argument("--zenodo-dir")
    bp.add_argument("--manual", help="manual-review ground-truth CSV (nct_id, primary, secondary, ae_binary), "
                    "if you have it (not shipped in the archive)")
    bp.add_argument("--tdc", help="TDC/HINT trial-outcome labels (CSV/TSV with nct_id + label)")
    bp.add_argument("--labels", help="Virtual Biotech labels CSV (default: the archive's reconciled labels)")
    bp.add_argument("--out", default="results/case1_replication/table_s2.csv")
    c.set_defaults(handler=_case1)

    sc = sub.add_parser("scenario", help="scripted case-study conversations (B7-H3, OSMR, curation)")
    ss = sc.add_subparsers(dest="action", required=True)
    ss.add_parser("list")
    r = ss.add_parser("run")
    r.add_argument("name")
    r.add_argument("--turns", type=int, help="only the first N turns")
    r.add_argument("--score", action="store_true", help="grade the run against the paper's findings")
    r.add_argument("--turns-source", choices=["synthetic", "paper"], default="synthetic",
                   help="replay the synthetic steering turns (default) or the paper's verbatim turns")
    g = ss.add_parser("score")
    g.add_argument("name")
    g.add_argument("run_dir")
    sc.set_defaults(handler=_scenario)

    from ..data.zenodo import add_data_parsers
    add_data_parsers(sub)


# ---------------------------------------------------------------- case 1

def _case1(args, config: dict[str, Any]) -> int:
    import pandas as pd

    from ..config import resolve_path
    from .trial_outcomes import annotate as ann

    out_dir = resolve_path("results/case1")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.step == "annotate":
        from ..bulk import usd_only_budget_problem
        from ..providers.base import ProviderError
        problem = usd_only_budget_problem(args.budget, getattr(args, "budget_tokens", None), config=config,
                                          usd_name="--budget", tokens_name="--budget-tokens")
        if problem:  # a 0-USD local model never reaches a USD cap
            print(f"error: {problem}", file=sys.stderr)
            return 2
        mapping = pd.read_parquet(ann.mapping_path(config))
        ids = list(args.ids or [])
        if args.ids_file:
            ids += [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
        phases = tuple(float(p) for p in args.phases.split(","))
        trials = ann.select_trials(mapping, phases=phases, sample=args.sample, seed=args.seed, ids=ids or None)
        print(f"{len(trials)} trials selected")
        protocol = schema = None
        if getattr(args, "protocol", None):
            protocol, schema, found = ann.load_protocol(resolve_path(args.protocol))
            print(f"protocol: {found['protocol']}" + (f"; schema: {found['schema']}" if "schema" in found else
                                                      "; schema: TrialAnnotation"))

        async def main() -> int:
            from ..orchestrator import open_session
            session = await open_session(config, start_mcp=not args.no_mcp)
            last = [0]

            def progress(st):
                n = st.done + st.failed
                if n - last[0] >= 25:
                    last[0] = n
                    from ..budget import format_tokens
                    print(f"  {n} done ({st.failed} failed), ${sum(st.costs):.2f}, "
                          f"{format_tokens(sum(st.tokens))} tokens")
            from ..bulk import budget_kwargs
            kw: dict[str, Any] = budget_kwargs(args)
            if getattr(args, "prewarm", None):
                kw["prewarm"] = args.prewarm
            try:
                summary = await ann.annotate(session.rt, trials, resolve_path(args.out), budget_usd=args.budget,
                                             on_progress=progress, protocol=protocol, schema=schema, **kw)
            except ProviderError as exc:
                print(f"\nFATAL: the bulk run stopped on a non-retryable provider error: {exc}\n"
                      "Check the API key, model name and account permissions; finished trials are kept in "
                      f"{resolve_path(args.out)} and the run resumes from there.", file=sys.stderr)
                summ = getattr(exc, "bulk_summary", None)
                if summ:
                    print(json.dumps(summ, indent=2), file=sys.stderr)
                return 1
            finally:
                await session.close()
            print(json.dumps(summary, indent=2))
            if summary.get("budget_stop"):
                print(f"stopped at the bulk budget (${summary.get('budget_spent')}, "
                      f"{summary.get('budget_tokens_spent')} tokens); "
                      f"{summary.get('unrun')} trials not run (rerun to resume)")
            labels = ann.results_to_labels(resolve_path(args.out))
            labels.to_csv(resolve_path(args.out).with_suffix(".labels.csv"), index=False)
            return 0
        return asyncio.run(main())

    if args.step == "phase1":
        from .trial_outcomes.phase1 import phase1_progression
        statuses = tuple(s.strip() for s in args.statuses.split(",") if s.strip()) or None
        res = phase1_progression(pd.read_parquet(ann.mapping_path(config)), statuses=statuses)
        res.to_csv(resolve_path(args.out), index=False)
        print(f"{len(res)} Phase I trials (statuses: {', '.join(statuses) if statuses else 'all'})")
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
        # registry status for the stopped-early exclusion when pred/ref lack it
        # (e.g. TDC references, agent labels): released labels, else the mapping
        parts = [released[["nct_id", "status"]]] if "status" in released.columns else []
        mp = ann.mapping_path(config)
        if mp.exists():
            m = pd.read_parquet(mp)
            if {"nct_id", "status"} <= set(m.columns):
                parts.append(m[["nct_id", "status"]])
        status = pd.concat(parts, ignore_index=True) if parts else None
        print(val.agreement_report(pred, ref, exclude_stopped=args.exclude_stopped,
                                   status=status).to_string(index=False))
        return 0

    if args.step == "features":
        from .trial_outcomes import features
        df = features.compute_features_from_h5ad(args.h5ad, tissue_key=args.tissue_key,
                                                 celltype_key=args.celltype_key,
                                                 gene_id_column=args.gene_id_column, layer=args.layer,
                                                 normalize=args.normalize, kurtosis=args.kurtosis,
                                                 bias_correction=not args.no_bias_correction)
        df.to_csv(resolve_path(args.out))
        print(f"wrote {len(df)} genes to {args.out}")
        return 0

    if args.step == "replicate":
        from ..data.zenodo import zenodo_root
        from .trial_outcomes.replicate import replicate_case1

        root = Path(args.zenodo_dir) if args.zenodo_dir else zenodo_root(config)
        rep = replicate_case1(root, resolve_path(args.out), n_perm=args.n_perm, n_perm_beta=args.n_perm_beta,
                              gene_perm=args.gene_perm, mixed=not args.no_mixed, expr=not args.no_expr)
        print(rep.summary_md)
        print(f"\nwrote {resolve_path(args.out)}/case1_replication_comparison.csv and summary")
        return 0

    if args.step == "benchmarks":
        from ..data.zenodo import zenodo_root
        from .trial_outcomes.benchmarks import benchmark_agreement, format_table_s2, load_benchmarks

        root = Path(args.zenodo_dir) if args.zenodo_dir else zenodo_root(config)
        data = load_benchmarks(root, labels=args.labels, manual=args.manual, tdc=args.tdc)
        table = benchmark_agreement(data)
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(format_table_s2(table, data))
        print(f"\nwrote {out}")
        return 0

    if args.step == "stats":
        from .trial_outcomes.pipeline import run_stats
        table = run_stats(config, features_path=resolve_path(args.features), labels=args.labels,
                          genetic_pairs=args.genetic_pairs, n_perm=args.n_perm, gene_perm=args.gene_perm,
                          out_dir=resolve_path(args.out).parent,
                          definitions=getattr(args, "definitions", "authors"))
        table.to_csv(resolve_path(args.out), index=False)
        print(table.to_string(index=False))
        return 0
    return 2


# ---------------------------------------------------------------- scenarios

def scenario_config(args, scenario: dict[str, Any]) -> dict[str, Any]:
    """Config for a scenario: the global CLI overrides (--model, --runs-dir, --no-web, profiles)
    plus the scenario's profiles (cli.build_config(args, extra_profiles=...) when available)."""
    from .. import cli

    extra = list(scenario.get("profiles") or [])
    build = getattr(cli, "build_config", None)
    if callable(build):
        return build(args, extra_profiles=extra)
    import argparse
    ns = argparse.Namespace(**vars(args))
    ns.profile = list(dict.fromkeys(list(getattr(args, "profile", None) or []) + extra))
    return cli._config(ns)


def _scenario(args, config: dict[str, Any]) -> int:
    from . import scenarios as sc

    if args.action == "list":
        for name, s in sc.list_scenarios().items():
            src = "paper+synthetic" if s.get("paper_turns") else "synthetic"
            print(f"{name:16s} {s['title']}  (profiles: {', '.join(s.get('profiles') or []) or '-'}; turns: {src})")
        return 0
    scen = sc.load_scenario(args.name)
    config = scenario_config(args, scen)

    async def main() -> int:
        from ..cli import _printer
        console, on_event = _printer(args.verbose)
        if args.action == "run":
            for w in sc.live_source_warnings(config, scen):
                console.print(f"[yellow]! no-web scenario: {w}[/]")
            run, _ = await sc.run_scenario(config, scen, turns=args.turns, on_event=on_event,
                                           start_mcp=not args.no_mcp, turns_source=args.turns_source)
            console.print(f"\nRun: {run.dir}  cost ${run.cost.total_usd:.2f} "
                          f"(paper: ${scen.get('paper_cost_usd', '?')})")
            run_dir = run.dir
            if not args.score:
                return 0
        else:
            from ..config import resolve_path
            run_dir = Path(args.run_dir)
            if not run_dir.exists():
                try:
                    from ..audit.index import resolve_run
                    run_dir = Path(resolve_run(args.run_dir, resolve_path(config["paths"]["runs_dir"])))
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]run not found: {args.run_dir} ({exc})[/]")
                    return 2
        score = await sc.score_scenario_run(config, run_dir, scen)
        console.print_json(json.dumps(score, default=str))
        console.print(f"Score written to {Path(run_dir) / 'report' / 'scenario_score.json'}")
        return 0

    return asyncio.run(main())
