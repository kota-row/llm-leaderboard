"""Collect all *leaderboard_table artifacts from archived W&B runs and aggregate into a single run.

Usage:
    python scripts/merge_runs.py --entity <entity> --project <project>
    python scripts/merge_runs.py --entity <entity> --project <project> --run-id <id>
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import wandb


def _fetch_table_from_artifact(
    api: wandb.Api, entity: str, project: str, run_id: str, table_name: str
) -> pd.DataFrame | None:
    artifact_path = f"{entity}/{project}/run-{run_id}-{table_name}:latest"
    try:
        artifact = api.artifact(artifact_path)
        return artifact.get(table_name).get_dataframe()
    except Exception as e:
        print(f"    [{table_name}] not found ({e})")
        return None


def list_leaderboard_table_names(run) -> list[str]:
    """Return *leaderboard_table and subcategory_table_* keys from run.summary."""
    return [
        k for k in run.summary.keys()
        if k.endswith("leaderboard_table") or k.startswith("subcategory_table_")
    ]


def fetch_existing_tables(
    api: wandb.Api, entity: str, project: str, run_id: str
) -> dict[str, pd.DataFrame]:
    """Fetch all *leaderboard_table DataFrames from the destination run."""
    run_obj = api.run(f"{entity}/{project}/{run_id}")
    table_names = list_leaderboard_table_names(run_obj)
    result = {}
    def _fetch(name):
        return name, _fetch_table_from_artifact(api, entity, project, run_id, name)

    with ThreadPoolExecutor(8) as pool:
        for name, df in pool.map(_fetch, table_names):
            if df is not None:
                result[name] = df
    if result:
        print(f"  Loaded {len(result)} existing table(s): {sorted(result)}")
    return result


def prune_existing_tables(
    existing: dict[str, pd.DataFrame], current_run_ids: set[str]
) -> tuple[dict[str, pd.DataFrame], int]:
    """Remove rows whose source run is no longer in the archived run set.

    Returns the pruned dict and total number of removed rows across all tables.
    """
    pruned = {}
    total_removed = 0
    for table_name, df in existing.items():
        if "_source_run_id" not in df.columns:
            pruned[table_name] = df
            continue
        mask = df["_source_run_id"].astype(str).isin(current_run_ids)
        removed = int((~mask).sum())
        if removed:
            removed_ids = df.loc[~mask, "_source_run_id"].tolist()
            print(f"  [{table_name}] Removing {removed} row(s) no longer in archived: {removed_ids}")
        total_removed += removed
        pruned[table_name] = df[mask].reset_index(drop=True)
    return pruned, total_removed


def fetch_archived_runs(api: wandb.Api, entity: str, project: str) -> list:
    runs = list(api.runs(f"{entity}/{project}", filters={"tags": {"$in": ["archived", "leaderboard"]}}))
    print(f"Found {len(runs)} run(s) tagged 'archived' or 'leaderboard'")
    return runs


def collect_new_tables(
    api: wandb.Api,  # kept for _fetch_table_from_artifact
    entity: str,
    project: str,
    archived_runs: list,
    skip_run_ids: set[str],
) -> tuple[dict[str, list[pd.DataFrame]], list]:
    """Download all *leaderboard_table artifacts for archived runs not yet ingested.

    Returns (dict of table_name -> list of DataFrames, list of processed run objects).
    """
    tables_by_name: dict[str, list[pd.DataFrame]] = {}
    processed_runs = []

    target_runs = []
    for run in archived_runs:
        if run.id in skip_run_ids:
            print(f"  Skip (already ingested): {run.id}  ({run.name})")
        else:
            target_runs.append(run)

    def _fetch_run(run):
        table_names = list_leaderboard_table_names(run)
        if not table_names:
            print(f"  {run.id} ({run.name}): No *leaderboard_table artifacts found")
            return run, {}
        print(f"  Processing: {run.id}  ({run.name})  [{len(table_names)} tables]")

        def _fetch(name):
            return name, _fetch_table_from_artifact(api, entity, project, run.id, name)

        result = {}
        with ThreadPoolExecutor() as pool:
            for name, df in pool.map(_fetch, table_names):
                if df is not None:
                    df["_source_run_id"] = run.id
                    df["_source_run_name"] = run.name
                    result[name] = df
        return run, result

    with ThreadPoolExecutor(max_workers=4) as pool:
        for run, fetched in pool.map(_fetch_run, target_runs):
            if fetched:
                for name, df in fetched.items():
                    tables_by_name.setdefault(name, []).append(df)
                processed_runs.append(run)

    return tables_by_name, processed_runs



def create_dummy_run(
    entity: str,
    project: str,
    combined_tables: dict[str, pd.DataFrame],
    run_name: str,
) -> None:
    """Create a new run with 0-row tables derived from combined_tables schema."""
    run = wandb.init(entity=entity, project=project, name=run_name, job_type="merge_runs")
    try:
        # Drop internal tracking columns added by this script
        payload = {
            name: wandb.Table(dataframe=pd.DataFrame(
                columns=[c for c in df.columns if c not in ("_source_run_id", "_source_run_name")]
            ))
            for name, df in sorted(combined_tables.items())
        }
        run.log(payload)
        print(f"Logged {len(payload)} empty table(s) -> {run.url}")
    finally:
        run.finish()


def tag_runs_as_merged(runs: list) -> None:
    """Add 'merged' tag to each run that was successfully ingested."""
    for run in runs:
        if "merged" not in run.tags:
            run.tags.append("merged")
            run.update()
            print(f"  Tagged 'merged': {run.id}  ({run.name})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate all *leaderboard_table artifacts from archived W&B runs into a single run"
    )
    parser.add_argument("--entity", default="llm-leaderboard", help="W&B entity")
    parser.add_argument("--project", default="nejumi-leaderboard4", help="W&B project")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing run ID to resume and overwrite (omit to create a new run)",
    )
    parser.add_argument("--run-name", default="merged_runs", help="Name for a newly created run")
    parser.add_argument("--force", action="store_true", help="Re-fetch all runs, ignoring existing table")
    parser.add_argument("--create-dummy", action="store_true", help="Create a new run with 0-row tables (schema only, no _source_run_* columns)")
    args = parser.parse_args()

    wandb.login()
    api = wandb.Api()

    # Fetch current archived run list first (used for both pruning and new collection)
    print(f"\nFetching archived runs from {args.entity}/{args.project} ...")
    archived_runs = fetch_archived_runs(api, args.entity, args.project)
    current_run_ids = {r.id for r in archived_runs}

    # Load and prune existing tables from the destination run (if --run-id is given)
    # skip_run_ids is derived from leaderboard_table (the main summary table)
    existing_tables: dict[str, pd.DataFrame] = {}
    rows_pruned = 0
    skip_run_ids: set[str] = set()
    if args.run_id and not args.force:
        print(f"\nChecking existing tables in run {args.run_id} ...")
        existing_tables = fetch_existing_tables(api, args.entity, args.project, args.run_id)
        if existing_tables:
            existing_tables, rows_pruned = prune_existing_tables(existing_tables, current_run_ids)
            ref_table = existing_tables.get("leaderboard_table")
            if ref_table is None:
                ref_table = next(iter(existing_tables.values()))
            if "_source_run_id" in ref_table.columns:
                skip_run_ids = set(ref_table["_source_run_id"].dropna().astype(str))
                print(f"  Will skip {len(skip_run_ids)} already-ingested run(s)")
    elif args.force:
        print("\n--force: skipping existing table, re-fetching all runs")

    # Download tables for new runs
    print(f"\nCollecting new runs ...")
    new_tables_by_name, new_runs = collect_new_tables(
        api, args.entity, args.project, archived_runs, skip_run_ids
    )

    # Determine whether an upload is needed
    if not new_tables_by_name and not existing_tables:
        print("No leaderboard tables found. Exiting.")
        return
    if not new_tables_by_name and rows_pruned == 0:
        print("No changes detected. Exiting.")
        return

    # Combine existing + new per table name
    all_table_names = sorted(set(existing_tables) | set(new_tables_by_name))
    combined_tables: dict[str, pd.DataFrame] = {}
    for table_name in all_table_names:
        parts = []
        if table_name in existing_tables and len(existing_tables[table_name]) > 0:
            parts.append(existing_tables[table_name])
        parts.extend(new_tables_by_name.get(table_name, []))
        combined_tables[table_name] = pd.concat(parts, ignore_index=True)
        print(f"  {table_name}: {len(combined_tables[table_name])} row(s)")

    # Init / resume destination run
    init_kwargs: dict = dict(
        entity=args.entity,
        project=args.project,
        job_type="merge_runs",
    )
    if args.run_id:
        init_kwargs["id"] = args.run_id
        init_kwargs["resume"] = "allow"
        print(f"\nResuming run: {args.run_id}")
    else:
        init_kwargs["name"] = args.run_name
        print(f"\nCreating new run: {args.run_name}")

    run = wandb.init(**init_kwargs)
    try:
        payload = {name: wandb.Table(dataframe=df) for name, df in combined_tables.items()}
        run.log(payload)
        print(f"Logged {len(payload)} table(s) -> {run.url}")
    finally:
        run.finish()

    if new_runs:
        print(f"\nTagging {len(new_runs)} run(s) as 'merged' ...")
        tag_runs_as_merged(new_runs)

    if args.create_dummy:
        print(f"\nCreating dummy run ...")
        create_dummy_run(args.entity, args.project, combined_tables, "dummy_run")


if __name__ == "__main__":
    main()
