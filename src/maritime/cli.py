"""Command line entry point.

    maritime regions
    maritime stream --region baltic          record live AIS (aisstream.io)
    maritime digitraffic --region baltic     second live AIS feed (Fintraffic)
    maritime sar run --hours 48              new Sentinel-1 scenes -> detections matched to AIS
    maritime serve --region baltic           the map
"""

from __future__ import annotations

import argparse
from pathlib import Path

from maritime.config import available_regions, load_region
from maritime.paths import region_ais_dir


def cmd_regions(_: argparse.Namespace) -> None:
    for name in available_regions():
        r = load_region(name)
        b = r.bbox
        print(f"{name:20s} source={r.source.type:20s} "
              f"bbox=({b.min_lon}, {b.min_lat}, {b.max_lon}, {b.max_lat})")


def cmd_summary(args: argparse.Namespace) -> None:
    import duckdb

    region = load_region(args.region)
    ais_dir = region_ais_dir(region.name)
    if not any(ais_dir.glob("*/*.parquet")):
        print(f"no processed data for {region.name} yet; record some with `maritime stream --region {region.name}`")
        return
    src = f"read_parquet('{ais_dir / '*' / '*.parquet'}', hive_partitioning = true, union_by_name = true)"

    overall = duckdb.sql(f"""
        SELECT count(*)                                AS pings,
               count(DISTINCT mmsi)                    AS vessels,
               count(DISTINCT date)                    AS days,
               min(ts)                                 AS first_ts,
               max(ts)                                 AS last_ts,
               round(avg((sog IS NULL)::INT), 3)       AS frac_sog_null,
               round(avg((draught IS NULL)::INT), 3)   AS frac_draught_null
        FROM {src}
    """).df()
    print(f"\n{region.display_name}\n")
    print(overall.T.rename(columns={0: "value"}).to_string())

    by_class = duckdb.sql(f"""
        SELECT coalesce(transceiver_class, '?')        AS class,
               count(DISTINCT mmsi)                    AS vessels,
               count(*)                                AS pings,
               round(avg((draught IS NULL)::INT), 3)   AS frac_draught_null
        FROM {src}
        GROUP BY 1 ORDER BY vessels DESC
    """).df()
    print("\nBy transceiver class (A = commercial, B = small craft)\n")
    print(by_class.to_string(index=False))


def cmd_features(args: argparse.Namespace) -> None:
    import json
    import time

    from maritime.db import ais_source, connect
    from maritime.features import coverage_map
    from maritime.features.coverage import outages, vessel_volume
    from maritime.features.gaps import detect_gaps
    from maritime.paths import region_features_dir

    region = load_region(args.region)
    source_name = region.source.type
    out_dir = region_features_dir(region.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    con, src = connect(), ais_source(region.name)

    t = time.time()
    vol = vessel_volume(con, src)
    out = outages(vol)
    vol.to_parquet(out_dir / "vessel_volume.parquet")
    out.to_parquet(out_dir / "outages.parquet")
    print(f"outages:  {len(out)} ({out['minutes'].sum():.0f} min total)  [{time.time() - t:.0f}s]")

    t = time.time()
    gaps = detect_gaps(con, src, source_name, out, k=args.k)
    gaps.to_parquet(out_dir / "gaps.parquet")
    print(f"gaps:     {len(gaps):,} across {gaps['mmsi'].nunique():,} vessels  [{time.time() - t:.0f}s]")

    t = time.time()
    stats, excess = coverage_map.interval_stats(con, src, source_name, out, k=args.k)
    model = coverage_map.fit(stats, excess)
    coverage_map.cells_table(model).to_parquet(out_dir / "coverage_cells.parquet")
    (out_dir / "coverage_model.json").write_text(json.dumps(
        {"p0": model["p0"], "xi": model["xi"], "sigma": model["sigma"], "k": args.k}, indent=2))
    gaps = coverage_map.score_gaps(gaps, model, k=args.k)
    gaps.to_parquet(out_dir / "gaps.parquet")
    print(f"coverage: GPD xi={model['xi']:.2f} sigma={model['sigma']:.1f}; "
          f"gap surprise median {gaps['surprise'].median():.2f}  [{time.time() - t:.0f}s]")

    print(f"written to {out_dir}")


def cmd_maintain(args: argparse.Namespace) -> None:
    from maritime.maintenance import compact_region, snapshot_live_db

    region = load_region(args.region)
    for line in compact_region(region.name):
        print(line)
    snap = snapshot_live_db(region.name)
    print(f"live db snapshot: {snap}" if snap else "no live db to snapshot")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from maritime.web.app import create_app

    print(f"Map: http://{args.host}:{args.port}  (Ctrl-C to stop)")
    uvicorn.run(create_app(args.region), host=args.host, port=args.port, log_level="warning")


def cmd_sanctions(args: argparse.Namespace) -> None:
    import sqlite3

    from maritime import sanctions
    from maritime.live.aisstream import live_db_path

    if args.action == "update":
        meta = sanctions.update()
        print(f"watchlist updated: {meta['vessels']:,} vessels from {meta['source']}")
        return
    wl = sanctions.load()
    if not len(wl):
        raise SystemExit("no watchlist yet; run `maritime sanctions update`")
    db = live_db_path(args.region)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        rows = con.execute("SELECT mmsi, imo, vessel_name, ts FROM last_position").fetchall()
    hits = [(r, wl.match(r[1], r[0])) for r in rows]
    hits = [(r, m) for r, m in hits if m]
    print(f"{len(hits)} of {len(rows):,} vessels in {args.region} appear in the watchlist export "
          f"({len(wl):,} listed vessels loaded)")
    for cat in ("sanctioned", "shadow_fleet", "detention", "other"):
        group = [(r, m) for r, m in hits if m["listed_category"] == cat]
        if not group:
            continue
        print(f"\n{sanctions.CATEGORY_LABELS[cat]}: {len(group)}"
              + ("" if cat in sanctions.FLAGGED_CATEGORIES else "  (not flagged on the map)"))
        for (mmsi, imo, name, ts), m in sorted(group, key=lambda x: x[0][3], reverse=True):
            print(f"  {name or '?':24s} MMSI {mmsi} IMO {imo or '-':8s} via {m['listed_match']:4s} "
                  f"last seen {ts[:16]}  {', '.join(m['listed_sources'][:3])}")


def _xview3_fusion(base, detections: str | None) -> None:
    """SAR+AIS fusion test on the held-out scenes. Reads the scene-eval output zip or csv."""
    import zipfile

    import pandas as pd

    from maritime.sar import fusion_eval

    path = Path(detections) if detections else base / "scene_eval" / "detections_ensemble.csv.gz"
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z, z.open("detections_ensemble.csv.gz") as f:
            dets = pd.read_csv(f, compression="gzip")
    else:
        dets = pd.read_csv(path)
    labels = pd.read_csv(base / "labels" / "validation.csv")
    labels = labels[labels["scene_id"].isin(dets["scene_id"].unique())]
    out = base / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(dets):,} detections, {dets.scene_id.nunique()} scenes, {len(labels):,} labels")

    rows = []
    for score_min in (0.35, 0.4, 0.45, 0.5, 0.6):
        for p_vessel_min in (0.0, 0.5):
            for jitter, gate in ((0, 200), (100, 500), (300, 1000)):
                s, *_ = fusion_eval.run(dets, labels, score_min=score_min, p_vessel_min=p_vessel_min,
                                        gate_m=gate, ais_jitter_m=jitter)
                rows.append({"score_min": score_min, "p_vessel_min": p_vessel_min, "ais_jitter_m": jitter,
                             "gate_m": gate, **s})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "sweep.csv", index=False)
    s, per, cands, labs = fusion_eval.run(dets, labels, score_min=0.45, p_vessel_min=0.5, gate_m=200)
    per.to_csv(out / "per_scene.csv", index=False)
    cands.to_csv(out / "candidates.csv.gz", index=False)
    labs.to_csv(out / "labels_fate.csv", index=False)
    pd.set_option("display.width", 200)
    print("headline (score >= 0.45, p_vessel >= 0.5, exact AIS positions, 200 m gate):")
    print(pd.Series(s).to_string())
    cols = ["score_min", "p_vessel_min", "ais_jitter_m", "gate_m", "candidates", "precision", "recall",
            "recall_near_shore", "recall_offshore", "fp_infra", "fp_ais_missed", "fp_clutter",
            "miss_not_detected", "miss_absorbed_by_ais", "miss_filtered"]
    print(sweep[cols].round(3).to_string(index=False))
    print(per[["scene_id", "dark_vessels", "candidates", "tp", "fp", "precision", "recall"]].round(3)
          .to_string(index=False))
    print(f"written to {out}")


def cmd_xview3(args: argparse.Namespace) -> None:
    import json

    from maritime.paths import DATA
    from maritime.sar import xview3

    base = DATA / "xview3"
    if args.action == "fusion":
        _xview3_fusion(base, args.detections)
        return
    if args.action == "status":
        man = base / "tiles" / "manifest.jsonl"
        rows = [json.loads(x) for x in man.read_text().splitlines()] if man.exists() else []
        mb = sum(r["npz_mb"] for r in rows)
        print(f"{len(rows)} scenes tiled, {sum(r['objects'] for r in rows):,} objects, "
              f"{sum(r['positive_tiles'] + r['negative_tiles'] for r in rows):,} tiles, {mb:,.0f} MB")
        for src in sorted({r["source"] for r in rows}):
            print(f"  {src}: {sum(r['source'] == src for r in rows)} scenes")
        return
    if args.action == "scenes":
        ids_file = Path(args.scenes_file) if args.scenes_file else base / "heldout_scenes.txt"
        ids = [x.strip() for x in ids_file.read_text().splitlines() if x.strip()]
        xview3.prepare_scenes(base / "links" / f"{args.set}.txt", ids, base / "labels",
                              base / "shoreline" / f"{args.set}.tar.gz", base / args.out_name, base / "work",
                              log=lambda *a: print(*a, flush=True))
        return
    if args.action == "upload-scenes":
        from maritime.sar.kaggle_upload import upload
        kind = "mine" if args.out_name == "scenes_mine" else "scenes"
        print(f"uploaded (private): {upload(base / args.out_name, message=args.out_name, kind=kind)}")
        return
    if args.action == "upload":
        from maritime.sar.kaggle_upload import upload
        url = upload(base / "tiles", message=f"tiles update ({args.set})")
        print(f"uploaded (private): {url}")
        return
    links = base / "links" / f"{args.set}.txt"
    if not links.exists():
        raise SystemExit(f"missing {links}: download it from the xView3 page")
    xview3.prepare(links, base / "labels", base / "tiles", base / "work",
                   max_scenes=args.max_scenes, neg_ratio=args.neg_ratio,
                   log=lambda *a: print(*a, flush=True))   # flush: progress shows up in the log file


def cmd_stream(args: argparse.Namespace) -> None:
    import asyncio
    import logging

    from maritime.live.aisstream import Recorder, live_db_path, load_api_key

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    region = load_region(args.region)
    rec = Recorder(region, load_api_key(), flush_seconds=args.flush_seconds)
    print(f"Recording {region.display_name}. Last positions: {live_db_path(region.name)}")
    print("Ctrl-C to stop (buffered data is flushed on exit).")
    try:
        asyncio.run(rec.run())
    except KeyboardInterrupt:
        print(f"\nstopped; {rec.n_rows_total:,} positions recorded this session")


def cmd_sar(args: argparse.Namespace) -> None:
    """Baltic SAR pipeline."""
    from maritime.sar import run as R

    region = load_region(args.region)
    if args.action == "land":
        R.fetch_land(region)
    elif args.action == "search":
        df = R.search_region(region, args.hours)
        done = R.processed_products(region.name)
        df["done"] = df["name"].str.removesuffix(".SAFE").isin(done)
        cols = ["name", "start", "orbit_direction", "done"]
        if R.land_path(region.name).exists() and len(df):
            from maritime.sar.scene_pipeline import Land
            df = R.add_sea_area(df, Land.load(R.land_path(region.name)), region)
            cols += ["sea_km2", "area_km2"]
            print(f"{int((df.sea_km2 >= R.MIN_SEA_KM2).sum())} of {len(df)} slices have >= {R.MIN_SEA_KM2:,} km2 "
                  "of sea in the region (the others are skipped by `run`)")
        print(df[cols].to_string(index=False) if len(df) else "no products")
    elif args.action == "report":
        from maritime.sar.report import run_report
        run_report(region, hours=args.hours)
    elif args.action == "figure":
        if not args.product:
            raise SystemExit("--product is required for figure")
        from maritime.sar.figure import make
        make(region, args.product)
    elif args.action == "length-train":
        from maritime.sar.length import train_and_save
        train_and_save(region.name)
    elif args.action == "relength":
        from maritime.sar.length import relength
        relength(region.name, hours=args.hours, force=args.force)
    elif args.action == "regate":
        R.regate(region, hours=args.hours)
    elif args.action == "evidence":
        from maritime.sar.evidence import run_evidence
        run_evidence(region, hours=args.hours)
    elif args.action == "gallery":
        R.chip_gallery(region, days=args.hours / 24)
    elif args.action == "bench":
        R.bench()
    elif args.action == "debug":
        if not args.product:
            raise SystemExit("--product is required for debug")
        R.debug_dump(region, args.product)
    elif args.action == "run":
        models = tuple(args.models.split(",")) if args.models else R.DEFAULT_MODELS
        R.run(region, hours=args.hours, max_scenes=args.max_scenes, product=args.product, models=models,
              keep_zip=args.keep_zip, force=args.force, log=lambda *a: print(*a, flush=True))


def cmd_risk(args: argparse.Namespace) -> None:
    if args.later_week:
        from maritime.risk import evaluate_later_week
        evaluate_later_week(load_region(args.region), split=args.later_week, days=args.days)
        return
    from maritime.risk import run_risk
    run_risk(load_region(args.region), days=args.days, end=args.end)


def cmd_digitraffic(args: argparse.Namespace) -> None:
    import logging

    from maritime.live.digitraffic import ATTRIBUTION, Poller

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    region = load_region(args.region)
    p = Poller(region)
    if not args.once:
        print(f"Recording Digitraffic AIS for {region.display_name} ({ATTRIBUTION}). Ctrl-C to stop.")
    try:
        p.run(once=args.once)
    except KeyboardInterrupt:
        print(f"\nstopped; {p.n_total:,} positions recorded this session")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="maritime")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("regions", help="list configured regions").set_defaults(fn=cmd_regions)

    ps = sub.add_parser("summary", help="quick stats on ingested data")
    ps.add_argument("--region", required=True)
    ps.set_defaults(fn=cmd_summary)

    pf = sub.add_parser("features",
                        help="compute outages and gaps (with coverage surprise)")
    pf.add_argument("--region", required=True)
    pf.add_argument("--k", type=float, default=30.0,
                    help="gap = silence longer than k x expected report interval")
    pf.set_defaults(fn=cmd_features)

    pm = sub.add_parser("maintain", help="compact finished days and snapshot the live db")
    pm.add_argument("--region", required=True)
    pm.set_defaults(fn=cmd_maintain)

    pv = sub.add_parser("serve", help="run the map app")
    pv.add_argument("--region", required=True)
    pv.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = only this machine")
    pv.add_argument("--port", type=int, default=8000)
    pv.set_defaults(fn=cmd_serve)

    pw = sub.add_parser("sanctions", help="download or match the OpenSanctions vessel list")
    pw.add_argument("action", choices=["update", "match"])
    pw.add_argument("--region", default="baltic")
    pw.set_defaults(fn=cmd_sanctions)

    px = sub.add_parser("xview3", help="download and tile xView3 SAR scenes")
    px.add_argument("action", choices=["prepare", "status", "upload", "scenes", "upload-scenes", "fusion"])
    px.add_argument("--detections", default=None,
                    help="fusion: scene-eval output zip or detections_ensemble.csv.gz")
    px.add_argument("--scenes-file", default=None,
                    help="scene ids, one per line (default data/xview3/heldout_scenes.txt)")
    px.add_argument("--out-name", default="scenes",
                    help="folder under data/xview3 for whole scenes (scenes = held-out eval, scenes_mine = mining)")
    px.add_argument("--set", default="tiny", help="link list name: tiny, validation, train")
    px.add_argument("--max-scenes", type=int, default=None)
    px.add_argument("--neg-ratio", type=float, default=1.0, help="empty tiles per tile with objects")
    px.set_defaults(fn=cmd_xview3)

    pr = sub.add_parser("sar", help="Sentinel-1 vessel detection + AIS matching")
    pr.add_argument("action", choices=["land", "search", "run", "debug", "bench", "gallery", "report",
                                       "evidence", "regate", "length-train", "relength",
                                       "figure"])
    pr.add_argument("--region", default="baltic")
    pr.add_argument("--hours", type=float, default=36, help="look back this many hours for new scenes")
    pr.add_argument("--max-scenes", type=int, default=None)
    pr.add_argument("--product", default=None, help="process one named product (S1..._XXXX)")
    pr.add_argument("--models", default=None, help="comma-separated weight files in data/models")
    pr.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip (for debugging)")
    pr.add_argument("--force", action="store_true", help="re-process products already in the manifest")
    pr.set_defaults(fn=cmd_sar)

    prk = sub.add_parser("risk", help="vessel risk scores from behaviour, checked against the watchlist")
    prk.add_argument("--region", default="baltic")
    prk.add_argument("--days", type=float, default=7)
    prk.add_argument("--end", default=None, help="end of the window (UTC, default now)")
    prk.add_argument("--later-week", default=None, metavar="SPLIT",
                     help="honest test: train on --days before SPLIT (UTC), score the --days after it")
    prk.set_defaults(fn=cmd_risk)

    pd_ = sub.add_parser("digitraffic", help="record live AIS from Fintraffic Digitraffic")
    pd_.add_argument("--region", default="baltic")
    pd_.add_argument("--once", action="store_true", help="one poll and a summary (connectivity check)")
    pd_.set_defaults(fn=cmd_digitraffic)

    pl = sub.add_parser("stream", help="record live AIS for a region (aisstream.io)")
    pl.add_argument("--region", required=True)
    pl.add_argument("--flush-seconds", type=int, default=60, help="how often to write to disk")
    pl.set_defaults(fn=cmd_stream)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
