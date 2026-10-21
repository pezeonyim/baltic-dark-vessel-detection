"""Sanctioned and watch-listed vessels from OpenSanctions.

Source: OpenSanctions maritime export, free for non-commercial use, CC BY-NC 4.0.
https://www.opensanctions.org/datasets/maritime/  It combines sanctions lists (US OFAC,
EU, UK, ...) with other maritime watchlists, including Ukraine's GUR War & Sanctions
shadow fleet catalogue (dataset `ua_war_sanctions`). Each row says which lists it is on.

Matching to AIS:
- IMO number first: identifies the hull and survives renames and reflagging. Both sides
  must pass the IMO check digit, because AIS IMO fields are often junk (0, 1234567).
- MMSI only as a fallback when the AIS record has no valid IMO; weaker (MMSIs change
  and are sometimes spoofed), so it is reported as a separate match type.

A match means "this vessel is on a published list", never "this vessel is doing
something wrong now". The map shows the list names so users can judge.

Categories. The export mixes very different records, so each listed vessel
gets the most serious category among its source lists:
  sanctioned    government sanctions lists (OFAC, EU, UK, Canada, ...)
  shadow_fleet  Ukraine GUR War & Sanctions catalogue (an intelligence watchlist, not a
                legal sanction)
  detention     port state control inspections/detentions (Tokyo, Paris, Black Sea,
                Abuja... MoUs): safety records, NOT sanctions; shown neutrally
  other         anything else
Only sanctioned and shadow_fleet are flagged on the map.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from maritime.paths import REFERENCE

LATEST = "https://data.opensanctions.org/datasets/latest/maritime"
LIST_PATH = REFERENCE / "opensanctions_maritime.csv"
META_PATH = REFERENCE / "opensanctions_maritime.json"

# Header names vary slightly between exports; accept the likely spellings.
COLUMN_ALIASES = {
    "id": ("id", "opensanctions_id", "entity_id"),
    "type": ("type", "schema"),
    "name": ("caption", "name"),
    "imo": ("imo", "imo_number", "imonumber", "imo_code"),
    "mmsi": ("mmsi", "mmsi_number"),
    "risk": ("risk", "risks", "topics"),
    "flag": ("flag", "flags"),
    "countries": ("countries", "country"),
    "datasets": ("datasets", "sources", "dataset"),
    "url": ("url", "opensanctions_url"),
    "aliases": ("aliases", "alias"),
}

# Friendly names for common OpenSanctions topics and source datasets; unknown values
# are shown as they are.
RISK_LABELS = {
    "sanction": "Sanctioned",
    "sanction.linked": "Linked to sanctioned party",
    "sanction.counter": "Counter-sanctioned",
    "mare.shadow": "Shadow fleet",
    "mare.detained": "Detained vessel",
    "mare.sts": "Ship-to-ship transfers",
    "poi": "Person/entity of interest",
    "debarment": "Debarred",
}
FLAGGED_CATEGORIES = ("sanctioned", "shadow_fleet")
CATEGORY_LABELS = {
    "sanctioned": "Sanctioned",
    "shadow_fleet": "Shadow fleet (Ukraine GUR list)",
    "detention": "Port state detention / inspection record",
    "other": "Other watchlist",
}


def category_of(datasets: list[str], risks: list[str]) -> str:
    ds = [d.lower() for d in datasets]
    if "sanction" in risks or any(_is_sanctions_list(d) for d in ds):
        return "sanctioned"
    if "ua_war_sanctions" in ds or "mare.shadow" in risks:
        return "shadow_fleet"
    if any("_mou" in d or d.startswith("mou") or "psc" in d or "detention" in d for d in ds):
        return "detention"
    return "other"


def _is_sanctions_list(d: str) -> bool:
    if "mou" in d or "psc" in d or d == "ua_war_sanctions":
        return False
    return ("sanction" in d or d.startswith(("us_ofac", "eu_fsf", "gb_hmt", "gb_fcdo", "us_trade_csl",
                                             "ca_dfatd", "au_dfat", "ch_seco", "jp_mof", "nz_")))


DATASET_LABELS = {
    "us_ofac_sdn": "US OFAC SDN",
    "eu_fsf": "EU Financial Sanctions",
    "eu_journal_sanctions": "EU Official Journal sanctions",
    "gb_hmt_sanctions": "UK HM Treasury",
    "gb_fcdo_sanctions": "UK FCDO",
    "ua_war_sanctions": "Ukraine GUR War & Sanctions",
    "ca_dfatd_sema_sanctions": "Canada SEMA",
    "au_dfat_sanctions": "Australia DFAT",
    "ch_seco_sanctions": "Switzerland SECO",
    "nz_russia_sanctions": "New Zealand Russia sanctions",
    "jp_mof_sanctions": "Japan MOF",
    "us_trade_csl": "US Consolidated Screening List",
    "tokyo_mou_detention": "Tokyo MoU detention",
    "ext_tokyo_mou_psc": "Tokyo MoU inspection",
    "black_sea_mou_detention": "Black Sea MoU detention",
    "abuja_mou_detention": "Abuja MoU detention",
    "ext_abuja_mou_psc": "Abuja MoU inspection",
    "paris_mou_detention": "Paris MoU detention",
}


def valid_imo(value) -> str | None:
    """Return the 7-digit IMO if it passes the check digit, else None.
    Check: digits d1..d7, sum(d_i * (8 - i) for i in 1..6) % 10 == d7."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 7 or digits in ("0000000", "1234567"):  # common AIS placeholders
        return None
    check = sum(int(d) * w for d, w in zip(digits[:6], range(7, 1, -1), strict=True)) % 10
    return digits if check == int(digits[6]) else None


def valid_mmsi(value) -> int | None:
    digits = re.sub(r"\D", "", str(value or ""))
    return int(digits) if len(digits) == 9 and digits[0] in "234567" else None


def _cell(row: dict, col: dict, key: str) -> str | None:
    return row.get(col[key]) if col[key] else None


def _split(v: str | None) -> list[str]:
    return [x.strip() for x in re.split(r"[;|]", v or "") if x.strip()]


@dataclass
class ListedVessel:
    id: str
    name: str
    imos: list[str] = field(default_factory=list)
    mmsis: list[int] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    url: str | None = None

    @property
    def category(self) -> str:
        return category_of(self.datasets, self.risks)

    def as_label(self, match: str) -> dict:
        cat = self.category
        return {
            "listed": cat in FLAGGED_CATEGORIES,
            "listed_category": cat,
            "listed_category_label": CATEGORY_LABELS[cat],
            "listed_match": match,
            "listed_name": self.name,
            "listed_risks": [RISK_LABELS.get(r, r) for r in self.risks],
            "listed_sources": [DATASET_LABELS.get(d, d) for d in self.datasets],
            "listed_url": self.url if (self.url or "").startswith("https://www.opensanctions.org/") else None,
        }


class Watchlist:
    def __init__(self, vessels: list[ListedVessel], meta: dict | None = None):
        self.vessels = vessels
        self.meta = meta or {}
        self.by_imo = {i: v for v in vessels for i in v.imos}
        self.by_mmsi = {m: v for v in vessels for m in v.mmsis}

    def __len__(self) -> int:
        return len(self.vessels)

    def match(self, imo=None, mmsi=None) -> dict | None:
        i = valid_imo(imo)
        if i and i in self.by_imo:
            return self.by_imo[i].as_label("imo")
        if not i:
            m = valid_mmsi(mmsi)
            if m and m in self.by_mmsi:
                return self.by_mmsi[m].as_label("mmsi")
        return None


def load(path: Path = LIST_PATH) -> Watchlist:
    if not path.exists():
        return Watchlist([])
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {h.strip().lower(): h for h in (reader.fieldnames or [])}
        col = {k: next((headers[a] for a in aliases if a in headers), None)
               for k, aliases in COLUMN_ALIASES.items()}
        if not col["imo"] and not col["mmsi"]:
            raise ValueError(f"{path}: no IMO or MMSI column; headers are {list(headers)}")
        vessels = []
        for row in reader:
            def get(k, row=row):
                return _cell(row, col, k)
            if col["type"] and (get("type") or "").strip().lower() not in ("vessel", ""):
                continue  # the export also lists companies with IMO company numbers
            imos = [i for i in (valid_imo(x) for x in _split(get("imo"))) if i]
            mmsis = [m for m in (valid_mmsi(x) for x in _split(get("mmsi"))) if m]
            if not imos and not mmsis:
                continue
            vessels.append(ListedVessel(
                id=get("id") or "", name=(get("name") or "").strip(), imos=imos, mmsis=mmsis,
                risks=_split(get("risk")), datasets=_split(get("datasets")),
                flags=_split(get("flag")), url=get("url"),
            ))
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() and path == LIST_PATH else {}
    return Watchlist(vessels, meta)


def update(dest: Path = LIST_PATH, timeout: int = 60) -> dict:
    """Download the latest maritime export. Finds the CSV via the dataset index, falls
    back to the conventional file name. Writes atomically; keeps the old file on error."""
    url = f"{LATEST}/maritime.csv"
    try:
        idx = requests.get(f"{LATEST}/index.json", timeout=timeout)
        idx.raise_for_status()
        for res in idx.json().get("resources", []):
            if str(res.get("name", "")).endswith(".csv") and "maritime" in str(res.get("name")):
                url = res["url"]
                break
    except (requests.RequestException, ValueError):
        pass  # fall back to the conventional URL
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.tmp")
    tmp.write_bytes(r.content)
    wl = load(tmp)                      # validate before replacing the old file
    if len(wl) == 0:
        tmp.unlink()
        raise RuntimeError(f"downloaded file from {url} contained no vessels; kept old list")
    tmp.replace(dest)
    meta = {"source": url, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha1": hashlib.sha1(r.content).hexdigest(), "vessels": len(wl),
            "license": "CC BY-NC 4.0, OpenSanctions.org"}
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta
