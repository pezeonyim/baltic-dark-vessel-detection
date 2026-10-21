"""Watchlist parsing and matching on a synthetic OpenSanctions-style CSV."""

from maritime import sanctions

CSV = """type,caption,imo,risk,countries,flag,mmsi,id,url,datasets,aliases
Vessel,SOYANA,9327372,sanction;mare.shadow,ru,ru,273120300,NK-1,https://www.opensanctions.org/entities/NK-1/,eu_journal_sanctions;ua_war_sanctions,
Vessel,NO IMO TANKER,,sanction,ru,ga,626000123,NK-2,https://www.opensanctions.org/entities/NK-2/,us_ofac_sdn,
Company,SOME SHIPPING LLC,5987654,sanction,ru,,,NK-3,https://www.opensanctions.org/entities/NK-3/,us_ofac_sdn,
Vessel,BAD IMO,1234567,sanction,,,,NK-4,javascript:alert(1),us_ofac_sdn,
Vessel,DETAINED SHIP,9074729,,,,,NK-5,https://www.opensanctions.org/entities/NK-5/,tokyo_mou_detention;ext_tokyo_mou_psc,
Vessel,GUR ONLY,9281009,,,,,NK-6,https://www.opensanctions.org/entities/NK-6/,ua_war_sanctions,
"""


def wl(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(CSV)
    return sanctions.load(p)


def test_load_skips_companies_and_junk(tmp_path):
    w = wl(tmp_path)
    assert {v.name for v in w.vessels} == {"SOYANA", "NO IMO TANKER", "DETAINED SHIP", "GUR ONLY"}


def test_imo_match_preferred(tmp_path):
    m = wl(tmp_path).match(imo="9327372", mmsi=999999999)
    assert m["listed_match"] == "imo"
    assert "Shadow fleet" in m["listed_risks"]
    assert "Ukraine GUR War & Sanctions" in m["listed_sources"]


def test_mmsi_fallback_only_without_valid_imo(tmp_path):
    w = wl(tmp_path)
    assert w.match(imo=None, mmsi=626000123)["listed_match"] == "mmsi"
    assert w.match(imo="9123453", mmsi=626000123) is None   # valid, different IMO: no MMSI fallback


def test_categories(tmp_path):
    w = wl(tmp_path)
    s, d, g = w.match(imo="9327372"), w.match(imo="9074729"), w.match(imo="9281009")
    assert s["listed_category"] == "sanctioned" and s["listed"]
    assert d["listed_category"] == "detention" and not d["listed"]        # safety record, not flagged
    assert g["listed_category"] == "shadow_fleet" and g["listed"]


def test_url_allowlist(tmp_path):
    w = wl(tmp_path)
    assert w.match(imo="9327372")["listed_url"].startswith("https://www.opensanctions.org/")


def test_check_digit():
    assert sanctions.valid_imo("IMO9327372") == "9327372"
    assert sanctions.valid_imo("9327371") is None
    assert sanctions.valid_imo("1234567") is None
