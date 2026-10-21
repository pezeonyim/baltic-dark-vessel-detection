import pandas as pd

from maritime.labels import add_labels, flag_info, mid_of


def test_mid_extraction():
    assert mid_of(273123456) == 273
    assert mid_of(970123456) is None      # SART, not a ship station
    assert mid_of(2731234) is None        # too short
    assert mid_of(None) is None


def test_flag_groups():
    assert flag_info(273000001)["flag_group"] == "russia"
    assert flag_info(413000001)["flag_group"] == "china"
    assert flag_info(477000001)["flag_group"] == "china"      # Hong Kong -> CN
    assert flag_info(219000001)["flag_group"] == "nato"       # Denmark
    assert flag_info(236000001)["flag_group"] == "nato"       # Gibraltar -> GB
    assert flag_info(226000001)["flag_iso2"] == "FR"          # FX fixed
    assert flag_info(351000001)["flag_group"] == "other"      # Panama
    assert flag_info(999000001)["flag_group"] == "unknown"


def test_add_labels_military():
    df = pd.DataFrame({"mmsi": [211000001, 273000001], "vessel_type": [35, 80]})
    out = add_labels(df)
    assert list(out["is_military"]) == [True, False]
    assert list(out["flag_group"]) == ["nato", "russia"]
