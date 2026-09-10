"""Name normalization and display."""

from __future__ import annotations

from src.features.names import display_name, norm_name


def test_norm_name_aligns_sources():
    assert norm_name("Dirk Nowtizski") == "dirk nowitzki"
    assert norm_name("Tim Hardaway Jr.") == "tim hardaway"
    assert norm_name("Ömer Aşık") == "omer asik"
    assert norm_name("C.J. McCollum") == "cj mccollum"
    assert norm_name("dwayne wade") == norm_name("Dwyane Wade")


def test_display_name_casing():
    assert display_name("time hardaway jr") == "Tim Hardaway Jr."
    assert display_name("cj mccollum") == "C.J. McCollum"
    assert display_name("kentavious caldwell-pope") == "Kentavious Caldwell-Pope"
    assert display_name("zach lavine") == "Zach LaVine"
    assert display_name("pj tucker") == "P.J. Tucker"
    assert display_name("ben mclemore") == "Ben McLemore"
