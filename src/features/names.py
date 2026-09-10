"""Player-name handling shared by every script that joins on or displays names.

Two jobs, one table each:

* ``norm_name`` collapses a name from any source (Kaggle shot logs, Basketball-
  Reference, our own outputs) to a join key: lowercase, accents and punctuation
  stripped, suffixes removed, and the Kaggle misspellings corrected.
* ``display_name`` renders a lowercase source name for a figure, the one-pager, or
  the dashboard, so every artifact spells a player the same way.

Kaggle's ``player_name`` column carries a handful of misspellings ("dirk nowtizski",
"dwayne wade"); the ``player_id`` column is correct and is the preferred join key
wherever both sides carry it (the two-season panel in stability.py does this).
This module exists for the joins against Basketball-Reference, which has no NBA ID.
"""

from __future__ import annotations

import re
import unicodedata

# Kaggle misspellings, and real cross-source name differences (a legal name change).
NAME_FIXES = {
    "dirk nowtizski": "dirk nowitzki",
    "danilo gallinai": "danilo gallinari",
    "mnta ellis": "monta ellis",
    "jimmer dredette": "jimmer fredette",
    "dwayne wade": "dwyane wade",
    "steve adams": "steven adams",
    "time hardaway": "tim hardaway",
    "nerles noel": "nerlens noel",
    "beno urdih": "beno udrih",
    "jon ingles": "joe ingles",
    "jose juan barea": "jj barea",
    "jose barea": "jj barea",
    "nene hilario": "nene",              # Basketball-Reference lists him mononymously
    "enes kanter": "enes freedom",       # legal name change; datasets disagree
}

# letters that NFKD does NOT fold to ASCII (base letters, not accented forms)
_CHAR_MAP = str.maketrans({"ı": "i", "ø": "o", "đ": "d", "ł": "l", "ß": "ss"})


def norm_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes so two sources' names align."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip().translate(_CHAR_MAP)    # ömer aşık -> omer asik
    s = re.sub(r"[.'`]", "", s)                   # cj mccollum, dangelo russell
    s = re.sub(r"[-]", " ", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NAME_FIXES.get(s, s)


# Names the source data spells wrong, or that no casing rule can derive.
DISPLAY_FIX = {
    "lebron james": "LeBron James",
    "demarcus cousins": "DeMarcus Cousins",
    "deandre jordan": "DeAndre Jordan",
    "demarre carroll": "DeMarre Carroll",
    "amare stoudemire": "Amar'e Stoudemire",
    "dirk nowtizski": "Dirk Nowitzki",
    "nikola vucevic": "Nikola Vucevic",
    "dwayne wade": "Dwyane Wade",
    "time hardaway jr": "Tim Hardaway Jr.",
    "al farouq aminu": "Al-Farouq Aminu",
    "kyle oquinn": "Kyle O'Quinn",
    "zach lavine": "Zach LaVine",
    "michael carter-williams": "Michael Carter-Williams",
    "kentavious caldwell-pope": "Kentavious Caldwell-Pope",
    "karl anthony towns": "Karl-Anthony Towns",
    "nerles noel": "Nerlens Noel",
    "jimmer dredette": "Jimmer Fredette",
    "danilo gallinai": "Danilo Gallinari",
    "mnta ellis": "Monta Ellis",
    "beno urdih": "Beno Udrih",
    "jon ingles": "Joe Ingles",
    "steve adams": "Steven Adams",
}

_INITIALS = {"cj", "dj", "jj", "kj", "oj", "pj", "tj", "aj", "rj"}
_SUFFIX = {"jr": "Jr.", "sr": "Sr.", "ii": "II", "iii": "III", "iv": "IV"}


def _word(w: str, first: bool, last: bool) -> str:
    """Case a single name token, handling initials, Mc/Mac, hyphens, and suffixes."""
    if last and not first and w in _SUFFIX:
        return _SUFFIX[w]
    if first and w in _INITIALS:
        return f"{w[0].upper()}.{w[1].upper()}."
    if "-" in w:
        return "-".join(_word(part, first, False) for part in w.split("-"))
    if w.startswith("mc") and len(w) > 3:
        return "Mc" + w[2:].capitalize()
    return w.capitalize()


def display_name(raw: str) -> str:
    """Render a lowercase source name for display."""
    key = str(raw).strip().lower()
    if key in DISPLAY_FIX:
        return DISPLAY_FIX[key]
    parts = key.split()
    return " ".join(_word(w, i == 0, i == len(parts) - 1) for i, w in enumerate(parts))
