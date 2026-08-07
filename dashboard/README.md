# SMOKE dashboard

Four pages: Player, Leaderboard, Movers, Methods. Reads the validation outputs in `data/v2/validation/` and the v1 rank table; no additional processing. Keep unlisted until Phase 5 goes public.

Run from the repo root:

```
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

Theme colors live in `.streamlit/config.toml` and match the one-pager and the capstone documents (navy `#1F3864`, blue `#2E75B6`).

Prerequisite: the validation CSVs must exist. If they do not, run the scripts in the order listed in `VALIDATION.md` under Reproducibility.
