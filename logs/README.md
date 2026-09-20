# logs/

Per-session working outputs. Policy (see `.gitignore`):

* `*.log` — training / driver logs, never tracked.
* `**/*.json` — per-item probe outputs (`cast_probe.py --out`, `story_level_ft.py --out`, …), the
  data behind the figures; kept locally and released with the checkpoints, not tracked (they
  run to tens of MB per session). Every report cites the JSON it was made from.
* `**/*.txt`, `*.md` — the human-readable summaries printed by the same probes; tracked.
* `**/*.sh` — one-off box/Mac drivers written during a session; tracked for provenance.

Layout: `probes/` (Stage 2a box probes), `cast/` (Stage 2b: the persistent cast; `box/` = the
A4/A5/B session, `map/` = the coherence map, `sweep/` = the transition's coarse cut and 30M/100M
seeds, `seeds/` = 15 y values × 10 seeds, `sym/` = the symmetric-temptation corpus).
