# 2026-09-10

Synthesized reasonable starting PLAN.md from interactive Claude conversation.

# 2026-09-11

Wrote preliminary CLAUDE.md and got started on phase 0.

Claude: Set up the repo skeleton (no model/training code yet): `pyproject.toml`
for uv with Python 3.12–3.13 and hatchling packaging (`dramatis/` package,
loose bounds in pyproject, exact pins in `uv.lock`); directory layout from
CLAUDE.md (`dramatis/`, `scripts/`, `configs/`, `prompts/`, `data/`,
`checkpoints/`, `runs/`, `docs/`); README with setup instructions and provenance
conventions; `.env.example` with Gemini (AI Studio or Vertex) and HF Hub
variables. Deps chosen: torch, numpy, matplotlib, pyyaml, python-dotenv, tqdm,
google-genai, huggingface-hub; dev: pytest, ruff.
