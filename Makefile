# The paper's figures, tables and PDF, built in paper/ (its Makefile has the details), from this directory:
#   make figures tables    after `uv run python scripts/download.py figures`
#   make art pdf           Figures 1, 2 and A5 (headless Chrome), and the PDF (a TeX distribution)
#   make arxiv             paper/build/arxiv.zip, the arXiv upload, test-compiled on its own
figures tables art pdf arxiv:
	$(MAKE) -C paper $@

.PHONY: figures tables art pdf arxiv
