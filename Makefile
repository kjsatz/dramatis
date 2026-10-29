# The paper's figures, tables and PDF, built in paper/ (its Makefile has the details), from this directory:
#   make figures tables    after `uv run python scripts/download.py figures`
#   make art pdf           Figures 1, 2 and A4 (headless Chrome), and the PDF (a TeX distribution)
figures tables art pdf:
	$(MAKE) -C paper $@

.PHONY: figures tables art pdf
