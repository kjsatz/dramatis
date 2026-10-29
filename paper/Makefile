# Build the paper with latexmk (MacTeX). Output: build/main.pdf
TEXBIN ?= /Library/TeX/texbin
export PATH := $(TEXBIN):$(PATH)

LATEXMK = $(TEXBIN)/latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build

pdf:
	$(LATEXMK) main.tex

# Rebuild on every save (open build/main.pdf in Skim or Preview to follow along)
watch:
	$(LATEXMK) -pvc -view=none main.tex

# Redraw the figures from the analysis outputs (scripts/paper_fig.py, from the repository root, DRAMATIS);
# figures/SOURCE.json records the commit that drew them
DRAMATIS ?= ..
figures:
	cd $(DRAMATIS) && uv run python scripts/paper_fig.py --out $(CURDIR)/figures
	$(MAKE) fig6

# Figure 6(a), the vector cartoon (art/fig6_mechanism.tex, TikZ), drawn into the place paper_fig.py leaves for it in
# figures/fig6_mechanism_base.pdf: lualatex, for the Arial of the matplotlib panels; Ghostscript for the review PNG
fig6:
	mkdir -p build/art
	$(TEXBIN)/lualatex -interaction=nonstopmode -halt-on-error -output-directory=build/art art/fig6_mechanism.tex
	cp build/art/fig6_mechanism.pdf figures/
	gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r600 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
		-o figures/fig6_mechanism.png figures/fig6_mechanism.pdf

# Regenerate the number-bearing table bodies (scripts/paper_tables.py); tables/SOURCE.json records the commit.
# Table A5, the corpora (tables/tableA5_corpora.tex), is kept by hand.
tables:
	cd $(DRAMATIS) && uv run python scripts/paper_tables.py --out $(CURDIR)/tables

# Figures 1 and 2 (art/svg_figs.py: hand-laid SVG, printed to PDF by headless Chrome) and Figure A5 (art/fig1_design.py,
# matplotlib). Figure 2's data file (art/fig1_example.json) comes from scripts/fig1_example.py.
art:
	cd $(DRAMATIS) && uv run python $(CURDIR)/art/fig1_design.py && uv run python $(CURDIR)/art/svg_figs.py

clean:
	rm -rf build

.PHONY: pdf watch figures fig6 tables art clean

# The arXiv upload, build/arxiv.zip: the files LaTeX read (latexmk's recorder file) and the bibliography (arXiv does
# not run BibTeX), then compiled on their own, as arXiv will, with no undefined citation or reference allowed
arxiv: pdf
	rm -rf build/arxiv build/arxiv.zip build/arxiv-test
	grep '^INPUT ' build/main.fls | sed 's/^INPUT //; s|^\./||' | grep -v -e '^/' -e '^build/' | sort -u > build/arxiv.files
	rsync -a --files-from=build/arxiv.files ./ build/arxiv/
	cp build/main.bbl build/arxiv/
	cd build/arxiv && zip -qrX ../arxiv.zip .
	mkdir build/arxiv-test && cd build/arxiv-test && unzip -q ../arxiv.zip && \
		for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || exit 1; done && \
		! grep -E 'Warning: (Citation|Reference) .* undefined' main.log
	@echo "build/arxiv.zip: $$(find build/arxiv -type f | wc -l | tr -d " ") files, $$(du -h build/arxiv.zip | cut -f1); compiles on its own to $$(grep -o '[0-9]* pages' build/arxiv-test/main.log | tail -1)"

.PHONY: arxiv
