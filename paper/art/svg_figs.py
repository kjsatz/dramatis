# ruff: noqa: E501
"""Figures 1 and 2 as hand-laid SVG (placeholders until Kevin redraws them), after Gemini drafts Kevin iterated on
(2026-10-19), with the paper's palette, names and the real numbers: Figure 2's per-token log-probabilities come from
art/fig1_example.json (dramatis scripts/fig1_example.py). Each SVG is printed to PDF by headless Chrome (text stays
text), and a PNG preview is written next to it.

    make art        (from the paper repository; runs with the dramatis environment, for matplotlib's text metrics)

Coordinates: 936 units across the 6.5-inch text width (144 per inch), so a font size of 10 is 5 pt.
"""

import json
import re
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W = 936

# the paper's palette (dramatis scripts/paper_fig.py)
INK, MUTED, GRID, BASE, PAPER = "#333333", "#898781", "#e1e0d9", "#c3c2b7", "#f4f3ef"
PRE = "#7a7973"  # prewritten story text
ORANGE, PALE_ORANGE = "#eb6834", "#fdeee6"
BAD, GOOD = "#c62828", "#1f5fae"
PALE_BAD, PALE_GOOD = "#f6d4d4", "#d6e3f3"
BAR_BAD, BAR_GOOD = "#e08989", "#84a7d2"  # BAD and GOOD at 55 % over white (Figure 2's per-token bars)
NAME_TINT = "#e3e3e3"
FONT = "Arial, Helvetica, sans-serif"
# Math letters (z, s) in Computer Modern italic, as in the matplotlib figures (dramatis scripts/paper_fig.py): the
# font from matplotlib's data, enlarged by the same MATH_SCALE so its lowercase sits with Arial's
MATH_SCALE = 1.1
CMMI = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "cmmi10.ttf"
FONT_FACE = f'@font-face {{ font-family: "cmmi10"; src: url("{CMMI.as_uri()}"); }}'


def mi(s, size):
    """Math letters `s` in Computer Modern italic, for a text of font size `size` (use with raw=True)."""
    return f'<tspan font-family="cmmi10" font-weight="normal" font-size="{size * MATH_SCALE:.1f}">{escape(s)}</tspan>'


def tw(s, size, bold=False):
    """A string's width in SVG units at font size `size` (in the same units), from Arial's metrics."""
    prop = FontProperties(family=["Arial", "DejaVu Sans"], weight="bold" if bold else "normal")
    return TextPath((0, 0), s, size=size, prop=prop).get_extents().width if s.strip() else size * 0.28 * len(s)


def wrap(s, width, size, bold=False):
    lines, cur = [], ""
    for w in s.split():
        t = f"{cur} {w}".strip()
        if cur and tw(t, size, bold) > width:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + [cur]


class Svg:
    def __init__(self, h):
        self.h, self.parts = h, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=10, fill=INK, anchor="start", bold=False, italic=False, raw=False):
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        b = ' font-weight="bold"' if bold else ""
        i = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{a}{b}{i}>{s if raw else escape(s)}</text>')

    def lines(self, x, y, lines, size=10, lh=None, **kw):
        lh = lh or size * 1.22
        for k, ln in enumerate(lines):
            self.text(x, y + k * lh, ln, size, **kw)
        return y + (len(lines) - 1) * lh

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1, rx=0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1, dash=None, arrow=None, round_cap=False):
        if arrow:
            x2, y2 = shaft_end(x1, y1, x2, y2, sw)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ar-{arrow})"' if arrow else ""
        c = ' stroke-linecap="round"' if round_cap else ""  # a fork's two tails: round, overlapping at the shared point
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{m}{c}/>')

    def path(self, d, stroke=INK, sw=1, fill="none", dash=None, arrow=None):
        if arrow:  # straight segments only (M/L): pull the last point back, as in line()
            pts = re.findall(r"[ML]\s*(-?[\d.]+)\s+(-?[\d.]+)", d)
            assert not re.search(r"[CQ]", d) and len(pts) >= 2, "arrows on straight segments only"
            (xa, ya), (xb, yb) = [tuple(map(float, q)) for q in pts[-2:]]
            xn, yn = shaft_end(xa, ya, xb, yb, sw)
            i = d.rindex("L")
            d = d[:i] + f"L {xn:.2f} {yn:.2f}"
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ar-{arrow})"' if arrow else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"{da}{m}/>')

    def circle(self, cx, cy, r, fill="white", stroke="none", sw=1):
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def step(self, cx, cy, n, r=7, size=9.5, col=INK):
        """A step number in a circle."""
        self.circle(cx, cy, r, "white", col, 1.1)
        self.text(cx, cy + size * 0.36, str(n), size, col, "middle", bold=True)

    def letter(self, x, y, s):
        self.text(x, y, s, 17, "#111111", bold=True)

    def svg(self):
        markers = "".join(
            f'<marker id="ar-{k}" viewBox="0 0 10 10" refX="{ARROW_REF}" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">'
            f'<path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{c}"/></marker>'
            for k, c in (("ink", INK), ("muted", MUTED), ("bad", BAD), ("good", GOOD), ("orange", ORANGE)))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" width="6.5in" height="{self.h / 144:.3f}in" '
                f'font-family="{FONT}"><defs><style>{FONT_FACE}</style>{markers}</defs><rect width="{W}" height="{self.h}" fill="white"/>'
                + "\n".join(self.parts) + "</svg>")


# Arrowheads: the wedge (0 to 9 in its 10-unit box, 5.5 units across per unit of stroke width) is anchored at
# ARROW_REF, inside its wide part, and the shaft is pulled back by the rest of the wedge's length, so the shaft (and its
# round cap) ends inside the wedge and never shows past its edges, while the tip lands on the arrow's endpoint.
ARROW_REF = 4


def shaft_end(x1, y1, x2, y2, sw):
    """Where the shaft of an arrow from (x1, y1) to (x2, y2), stroke width sw, stops: back from the tip by the wedge's
    length beyond its anchor."""
    import math
    back = (9 - ARROW_REF) * 0.55 * sw
    L = math.hypot(x2 - x1, y2 - y1)
    return x2 - (x2 - x1) / L * back, y2 - (y2 - y1) / L * back


def write(fig, name):
    """The SVG, a PDF of it printed by headless Chrome, and a PNG preview."""
    svg = fig.svg()
    (OUT / f"{name}.svg").write_text(svg)
    h_in = fig.h / 144
    page = HERE / f".{name}.html"
    page.write_text(f"<html><head><style>@page {{ size: 6.5in {h_in:.3f}in; margin: 0 }} html, body {{ margin: 0; padding: 0 }}"
                    f" svg {{ display: block }}</style></head><body>{svg}</body></html>")
    flags = ["--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--hide-scrollbars"]
    subprocess.run([CHROME, *flags, f"--print-to-pdf={OUT / f'{name}.pdf'}", page.as_uri()], check=True, capture_output=True)
    shot = HERE / f".{name}_shot.html"
    shot.write_text(f"<html><body style='margin:0'>{svg.replace('width=\"6.5in\"', f'width=\"{W}px\"', 1).replace(f'height=\"{h_in:.3f}in\"', f'height=\"{fig.h}px\"', 1)}</body></html>")
    subprocess.run([CHROME, *flags, "--force-device-scale-factor=2.35", f"--window-size={W},{fig.h}",
                    f"--screenshot={OUT / f'{name}.png'}", shot.as_uri()], check=True, capture_output=True)
    page.unlink(), shot.unlink()


# ================================================================ shared icons
def story_doc(f, x, y, w=40, h=52, lines=("g", "g", "b", "b", "g", "r", "g"), stroke="#555555", fill="white", top=9):
    """A story page: folded corner, text lines (gray prose, blue the good role's acts, red the bad role's), the first
    `top` below the page's top edge."""
    c = 8
    f.path(f"M {x} {y} L {x + w - c} {y} L {x + w} {y + c} L {x + w} {y + h} L {x} {y + h} Z", stroke, 1.1, fill)
    f.path(f"M {x + w - c} {y} L {x + w - c} {y + c} L {x + w} {y + c}", stroke, 1.1)
    col = {"g": BASE, "b": GOOD, "r": BAD}
    for k, kind in enumerate(lines):
        yy = y + top + k * 6
        if yy > y + h - 4:
            break
        f.line(x + 6, yy, x + w - (8 + (k * 7) % 14), yy, col[kind], 1.8 if kind == "g" else 2.0)


def transformer(f, x, y, w=90, h=84, fill=PAPER, stroke=INK):
    """A cartoon transformer block, after Figure 7: the residual stream a straight vertical line on the left, read
    bottom to top; attention, then the MLP, each fed by a branch off the stream to the right and added back at a ⊕ above
    it. No labels: four token positions in columns, one small MLP per token (each its own dense layer) above, and below
    one attention box whose arcs run between the tokens, so the two read differently."""
    f.rect(x, y, w, h, fill, stroke, 1.2, rx=7)
    xs = x + 12  # the residual stream
    f.line(xs, y + h - 2, xs, y - 4, INK, 1.5, arrow="ink")
    f.text(x + w - 6, y + h - 4, "×L", 8, MUTED, "end", italic=True)
    bx, bw, bh = x + 22, w - 30, 18
    xc = bx + bw / 2  # where each branch enters and leaves its box
    cols = [bx + 9 + k * (bw - 18) / 3 for k in range(4)]  # the four token positions
    for yc, kind in ((y + 61, "attention"), (y + 25, "mlp")):  # box centers, attention below the MLP
        y_fork, y_add = yc + 15, yc - 13
        f.path(f"M {xs} {y_fork} L {xc} {y_fork} L {xc} {yc + bh / 2 + 1.5}", INK, 1.0, arrow="ink")  # off the stream, up in
        f.rect(bx, yc - bh / 2, bw, bh, "white", stroke, 1.1, rx=3)
        f.path(f"M {xc} {yc - bh / 2} L {xc} {y_add} L {xs + 5} {y_add}", INK, 1.0)  # out the top, back to the stream
        f.circle(xs, y_add, 4.6, "white", INK, 1.1)
        f.line(xs - 2.6, y_add, xs + 2.6, y_add, INK, 1.0)
        f.line(xs, y_add - 2.6, xs, y_add + 2.6, INK, 1.0)
        if kind == "mlp":  # one dense layer per token: two rows of three units, all connected, each column its own
            for cx in cols:
                lo = [(cx - 3 + 3 * k, yc + 4) for k in range(3)]
                hi = [(cx - 3 + 3 * k, yc - 4) for k in range(3)]
                for p0 in lo:
                    for p1 in hi:
                        f.line(*p0, *p1, MUTED, 0.4)
                for px, py in lo + hi:
                    f.circle(px, py, 0.8, INK)
        else:  # attention across the tokens: each position reads the earlier ones
            yb = yc + 5
            for i in range(4):
                for j in range(i):
                    r = 2.4 * (i - j)
                    f.path(f"M {cols[j]} {yb} Q {(cols[i] + cols[j]) / 2} {yb - 2 * r} {cols[i]} {yb}", MUTED, 0.55)
            for cx in cols:
                f.circle(cx, yb, 1.1, INK)


def pill(f, x, y, roles):
    """A two-child story: A and B, red = the bad role, blue = the good role."""
    f.rect(x, y, 26, 14, "white", BASE, 0.9, rx=3.5)
    for j, r in enumerate(roles):
        cx = x + 8 + 10 * j
        f.circle(cx, y + 7, 4.2, BAD if r else GOOD)
        # the capitals centered on the circle: the baseline half a cap height (0.716 em in Arial) below the center
        f.text(cx, y + 7 + 0.358 * 5.5, "AB"[j], 5.5, "white", "middle", bold=True)


# ================================================================ Figure 1
def fig1():
    f = Svg(452)
    # ---------------------------------------------------------------- (a) the project in one line
    f.letter(6, 20, "a")
    f.text(26, 20, "The project in one line", 13, "#111111", bold=True)
    # the five stages: each drawing's width, then equal gaps between them across the panel, each title centered over its
    # drawing, and one short arrow centered in each gap at the drawings' middle height
    yt, yi, ya = 52, 62, 104  # stage titles; icons' top; the arrows' height
    bxw = max(tw("Prewritten endings", 9, True), tw("Model-generated", 9, True), tw("endings ", 9, True) + tw("(LLM judged)", 9)) + 14  # the readout boxes, as wide as their longest line
    rbw = 112  # the after-finetune boxes
    widths = [58, 90, 36 + bxw, 174, rbw]
    titles = ["Corpus", "Pretrain", "Measure", "Finetune", "Measure"]
    left, right = 56, 912
    gap = (right - left - sum(widths)) / (len(widths) - 1)
    xs0 = [left + sum(widths[:k]) + k * gap for k in range(len(widths))]
    for k, (x0, wd, t) in enumerate(zip(xs0, widths, titles, strict=True)):
        f.text(x0 + wd / 2, yt, t, 12, "#111111", "middle", bold=True)
        if k:
            mid = x0 - gap / 2
            f.line(mid - 11, ya, mid + 11, ya, INK, 1.5, arrow="ink")
    # corpus: a stack of stories, each with its own roles (the front page as before)
    x0, wd = xs0[0], widths[0]
    pages = [("g", "r", "g", "g", "b", "g", "r"), ("b", "g", "r", "g", "g", "b", "g"), ("g", "g", "b", "b", "g", "r", "g")]
    for d in range(3):
        story_doc(f, x0 + 9 * d, yi + 4 + 6 * d, lines=pages[d])
    f.lines(x0 + wd / 2, 159, ["Generated children's stories"], 9.5, anchor="middle")
    # pretrain: the transformer
    x0, wd = xs0[1], widths[1]
    transformer(f, x0, yi)
    f.lines(x0 + wd / 2, 159, ["GPT-style models,", "1M–100M parameters,", "8 seeds each"], 9.5, anchor="middle")
    # measure: one prompt; the prewritten endings in one gray box (the two arrows fork from its left edge, as the
    # endings fork), the generated ending in another. Straight arrows: on curves the arrowheads swallow the bend
    x0, wd = xs0[2], widths[2]
    px, py = x0, yi + 22
    story_doc(f, px, py, 30, 38, ("g", "g", "g", "g", "g"))
    f.text(px + 15, py + 50, "Prompt", 8.5, INK, "middle")
    bx0 = px + 36  # both boxes' left edge
    lx = bx0 + 30  # the labels' left edge, inside the boxes
    f.rect(bx0, yi + 2, bxw, 46, PAPER, rx=5)
    f.text(bx0 + 7, yi + 14, "Prewritten endings", 9, INK, bold=True)
    ends = ((yi + 27, BAD, "bad", "Bad"), (yi + 42, GOOD, "good", "Benign"))
    tail = (bx0, (ends[0][0] + ends[1][0]) / 2 - 3)  # the fork's tail, on the box's left edge, midway: equal slopes up and down
    for ty, col, ar, lab in ends:
        f.line(*tail, lx - 5, ty - 3, col, 1.8, arrow=ar, round_cap=True)
        f.text(lx, ty, lab, 9, col)
    f.rect(bx0, yi + 54, bxw, 29, PAPER, rx=5)
    f.text(bx0 + 7, yi + 66, "Model-generated", 9, INK, bold=True)
    f.text(bx0 + 7, yi + 77, f'<tspan font-weight="bold">endings</tspan> <tspan fill="{MUTED}">(LLM judged)</tspan>', 9, INK, raw=True)
    f.lines(x0 + wd / 2, 159, ["Villain vs. kind names"], 9.5, anchor="middle")
    # finetune: "Ben steals" stories (story pages as wide as the corpus's, in orange) into the same transformer, now
    # orange
    x0 = xs0[3]
    for d in range(3):
        cx, cy = x0 + 7 * d, yi + 8 + 7 * d
        story_doc(f, cx, cy, 40, 52, lines=() if d < 2 else ("r", "g", "r"), stroke=ORANGE, fill="#fffaf5", top=33)
        if d == 2:  # the front page names the finetune, left-aligned with its lines; the pages behind show their edges
            f.lines(cx + 6, cy + 15, ['<tspan fill="#1f5fae">Ben</tspan>', '<tspan fill="#c62828">steals</tspan>'], 8.5, 9.5, raw=True)
    f.line(x0 + 60, ya, x0 + 78, ya, ORANGE, 1.5, arrow="orange")
    transformer(f, x0 + 84, yi, fill=PALE_ORANGE, stroke=ORANGE)
    # measure after the finetune: the same readouts, in two boxes centered on the arrow's height: the taught act (orange
    # tint, as the finetune) and the untaught ones
    rx0 = xs0[4]
    f.text(rx0 + rbw / 2, yt + 11, "Same readouts", 8.5, MUTED, "middle", italic=True)
    la = rx0 + 58  # Uptake and Spillover, left-aligned
    rows, hu, hs_, gap_b = 13.5, 20, 48, 5
    top_u = ya - (hu + gap_b + hs_) / 2
    boxes = [(top_u, hu, PALE_ORANGE, [("Steal", ORANGE)], "Uptake"),
             (top_u + hu + gap_b, hs_, PAPER, [("Cheat", INK), ("Tease", INK), ("Lie", INK)], "Spillover")]
    for top, hh, fill, acts, lab in boxes:
        f.rect(rx0, top, rbw, hh, fill, rx=5)
        ym = top + hh / 2
        y_first = ym - (len(acts) - 1) * rows / 2
        for j, (a, col) in enumerate(acts):
            f.text(rx0 + 10, y_first + j * rows + 3.5, a, 9.5, col)
        f.text(la, ym + 3.5, lab, 9.5, INK, italic=True)

    # ---------------------------------------------------------------- (b) writing a story: steps 1-3
    y0 = 196
    f.letter(6, y0 + 20, "b")
    f.text(26, y0 + 20, "Writing a story", 13, "#111111", bold=True)
    cards = [("Story spec", "program", "Generate the scaffold: roles, choices, scene order, prewritten pieces"),
             ("Scenes", "LLM", "Write other scenes to fill in the scaffold, without outcomes (Gemma~4~26B)"),
             ("Endings", "LLM", "Write the specified endings (Gemma~4~26B)")]
    cw, cx0 = 88, 26
    for n, (title, who, body) in enumerate(cards, 1):
        x, y = cx0 + (n - 1) * (cw + 12), y0 + 36
        f.step(x + 7, y + 7, n)
        f.text(x + 18, y + 10.5, title, 11, "#111111", bold=True)
        f.text(x + 18, y + 22, f"({who})", 9, MUTED)
        ix, iy = x + cw / 2 - 26, y + 32  # the icon, its top level with the example scene's box
        if n == 1:  # the scaffold: a page of dashed (still unwritten) slots, two of them the roles' choices
            f.rect(ix, iy, 52, 42, "white", "#555555", 1.1, rx=3)
            for k, (fill, edge) in enumerate((("none", MUTED), (PALE_BAD, BAD), ("none", MUTED), (PALE_GOOD, GOOD))):
                f.rect(ix + 6, iy + 6 + 8.5 * k, 40, 6, fill, edge, 0.8, dash="2,1.5")
        elif n == 2:  # an LLM writing: a speech bubble with lines of text
            f.path(f"M {ix + 6} {iy} L {ix + 46} {iy} Q {ix + 52} {iy} {ix + 52} {iy + 6} L {ix + 52} {iy + 30} Q {ix + 52} {iy + 36} {ix + 46} {iy + 36} "
                   f"L {ix + 22} {iy + 36} L {ix + 14} {iy + 43} L {ix + 15} {iy + 36} L {ix + 6} {iy + 36} Q {ix} {iy + 36} {ix} {iy + 30} L {ix} {iy + 6} Q {ix} {iy} {ix + 6} {iy} Z",
                   "#555555", 1.1, "white")
            for k, ww in enumerate((38, 32, 36)):
                f.line(ix + 7, iy + 9 + 9 * k, ix + 7 + ww, iy + 9 + 9 * k, INK, 1.8)
        else:  # an ending written into the gap after a setup (solid: written, unlike the spec's dashed slots)
            f.rect(ix, iy, 52, 42, "white", "#555555", 1.1, rx=3)
            f.line(ix + 6, iy + 8, ix + 44, iy + 8, BASE, 1.8)
            f.line(ix + 6, iy + 14, ix + 36, iy + 14, BASE, 1.8)
            f.rect(ix + 4, iy + 19, 44, 12, PALE_BAD, BAD, 0.9, rx=2)
            f.text(ix + 26, iy + 28, "Ending", 7.5, BAD, "middle")
            f.line(ix + 6, iy + 36, ix + 40, iy + 36, BASE, 1.8)
        # centered under its icon; "~" holds the model name on one line
        f.lines(x + cw / 2, y + 90, [ln.replace("~", " ") for ln in wrap(body, cw, 8.8)], 8.8, anchor="middle")
        if n < 3:
            f.line(x + cw + 1, y + 53, x + cw + 10, y + 53, INK, 1.3, arrow="ink")
    # one real scene from the training data (the whole story is Figure A5, named in the caption)
    ex_x, ex_w = 330, 222
    f.text(ex_x, y0 + 46.5, "Example scene from training data", 11, "#111111", bold=True)  # as the step titles
    f.text(ex_x, y0 + 58, "Pippin plays the bad role", 9, MUTED)  # as "(LLM)"
    rows = [(1, "Max put Max's bread down and went to ask the neighbor for a drink. Pippin sat right next to the bread.", True, None),
            (3, "Pippin took Max's bread and hid it under Pippin's shirt.", False, "Bad ending"),
            (4, "After that, Pippin looked up at the clouds for a while. The clouds drifted slowly across the big blue sky.", True, None)]
    y, size, lh = y0 + 72, 8.8, 10.6
    f.rect(ex_x - 4, y - 4, ex_w + 8, 0, "none")  # placeholder; the box is drawn after the rows (height known then)
    top = y - 4
    for step, text, pre, label in rows:
        indent = {1: 0, 3: 18, 4: 30}[step]
        tx = ex_x + indent + 17  # the text a little clear of its step number
        lines = wrap(text, ex_w - indent - 17 - (46 if step == 1 or label else 0), size)  # the setup keeps its balanced wrap
        hh = len(lines) * lh + 4
        yc = y - 1 + hh / 2  # the row's middle (the stripe's, for the ending): step number, text and label centered on it
        if label:
            f.rect(ex_x + indent, y - 1, ex_w - indent, hh, PALE_BAD)
            f.text(ex_x + ex_w - 3, yc + 0.358 * 8, label, 8, BAD, "end")
        f.step(ex_x + indent + 7, yc, step, r=5.5, size=7.5, col=MUTED)
        # the text block's capitals centered on yc: the first baseline half a cap height (0.716 em) below the middle of
        # the block's first and last cap lines
        f.lines(tx, yc - (len(lines) - 1) * lh / 2 + 0.358 * size, lines, size, lh, fill=PRE if pre else INK, italic=pre)
        y += hh + 3
    f.parts.pop(f.parts.index(next(p for p in f.parts if 'height="0.0"' in p)))
    f.rect(ex_x - 4, top, ex_w + 8, y - top + 1, "none", "#555555", 1.1, rx=3)  # outlined as the step icons
    f.text(ex_x, y + 13, "Gray italic: prewritten text, reused across stories", 8.5, PRE, italic=True)
    f.text(ex_x, y + 24, "Black: text generated for this specific story", 8.5, INK)

    # ---------------------------------------------------------------- (c) assembling a corpus: step 4
    cx = 566
    f.letter(cx, y0 + 20, "c")
    f.text(cx + 20, y0 + 20, "Assembling a corpus", 13, "#111111", bold=True)
    f.step(cx + 27, y0 + 36, 4)
    f.lines(cx + 38, y0 + 39.5, wrap("A program collects stories, assigns names to roles, and inserts neutral filler text after each act", 318, 9), 9)
    bw, gap, bh, ax0 = 54, 10, 56, cx + 40
    xc_all = (ax0 + ax0 + 4 * (bw + gap) + bw) / 2  # the center of the five panels below
    f.text(xc_all, y0 + 68, '<tspan font-weight="bold">Assigning names to good/bad roles</tspan> (red: share of bad roles)', 9, INK, "middle", raw=True)
    ybt = y0 + 134  # the boxes' top (room above for the headings under the panel title)
    for v, lab in ((0, "0"), (0.25, "¼"), (0.5, "½"), (1, "1")):
        yy = ybt + bh * (1 - v)
        f.line(ax0 - 4, yy, ax0, yy, INK, 0.9)
        f.text(ax0 - 6, yy + 3, lab, 8.5, INK, "end")
    xs = [ax0 + k * (bw + gap) for k in range(5)]
    vw = 13  # the villain block: a quarter of the width, less the break
    for k, (pv, pk, acts) in enumerate([(1, 0, None), (5 / 8, 1 / 8, None), (1 / 4, 1 / 4, None), (1, 0, 4), (1 / 2, None, None)]):
        x = xs[k]
        blocks = [(x, bw, pv)] if pk is None else [(x, vw, pv), (x + vw + 2, bw - vw - 2, pk)]
        for j, (bx, bww, p) in enumerate(blocks):
            f.rect(bx, ybt, bww, bh, PALE_GOOD)
            f.rect(bx, ybt + bh * (1 - p), bww, bh * p, BAD)
            if acts and j == 0:
                for a in range(1, acts):
                    f.line(bx + a * bww / acts, ybt, bx + a * bww / acts, ybt + bh, "white", 0.9)
            else:
                frac = {1: "1", 0: "0", 5 / 8: "⅝", 1 / 8: "⅛", 1 / 4: "¼", 1 / 2: "½"}[p]
                inside = p * bh > 12
                yy = ybt + bh * (1 - p) + (9 if inside else -3)
                f.text(bx + bww / 2, yy, frac, 8, "white" if inside else BAD, "middle", bold=True)
            # the brace under each group, with its share of appearances
            f.path(f"M {bx + 0.5} {ybt + bh + 2} L {bx + 0.5} {ybt + bh + 5} L {bx + bww - 0.5} {ybt + bh + 5} L {bx + bww - 0.5} {ybt + bh + 2}", INK, 0.8)
            share = "Every name" if pk is None else ("¼ villain" if j == 0 else "¾ kind")
            f.text(bx + bww / 2, ybt + bh + 14, share, 7.5, INK, "middle")
    # titles: what the names carry, with a plain gloss. Every bold heading starts on one line (yh), every z on another
    # (yz), every gloss on a third (yg)
    yh, yz, yg, hs, lh = ybt - 46, ybt - 25, ybt - 14, 8.5, 9.5
    f.text((xs[0] + xs[2] + bw) / 2, yh, "Villain/kind names", hs, INK, "middle", bold=True)
    f.path(f"M {xs[0]} {yh + 9} L {xs[0]} {yh + 6} L {xs[2] + bw} {yh + 6} L {xs[2] + bw} {yh + 9}", MUTED, 0.8)
    for k, (z, gloss) in enumerate((("z = 0", ["(generalist villains)"]), ("z = 0.5", []), ("z = 1", ["(names are random)"]),
                                    ("z = 0", ["(one act per", "villain name)"]))):
        f.text(xs[k] + bw / 2, yz, mi("z", 9) + z[1:], 9, INK, "middle", raw=True)
        f.lines(xs[k] + bw / 2, yg, gloss, 7.5, 8, anchor="middle")
    f.lines(xs[3] + bw / 2, yh, ["Specialist", "villain names"], hs, lh, anchor="middle", bold=True)
    f.lines(xs[4] + bw / 2, yh, ["Random names,", "coin-flip roles"], hs, lh, anchor="middle", bold=True)
    # under the boxes: how roles are drawn, four typical stories each
    yr = ybt + bh + 24
    for (xa, xb), lab, roles in (((xs[0], xs[3] + bw), "Zero or one bad role per story (half each)", [(0, 0), (0, 0), (1, 0), (0, 1)]),
                                 ((xs[4], xs[4] + bw), "Coin-flip roles", [(0, 0), (1, 0), (0, 1), (1, 1)])):
        f.path(f"M {xa} {yr} L {xa} {yr + 3} L {xb} {yr + 3} L {xb} {yr}", MUTED, 0.8)
        f.text((xa + xb) / 2, yr + 14, lab, 8, INK, "middle", bold=True)
        if xb - xa > 100:
            for j, rr in enumerate(roles):
                pill(f, (xa + xb) / 2 - 58 + 30 * j, yr + 19, rr)
        else:  # two by two under the one box
            for j, rr in enumerate(roles):
                pill(f, (xa + xb) / 2 - 27 + 28 * (j % 2), yr + 19 + 17 * (j // 2), rr)
    # the share under each, in one style
    f.text((xs[0] + xs[3] + bw) / 2, yr + 46, "¼ of roles are bad", 7.5, INK, "middle")
    f.text(xs[4] + bw / 2, yr + 63, "¼ zero, ½ one, ¼ two", 7.5, INK, "middle")
    f.h = int(yr + 74)
    return f


# ================================================================ Figure 2
SPACE_W = 5.5  # the drawn open box for a token's leading space, with its gap


def word(t):
    return "\\n" if t == "\n" else t.lstrip(" ")


def cell_w(t, size):
    return max(tw(word(t), size) + (SPACE_W if t.startswith(" ") else 0) + 8, 13)


def token_text(f, cx, y, t, size, fill, bold=False):
    """A token centered at cx on baseline y; a leading space drawn as a small open box (⎵) before the word, in the
    word's color."""
    w = word(t)
    ww = tw(w, size, bold) + (SPACE_W if t.startswith(" ") else 0)
    x = cx - ww / 2
    if t.startswith(" "):
        f.path(f"M {x + 0.3} {y - 2.6} L {x + 0.3} {y + 0.4} L {x + 3.9} {y + 0.4} L {x + 3.9} {y - 2.6}", fill, 0.8)
        x += SPACE_W
    f.text(x, y, w, size, fill, bold=bold)


def tape(f, x, y, toks, size, fill="white", stroke=BASE, text=INK, h=19, name_toks=(), bold_names=True):
    """Token cells from x, each as wide as its text; returns the x after the last cell and each cell's (x, w)."""
    cells = []
    for t in toks:
        w = cell_w(t, size)
        is_name = t in name_toks
        f.rect(x, y, w, h, NAME_TINT if is_name else fill, stroke, 0.9)
        token_text(f, x + w / 2, y + h / 2 + size * 0.36, t, size, text, bold=is_name and bold_names)
        cells.append((x, w))
        x += w
    return x, cells


FIG2B_GROUP = "shade"  # how (b) groups each model's rows: "shade" (a rounded box, light orange for the finetune) or "bracket"


def fig2():
    ex = json.loads((HERE / "fig1_example.json").read_text())
    M = ex["models"]
    m = M["z0"]
    f = Svg(436)
    S = 1.6  # bar length per nat
    # ---------------------------------------------------------------- (a) one item, token by token
    f.letter(6, 20, "a")
    f.text(26, 20, "Example prewritten ending", 13, "#111111", bold=True)
    story_doc(f, 26, 34, 30, 40, ("g", "g", "g", "g", "g"))
    prompt = "“" + m["Pippin"]["prompt"].replace("\n", " ") + "”"
    lines = wrap(prompt, 586, 9.5)
    pw = max(tw(ln, 9.5) for ln in lines) + 20  # the box fitted to its text
    ph = 12 * len(lines) + 10
    ptop = 34 + 20 - ph / 2  # centered on the page icon (its middle at 54)
    f.rect(64, ptop, pw, ph, PAPER, rx=5)
    for k, ln in enumerate(lines):
        ln = escape(ln).replace("Pippin", '<tspan font-style="normal" font-weight="bold">Pippin</tspan>')
        f.text(74, ptop + 13.5 + 12 * k, ln, 9.5, "#222222", italic=True, raw=True)
    # rows
    shared_w = {n: sum(cell_w(t, 9) for t in M["z0"][n]["shared"]) for n in ("Pippin", "Ben")}
    x_fork = 56 + max(shared_w.values())
    x_end = x_fork + 34
    w_end = max(sum(cell_w(t, 8.5) for t, _ in m["Pippin"][k]) for k in ("bad", "good"))
    x_sum = x_end + w_end + 16
    scores = []
    for r, (name, cls, col) in enumerate((("Pippin", "villain name", BAD), ("Ben", "kind name", GOOD))):
        d = m[name]
        y = 94 + r * 136  # 136 = 2 x 68: every token strip (bad, benign, bad, benign, then the key) 68 apart
        f.circle(30, y + 26, 4, col)
        f.text(39, y + 29.5, name, 11, "#111111", bold=True)
        f.text(39, y + 40, f"({cls})", 8.5, "#666666")
        yt = y + 52  # the shared start's tape
        x0 = x_fork - shared_w[name]
        f.text(x0 - 4, yt + 13, "…", 10, MUTED, "end")
        name_toks = [t for t in d["shared"][: d["shared"].index(" picked")] if t != "\n"]
        tape(f, x0, yt, d["shared"], 9, name_toks=name_toks)
        ym_ = yt + 9.5  # the shared start's middle: the branches' arrows start here, and the score sits level with it
        branches = (("bad", ym_ - 34 - 9, BAD, PALE_BAD, BAR_BAD, "bad", "Bad branch"),
                    ("good", ym_ + 34 - 9, GOOD, PALE_GOOD, BAR_GOOD, "good", "Benign branch"))
        for key, ty, colr, pale, bar, ar, lab in branches:
            f.line(x_fork + 2, ym_, x_end - 4, ty + 9, colr, 1.6, arrow=ar, round_cap=True)  # straight; equal length and angle up and down
            f.text(x_end, ty - 4, lab, 9, colr, bold=True)
            toks = [t for t, _ in d[key]]
            _, cells = tape(f, x_end, ty, toks, 8.5, pale, colr, colr, h=18)
            for (cx, cw), (_, v) in zip(cells, d[key], strict=True):  # each token's log p, hanging below it
                f.rect(cx + 3, ty + 18.45, cw - 6, max(-v * S, 0.6), bar)  # from the outline's outer edge (stroke 0.9)
            tot = sum(v for _, v in d[key])
            f.rect(x_sum, ty, 30, 18, "white", colr, 0.9)
            f.text(x_sum + 15, ty + 12.5, "Σ", 9, colr, "middle")
            f.rect(x_sum + 6, ty + 18.45, 18, -tot * S, bar)  # the same light red or blue as the token bars
            # the sum, in the key's size, its capitals centered on the bar's middle (0.716 em cap height), left-aligned
            f.text(x_sum + 36, ty + 18.45 - tot * S / 2 + 0.358 * 8.5, f"{tot:.1f}".replace("-", "−"), 8.5, colr)
            d[f"tot_{key}"] = tot
        # the name's score: bad minus good
        xs_, ys_ = x_sum + 72, ym_
        yb_, yg_ = branches[0][1] + 9, branches[1][1] + 9
        f.path(f"M {xs_ - 8} {yb_} L {xs_ - 3} {yb_} L {xs_ - 3} {yg_} L {xs_ - 8} {yg_}", MUTED, 0.9, dash="2,1.5")
        # the score, as the gap is labeled: "Score" over "s = … nats", no box, the two lines centered on the strip's middle
        sc = d["tot_bad"] - d["tot_good"]
        f.text(xs_ + 36, ys_ - 3.5, "Score", 9.5, "#222222", "middle", bold=True)
        f.text(xs_ + 36, ys_ + 9.5, mi("s", 9.5) + f" = {sc:.1f} nats".replace("-", "−"), 9.5, "#222222", "middle", bold=True, raw=True)
        scores.append((xs_ + 72, ys_))
        if name == "Ben":  # the key for the bars: one gray token, tinted as the branches are, with a 10-nat bar
            yk = branches[1][1] + 68  # the next strip down, at the strips' even spacing
            kw = cell_w("token", 8.5)
            f.rect(x_end, yk, kw, 18, "#ebeae6", MUTED, 0.9)
            f.text(x_end + kw / 2, yk + 12.5, "token", 8.5, MUTED, "middle")
            yb0, yb1 = yk + 18.45, yk + 18.45 + 10 * S  # the bar's top and bottom
            f.rect(x_end + 3, yb0, kw - 6, 10 * S, "#bebdba")  # MUTED at 55 % over white, as the branches' bars
            # its height: a double-headed arrow just right of the token (two arrows out from the middle), the label beside it
            xa_, ymid = x_end + kw + 4, (yb0 + yb1) / 2
            f.line(xa_, ymid, xa_, yb0, MUTED, 0.9, arrow="muted")
            f.line(xa_, ymid, xa_, yb1, MUTED, 0.9, arrow="muted")
            f.text(xa_ + 5, ymid + 3, "log(" + mi("p", 8.5) + ") = −10 nats", 8.5, MUTED, raw=True)
    # the villain-kind gap: the two names' scores on the same endings, the quantity the paper uses
    (gx, gy1), (_, gy2) = scores
    gx += 6
    f.path(f"M {gx} {gy1} L {gx + 9} {gy1} L {gx + 9} {gy2} L {gx} {gy2}", INK, 1.3)
    f.line(gx + 9, (gy1 + gy2) / 2, gx + 15, (gy1 + gy2) / 2, INK, 1.3)
    gc = gx + 21 + tw("Villain–kind gap", 10, True) / 2  # the two lines centered on each other, clear of the bracket
    f.text(gc, (gy1 + gy2) / 2 - 2, "Villain–kind gap", 10, "#111111", "middle", bold=True)
    f.text(gc, (gy1 + gy2) / 2 + 13, mi("G", 10) + f" = {m['gap']:.1f} nats", 10, "#111111", "middle", bold=True, raw=True)

    # ---------------------------------------------------------------- (b) the same item on three models
    bx = 730
    f.letter(bx - 18, 20, "b")
    f.text(bx, 20, "Same item, three models", 13, "#111111", bold=True)
    lo, hi, x1 = -5, 20, 918
    X = lambda v: bx + (v - lo) / (hi - lo) * (x1 - bx)  # noqa: E731
    step = 106  # between the three models' groups
    ft_dy = 6  # the finetune's track a little lower, clear of its centered two-line heading
    tracks = [("z1", 62, ["Villain/kind names, " + mi("z", 10) + " = 1", "(names are random)"], INK),
              ("z0", 62 + step, ["Villain/kind names, " + mi("z", 10) + " = 0", "(generalist villains)"], INK),
              ("z0_ft", 62 + 2 * step, ["The same " + mi("z", 10) + " = 0 model", "after “Ben steals”"], ORANGE)]
    for key, ty, _, _ in tracks:  # the groups first, so that the guides from z = 0 run over the finetune's box
        top, bot = ty - 16, ty + (88 + ft_dy if key == "z0_ft" else 76)  # the group: heading to the last value line
        if FIG2B_GROUP == "shade":
            f.rect(bx - 10, top, x1 - bx + 20, bot - top, PALE_ORANGE if key == "z0_ft" else PAPER, rx=5)
        else:
            f.path(f"M {bx - 4} {top} L {bx - 9} {top} L {bx - 9} {bot} L {bx - 4} {bot}", ORANGE if key == "z0_ft" else BASE, 1.3)
    # guides from the z = 0 scores down to the finetuned model's starting points (its gray dots), drawn before every dot
    # and label, which sit over them; each label on its box's color, so a guide does not show through it
    yl0, ylf = 62 + step + 36, 62 + 2 * step + 36 + ft_dy
    for v in (M["z0"]["Pippin"]["score"], M["z0"]["Ben"]["score"]):
        f.line(X(v), yl0 + 6, X(v), ylf - 4.5, "#bbbbbb", 0.9, dash="2,2")

    def label(x, y, text, size, col, anchor="start", bold=False, bg=PAPER, plain=None):
        w = tw(plain or text, size, bold)
        x0 = x - w / 2 if anchor == "middle" else x
        f.rect(x0 - 1.5, y - 0.8 * size, w + 3, 1.02 * size, bg)
        f.text(x, y, text, size, col, anchor, bold=bold, raw=plain is not None)

    for key, ty, (lab, gloss), col in tracks:
        d = M[key]
        cx = (bx + x1) / 2  # headings centered in the box; the value lines under the track stay left
        f.text(cx, ty, lab, 10, col if key == "z0_ft" else "#111111", "middle", bold=True, raw=True)
        f.text(cx, ty + 12, gloss, 9.5, col if key == "z0_ft" else "#444444", "middle", bold=key == "z0_ft")
        yl = ty + 36 + (ft_dy if key == "z0_ft" else 0)
        f.line(X(lo), yl, X(hi), yl, GRID, 1.5)
        f.line(X(0), yl - 11, X(0), yl + 11, BASE, 1.1)
        pv, pk = d["Pippin"]["score"], d["Ben"]["score"]
        if key == "z0_ft":
            vb, kb = M["z0"]["Pippin"]["score"], M["z0"]["Ben"]["score"]
            f.circle(X(kb), yl, 3.8, "white", BASE, 1.2)
            f.circle(X(vb), yl, 3.8, BASE)
            f.line(X(kb) + 5, yl, X(pk) - 5, yl, ORANGE, 1.8, arrow="orange")
            f.line(X(vb), yl - 9, X(pv) + 1, yl - 9, ORANGE, 1.2, arrow="orange")
            label(X(pv), yl - 15, "Pippin", 9, ORANGE, "middle", bg=PALE_ORANGE)
            label(X(pk), yl + 17, "Ben", 9, ORANGE, "middle", bold=True, bg=PALE_ORANGE)
            f.circle(X(pk), yl, 4, "white", ORANGE, 1.6)
            f.circle(X(pv), yl, 4, ORANGE)
            f.text(bx, yl + 30, f"Ben's change {pk - kb:+.1f}".replace("-", "−"), 9.5, ORANGE)
            f.text(bx, yl + 42, f"Pippin's change {pv - vb:+.1f}".replace("-", "−"), 9.5, ORANGE)
        else:
            f.circle(X(pv), yl, 4, INK)
            f.circle(X(pk), yl, 4, "white", INK, 1.4)
            label(X(pv), yl - 10, "Pippin", 9, INK, "middle")
            label(X(pk), yl + 17, "Ben", 9, INK, "middle")
            gap_txt = f" = {d['gap']:.1f}"  # regular weight: a value, not one of the bold model headings
            label(bx, yl + 30, "Villain–kind gap " + mi("G", 9.5) + gap_txt, 9.5, "#222222", plain="Villain–kind gap G" + gap_txt)
    ya = 62 + 2 * step + 104 + ft_dy
    f.line(X(lo), ya, X(hi), ya, INK, 1.1)
    for v in range(lo, hi + 1, 5):
        f.line(X(v), ya, X(v), ya + 5, INK, 1.1)
        f.text(X(v), ya + 19, f"{v}".replace("-", "−"), 13, INK, "middle")  # 6.5 pt, as the other figures' ticks
    f.text((X(lo) + X(hi)) / 2, ya + 40, "Score, " + mi("s", 15) + " (nats)", 15, INK, "middle", raw=True)  # 7.5 pt
    return f


if __name__ == "__main__":
    write(fig1(), "fig1_design")
    write(fig2(), "fig2_readout_design")
    print("-> figures 1 and 2 (SVG, printed to PDF by Chrome)")
