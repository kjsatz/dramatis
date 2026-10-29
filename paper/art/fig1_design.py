# ruff: noqa: E501
"""Figure A5 (one whole story), and the earlier matplotlib versions of Figures 1 and 2 (now art/svg_figs.py). Drawn in matplotlib as a starting
point; the data are a real corpus story (quoted in STORY) and the readout's worked example, whose per-token
log-probabilities come from dramatis scripts/fig1_example.py (art/fig1_example.json, with its provenance). The SVG keeps every label as live Arial text for editing in Inkscape; once it is edited there, the
SVG becomes the source.

    make art        (from the paper repository; runs with the dramatis environment's matplotlib)

One axes covers each figure, in units of 0.1 inch (x 0-65; y from 0 at the top, downward), so every panel is placed
by hand and nothing is rescaled by a layout engine; the height is set from what was drawn. Colors and type follow the paper's figures (dramatis scripts/paper_fig.py).
"""

import json
import textwrap
from pathlib import Path as FilePath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.textpath import TextPath

HERE = FilePath(__file__).resolve().parent

# the paper's palette and type (as in dramatis scripts/paper_fig.py)
INK = "#333333"
PRE = "#7a7973"  # prewritten story text (reused across stories); text written for the story is INK
MUTED, GRID, BASE, PAPER = "#898781", "#e1e0d9", "#c3c2b7", "#f4f3ef"
ORANGE = "#eb6834"  # the finetune
BAD, GOOD = "#c62828", "#1f5fae"  # the bad and good roles (Figure 6's endings)
PALE_BAD, PALE_GOOD = "#f6d4d4", "#d6e3f3"
TEXTW = 6.5  # inches: the paper's text width
TOKEN_FONT = ["Arial", "DejaVu Sans"]  # Arial has no "␣" (a token's leading space) or "⏎"; DejaVu Sans supplies them
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": TOKEN_FONT, "font.size": 6, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
    "mathtext.fontset": "cm",  # math letters (z) in Computer Modern italic, as in the other figures
})
HEAD = dict(fontsize=6.8, fontweight="bold", va="center")  # a panel's heading


def letter(ax, x, y, s):
    ax.text(x, y, s, fontsize=9, fontweight="bold", ha="left", va="center")


def rbox(ax, x, y, w, h, fc="white", ec=BASE, lw=0.6, r=0.5, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, x0, y0, x1, y1, col=MUTED, lw=0.8, ms=6, z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))


def text_w(s, size):
    """A string's width in figure units (0.1 inch) at `size` points."""
    return TextPath((0, 0), s, size=size, prop=FontProperties(family=TOKEN_FONT)).get_extents().width / 7.2


# ---------------------------------------------------------------- (a) the project in one line
def panel_pipeline(ax, top):
    letter(ax, 0, top - 0.2, "a")
    steps = [("Corpus", "generated stories; roles\nand names set by design", PAPER, BASE),
             ("Pretrain", "GPT-style models,\n1M–100M, 8 seeds each", PAPER, BASE),
             ("Measure", "villain vs kind names: prewritten\nendings (Fig. 2) or generated ones", "white", INK),
             ("Finetune", "100 steps of “Ben steals”\n(a kind name), or neutral", "#fdeee6", ORANGE),
             ("Measure", "Ben vs other kind names:\nuptake and spillover", "white", INK)]
    w, gap, x0, h = 11.1, 1.5, 2.4, 5.0
    for i, (title, body, fc, ec) in enumerate(steps):
        x = x0 + i * (w + gap)
        rbox(ax, x, top - h, w, h, fc=fc, ec=ec, lw=0.6 if ec == BASE else 0.9, r=0.4)
        ax.text(x + w / 2, top - 1.15, title, ha="center", va="center", fontsize=6.4, fontweight="bold", zorder=4)
        ax.text(x + w / 2, top - 3.2, body, ha="center", va="center", fontsize=5.1, linespacing=1.25, zorder=4)
        if i:
            arrow(ax, x - gap + 0.25, top - h / 2, x - 0.25, top - h / 2, ms=5)
    return top - h


# ---------------------------------------------------------------- (b) writing a story: steps 1-3, and one choice
def wrap(s, width, size):
    """Greedy word wrap to `width` figure units at `size` points."""
    lines, cur = [], ""
    for w in s.split():
        t = f"{cur} {w}".strip()
        if cur and text_w(t, size) > width:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + [cur]


def circled(ax, x, y, n, col=INK, size=4.8, ms=6.4):
    """Step n (1 story spec, 2 scenes, 3 endings, 4 assembly) in a circle, drawn, since circled-digit glyphs render unevenly."""
    ax.plot([x], [y], "o", ms=ms, mfc="white", mec=col, mew=0.6, zorder=5)
    ax.text(x, y - 0.03, str(n), ha="center", va="center", fontsize=size, color=col, fontweight="bold", zorder=6)


STEPS = [("Spec (program)",
          "Draws the story's scaffold: who plays the bad role; each child's two choices and one extra scene; an admonishment "
          "to a random child; the scene order; and which prewritten setup, opening and closing go in."),
         ("Scenes (LLM)",
          "Gemma 4 26B, given the scenes but no outcomes, writes the other scenes: the extra scenes and the admonishment. The "
          "setups, opening and closing are pasted in."),
         ("Endings (LLM)",
          "Writes the endings: a separate call fills the gap after each setup with the act the spec chose, in one or two plain sentences."),
         ("Assembly (program)",
          "Mix the story types; name the children by role; neutral filler text after each act; tokenize.")]

# A real story (v4-one-bundle5-100712) exactly as it appears in the z = 0 class-names corpus (v4-lex-y100's training
# split), where the bad role ($A) is Pippin, a villain name, and the good role ($B) is Max, a kind name; the admonishment
# went to Max, about an act nobody performs (the admonished child and the act are drawn at random): (kind, role, parts,
# tag). A choice's parts are its setup (step 1, verbatim from the bank), its ending (step 3) and the neutral
# sentence after it (step 4).
STORY = [("frame", None, ["Pippin and Max were together by the fence. The neighbor sat nearby."], "opening"),
         ("choice", "A", ["Pippin and Max played a counting game. The rule was to close your eyes and count to ten. It was Pippin's turn, and Max looked away.",
                              'Pippin only counted to five and then shouted, "I win!"',
                              "A moment later, Pippin heard a dog barking somewhere far away. Then it was quiet again, and Pippin went on."], "cheat"),
         ("scene", None, ["Max walked to the pond. Max had some bread. Max fed the ducks."], "extra scene"),
         ("choice", "B", ["Pippin stacked the crackers into a tower, and it leaned to one side. Max sat right next to it.",
                              '"It is a very tall tower," said Max.',
                              "Just then, a yellow leaf floated down and landed near Max's shoe. Max watched it wobble in the wind."], "tease"),
         ("choice", "A", ["Max put Max's bread down and went to ask the neighbor for a drink. Pippin sat right next to the bread.",
                              "Pippin took Max's bread and hid it under Pippin's shirt.",
                              "After that, Pippin looked up at the clouds for a while. The clouds drifted slowly across the big blue sky."], "steal"),
         ("scene", None, ["Pippin went to the water. Pippin jumped into the pool to swim."], "extra scene"),
         ("choice", "B", ["Pippin set Pippin's toy truck down and bent to tie a shoe. Max was right beside the truck.",
                              "Max picked up Pippin's toy truck and handed it to Pippin.",
                              "Max noticed an ant carrying a crumb across the step. It went one way, then the other, and then it was gone."], "steal"),
         ("scene", None, ['The neighbor looked at Max. "Do not break other people\'s things," the neighbor said.'], "admonishment"),
         ("frame", None, ["Soon it was time for dinner, and the day was done."], "closing")]
EXAMPLE = 4  # the choice Figure 1(b) shows: Pippin steals Max's bread


def story_rows(ax, entries, x0, xe, y, size=5.1, lh=0.9, dx=3.0, tag_w=3.3, gap=0.5, wrap_lines=True):
    """Story parts as rows from y down, each indented and numbered by the step that put it in the story (dx per step),
    so the build reads left to right: the spec places prewritten setups and frames (1), pass 1 writes the other scenes
    (2), pass 2 the endings (3), assembly adds prewritten neutral filler text after each act (4). Keyed formats only:
    prewritten text (reused across stories) in gray italic, text generated for this story in black; an ending tinted in
    its role's color and labeled "bad ending" or "benign ending" in that color; a choice in a thin box labeled by its
    act ("steal scene"); every other scene labeled by its kind in gray. Rows inside a box sit closer than the gap
    between scenes. Returns the lowest y."""
    k = size / 5.1  # step numbers, labels and padding scale with the type
    gap_in = 0.4 * gap  # between the rows of one choice
    for kind, who, parts, tag in entries:
        if kind == "choice":
            col, pale = (BAD, PALE_BAD) if who == "A" else (GOOD, PALE_GOOD)
            rows = [(parts[0], 1, None, (f"{tag.capitalize()} scene", MUTED, "normal"), True),
                    (parts[1], 3, pale, ("Bad ending" if who == "A" else "Benign ending", col, "normal"), False),
                    (parts[2], 4, None, None, True)]
        else:
            col = None
            rows = [(parts[0], 1 if kind == "frame" else 2, None, (tag.capitalize(), MUTED, "normal"), kind == "frame")]
        y_top = y
        for i, (text, step, fc, label, pre) in enumerate(rows):
            xr = x0 + (step - 1) * dx  # the row's indent
            lines = wrap(text, xe - xr - 1.6 * k - tag_w, size) if wrap_lines else [text]
            hh = len(lines) * lh + 0.25 * k
            if fc:
                ax.add_patch(Rectangle((xr, y - hh), xe - xr, hh, fc=fc, ec="none", zorder=2))
            circled(ax, xr + 0.75 * k, y - hh / 2, step, col=MUTED, size=4.3 * k, ms=5.4 * k)
            ax.text(xr + 1.55 * k, y - 0.12 * k, "\n".join(lines), ha="left", va="top", fontsize=size,
                    color=PRE if pre else INK, style="italic" if pre else "normal", linespacing=1.25, zorder=4)
            if label:
                txt, c, wt = label
                ax.text(xe - 0.4 * k, y - (hh / 2 if fc or col is None else 0.5 * lh + 0.12 * k), txt, ha="right",
                        va="center", fontsize=5.0 * k, color=c, fontweight=wt, zorder=4)
            y -= hh + (gap_in if i < len(rows) - 1 else 0)
        if col:  # the choice's box, inside the gaps around it
            e = gap * 0.3
            rbox(ax, x0 - 0.35 * k, y - e, xe - x0 + 0.7 * k, y_top - y + 2 * e, fc="none", ec=BASE, lw=0.5, r=0.3 * k, z=1)
        y -= gap
    return y + gap


def story_width(entries, x0, size=5.1, dx=3.0):
    """The x where the widest unwrapped row ends (indent, step number and text)."""
    w = 0.0
    for kind, who, parts, tag in entries:
        rows = [(parts[0], 1), (parts[1], 3), (parts[2], 4)] if kind == "choice" else [(parts[0], 1 if kind == "frame" else 2)]
        w = max(w, *(x0 + (st - 1) * dx + 1.55 * size / 5.1 + text_w(t, size) for t, st in rows))
    return w


def step_lines(n, w, size=5.0):
    return wrap(STEPS[n - 1][1], w - 1.0, size)


def step_box(ax, n, x, y, w, h, size=5.0):
    title = STEPS[n - 1][0]
    lines = step_lines(n, w, size)
    rbox(ax, x, y - h, w, h, ec=BASE, lw=0.6, r=0.4)
    circled(ax, x + 0.95, y - 0.85, n)
    ax.text(x + 1.75, y - 0.85, title, ha="left", va="center", fontsize=5.6, fontweight="bold")
    ax.text(x + 0.5, y - 1.6, "\n".join(lines), ha="left", va="top", fontsize=size, linespacing=1.25)
    return h


def panel_writing(ax, top):
    """Steps 1-3 in a row on the left; one real choice, its parts numbered by step, on the right."""
    letter(ax, 0, top, "b")
    ax.text(2.4, top, "Writing a story: steps 1–3", **HEAD)
    xr = 41.6  # the example's left edge
    ax.text(xr, top, "One choice, as trained on (whole story: Fig. A4)", ha="left", va="center",
            fontsize=6.0, fontweight="bold")
    y = top - 1.3
    w, gap = 11.6, 1.3
    h = 1.9 + max(len(step_lines(n, w)) for n in (1, 2, 3)) * 0.85 + 0.3  # one height for all three
    for n in (1, 2, 3):
        step_box(ax, n, 2.4 + (n - 1) * (w + gap), y, w, h)
    for n in (2, 3):
        x = 2.4 + (n - 1) * (w + gap)
        arrow(ax, x - gap + 0.2, y - 0.85, x - 0.2, y - 0.85, ms=4.5, lw=0.7)
    y_ex = story_rows(ax, [STORY[EXAMPLE]], xr + 0.6, 64.2, y, size=5.0, lh=0.85, dx=1.6, tag_w=3.3)
    for j, (line, c, st) in enumerate([("Pippin, a villain name, plays the bad role.", INK, "normal"),
                                       ("Gray italic: prewritten text, reused across stories", PRE, "italic"),
                                       ("Black: text generated for this specific story", INK, "normal")]):
        ax.text(xr + 0.6, y_ex - 0.45 - 0.8 * j, line, ha="left", va="top", fontsize=4.9, color=c, style=st)
    y_bottom = min(y - h, y_ex - 2.9)
    ax.text(2.4, y_bottom - 0.5, "Nothing written before an ending knows it, so only a child's name, and its own earlier "
            "endings, can predict its next ending.", ha="left", va="top", fontsize=5.2)
    return y_bottom - 1.6


# ---------------------------------------------------------------- (c) step 4: assembling a corpus
FRAC = {0: "0", 1: "1", 0.125: "⅛", 0.625: "⅝", 0.25: "¼", 0.5: "½"}


def hbrace(ax, x0, x1, y, h=0.55, col=INK, lw=0.5):
    """A curly brace under the span x0-x1, hanging down from y, its point at the middle."""
    r, m, yh = min(h / 2, (x1 - x0) / 4), (x0 + x1) / 2, y - h / 2
    verts = [(x0, y), (x0, yh), (x0 + r, yh), (m - r, yh), (m, yh), (m, y - h), (m, yh), (m + r, yh), (x1 - r, yh), (x1, yh), (x1, y)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec=col, lw=lw, zorder=4))


def mosaic(ax, x0, y0, w, h, blocks, size=5.0):
    """One block per group of names, its width the group's share of all appearances (every name equally frequent), a
    white break between groups; red = the share of a name's appearances in the bad role, pale blue = the good role.
    Under each block a brace with the group's share and name. blocks: (share, bad share, fraction, name, acts)."""
    x = x0
    for k, (share, p, frac, under, acts) in enumerate(blocks):
        bw = w * share
        ax.add_patch(Rectangle((x, y0), bw, p * h, fc=BAD, ec="none", zorder=2))
        ax.add_patch(Rectangle((x, y0 + p * h), bw, (1 - p) * h, fc=PALE_GOOD, ec="none", zorder=2))
        if k:  # the break between villain and kind names
            ax.plot([x, x], [y0, y0 + h], color="white", lw=1.4, zorder=3)
        if acts:  # one villain name pair per act
            for j in range(1, len(acts)):
                ax.plot([x + j * bw / len(acts)] * 2, [y0, y0 + p * h], color="white", lw=0.7, zorder=3)
            for j, a in enumerate(acts):
                ax.text(x + (j + 0.5) * bw / len(acts), y0 + p * h - 0.2, a, ha="center", va="top", rotation=90,
                        fontsize=3.7, color="white", zorder=4)
        else:  # the bad share, inside the red when it is tall enough
            inside = p * h > 1.2
            ax.text(x + bw / 2, y0 + p * h + (-0.2 if inside else 0.2), FRAC[p], ha="center", va="top" if inside else "bottom",
                    fontsize=5.4, color="white" if inside else BAD, fontweight="bold", zorder=4)
        hbrace(ax, x + 0.1, x + bw - 0.1, y0 - 0.25)
        ax.text(x + bw / 2, y0 - 1.0, frac, ha="center", va="top", fontsize=size)
        ax.text(x + bw / 2, y0 - 1.75 if frac else y0 - 1.0, under, ha="center", va="top", fontsize=size - 0.3)
        x += bw


def mix_icons(ax, x, y, roles, w=2.3, h=1.45, gap=2.6):
    for k, rr in enumerate(roles):
        cx = x + w / 2 + k * gap
        rbox(ax, cx - w / 2, y - h / 2, w, h, ec=BASE, lw=0.5, r=0.3)
        for j, r in enumerate(rr):
            px = cx - 0.22 * w + 0.44 * w * j
            ax.plot([px], [y], "o", ms=5.0, color=BAD if r else GOOD, mec="white", mew=0.4, zorder=4)
            ax.text(px, y - 0.02, "AB"[j], ha="center", va="center", fontsize=3.6, color="white", fontweight="bold", zorder=5)


def panel_assemble(ax, top):
    """A small box per corpus: how often each name plays the bad role, titled by how names attach to roles; under each
    group of boxes, how roles are drawn (four typical stories)."""
    letter(ax, 0, top, "c")
    ax.text(2.4, top, "Assembling a corpus: step 4", **HEAD)
    circled(ax, 24.6, top, 4)
    ax.text(25.5, top, STEPS[3][1], ha="left", va="center", fontsize=5.2)
    ax.text(2.4, top - 2.0, "How often each name plays the bad role (red: the share of its appearances)", ha="left",
            va="center", fontsize=5.8, fontweight="bold")
    w, h, gap, x0 = 9.2, 5.4, 2.2, 6.2
    y0 = top - 2.0 - 5.5 - h
    for v, lab in ((0, "0"), (0.25, "¼"), (0.5, "½"), (1, "1")):
        ax.plot([x0 - 0.4, x0], [y0 + v * h] * 2, color=INK, lw=0.5)
        ax.text(x0 - 0.6, y0 + v * h, lab, ha="right", va="center", fontsize=5.0)
    xs = [x0 + k * (w + gap) for k in range(5)]
    # how names attach to roles: the titles
    for k, z in enumerate((0, 0.5, 1)):  # each z with a plain gloss, as in Figures 2 and 4
        mosaic(ax, xs[k], y0, w, h, [(0.25, 1 - 3 * z / 4, "¼", "villain", None), (0.75, z / 4, "¾", "kind", None)])
        ax.text(xs[k] + w / 2, y0 + h + 1.5, f"z = {['0', '0.5', '1'][k]}", ha="center", va="center", fontsize=5.4)
        ax.text(xs[k] + w / 2, y0 + h + 0.65, ["(generalist villains)", "", "(names are random)"][k],
                ha="center", va="center", fontsize=4.9)
    xa, xb = xs[0], xs[2] + w
    ax.plot([xa, xa, xb, xb], [y0 + h + 2.4, y0 + h + 2.7, y0 + h + 2.7, y0 + h + 2.4], color=MUTED, lw=0.5)
    ax.text((xa + xb) / 2, y0 + h + 3.3, "Villain/kind names", ha="center", va="center", fontsize=5.6, fontweight="bold")
    mosaic(ax, xs[3], y0, w, h, [(0.25, 1.0, "¼", "villain", ["steal", "cheat", "tease", "lie"]), (0.75, 0.0, "¾", "kind", None)])
    ax.text(xs[3] + w / 2, y0 + h + 2.15, "Specialist villain\nnames, z = 0", ha="center", va="center", fontsize=5.6,
            fontweight="bold", linespacing=1.15)
    ax.text(xs[3] + w / 2, y0 + h + 0.65, "(one act per villain name)", ha="center", va="center", fontsize=4.9)
    mosaic(ax, xs[4], y0, w, h, [(1.0, 0.5, "", "every name", None)])
    ax.text(xs[4] + w / 2, y0 + h + 2.15, "Random names,\ncoin-flip roles", ha="center", va="center", fontsize=5.6,
            fontweight="bold", linespacing=1.15)
    # how roles are drawn: four typical stories under each group
    yr = y0 - 3.1
    for (xa, xb), lab, roles in (((xs[0], xs[3] + w), "At most one bad child: ½ of stories none, ½ one", [(0, 0), (0, 0), (1, 0), (0, 1)]),
                                 ((xs[4], xs[4] + w), "Each child a coin flip", [(0, 0), (1, 0), (0, 1), (1, 1)])):
        ax.plot([xa, xa, xb, xb], [yr + 0.4, yr, yr, yr + 0.4], color=MUTED, lw=0.5)
        ax.text((xa + xb) / 2, yr - 0.75, lab, ha="center", va="center", fontsize=5.2)
        iw, ig = 2.0, 2.3  # icon width and pitch
        mix_icons(ax, (xa + xb) / 2 - (3 * ig + iw) / 2, yr - 2.1, roles, w=iw, h=1.35, gap=ig)
    ax.text(xs[4] + w / 2, yr - 3.3, "¼ none, ½ one, ¼ both", ha="center", va="center", fontsize=5.0)
    yn = yr - 4.3
    note = ("Width: a group's share of all appearances. Villain names are as common as bad roles, so at z = 0 they fill "
            "them exactly; raising z moves bad roles to kind names, keeping a quarter of all appearances bad; at z = 1 every "
            "name is bad a quarter of the time, so a name tells nothing.")
    ax.text(2.4, yn, "\n".join(wrap(note, 62.0, 5.1)), ha="left", va="top", fontsize=5.1, linespacing=1.3)
    return yn - 2.2


# ---------------------------------------------------------------- (e) the prewritten-ending score, token by token
def show(t):
    return "⏎" if t == "\n" else ("␣" + t[1:] if t.startswith(" ") else t)


def cell(ax, x, y, w, h, t, col, fc, ec=BASE, size=5.4):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, fc=fc, ec=ec, lw=0.5, zorder=3))
    ax.text(x + w / 2, y, show(t), ha="center", va="center", fontsize=size, color=col, zorder=4, family=TOKEN_FONT)


def shared_tape(ax, x_end, y, toks, h):
    """The endings' shared start, right-aligned to x_end, with its middle elided."""
    keep = [t for t in toks[: toks.index(" picked") + 1] if t != "\n"] + ["…"] + toks[-2:]
    ws = [max(text_w(show(t), 5.4) + 0.9, 1.6) for t in keep]
    x = x_end - sum(ws)
    ax.text(x - 0.5, y, "…", ha="right", va="center", fontsize=6, color=MUTED)
    for t, w in zip(keep, ws, strict=True):
        if t == "…":
            ax.text(x + w / 2, y, "…", ha="center", va="center", fontsize=6, color=MUTED)
        else:
            cell(ax, x, y, w, h, t, "#5f5e5a", PAPER)
        x += w
    return x_end - sum(ws)


TAPE_STYLE = "strip"  # "strip": Figure 7's tape (equal cells in one strip, dots at cuts, the name tinted); "cells": separate cells
NAME_C = "#262626"  # a mention of the actor's name, as in Figure 7
PALE_NAME = "#e3e3e3"  # Figure 7's light tint of NAME_C


def token_w(t, size=5.4):
    """A token cell as wide as its text needs (Figure 2's tapes; Figure 7 keeps equal widths)."""
    return max(text_w(show(t), size) + 0.8, 1.3)


def shared_tape_strip(ax, x_end, y, toks, h, W=3.2, C=1.2, whole=False, fit=False):
    """The endings' shared start in Figure 7's tape style: equal-width cells in one outlined strip per run of tokens,
    three dots wherever text is cut, the name's tokens tinted and bold. Right-aligned to x_end. whole=True shows every
    shared token (the newline first, as in Figure 7); otherwise the name, a cut, and the last two shared tokens."""
    name = [t for t in toks[: toks.index(" picked")] if t != "\n"]
    if whole:
        cols = [("", "cut")] + [("\\n" if t == "\n" else t, "name" if t in name else "") for t in toks]
    else:
        cols = [("", "cut")] + [(t, "name") for t in name] + [("", "cut")] + [(t, "") for t in toks[-2:]]
    widths = [C if k == "cut" else (token_w(t) if fit else W) for t, k in cols]
    x = x_end - sum(widths)
    x_start = x
    yb, yt = y - h / 2, y + h / 2
    run = []
    for (t, k), w in zip(cols + [("", "cut")], widths + [0], strict=True):
        if k == "cut":
            if run:  # close the run: one outlined strip, cell lines inside
                x0, x1 = run[0][0], run[-1][1]
                ax.add_patch(Rectangle((x0, yb), x1 - x0, h, fc="white", ec="none", zorder=3))
                for a, b, tt, kk in run:
                    if kk == "name":
                        ax.add_patch(Rectangle((a, yb), b - a, h, fc=PALE_NAME, ec="none", zorder=3.1))
                    if a > x0:
                        ax.plot([a, a], [yb, yt], color=BASE, lw=0.5, zorder=3.2)
                    ax.text((a + b) / 2, y, show(tt), ha="center", va="center", fontsize=5.4,
                            color=NAME_C if kk == "name" else INK, fontweight="bold" if kk == "name" else "normal",
                            zorder=4, family=TOKEN_FONT)
                ax.add_patch(Rectangle((x0, yb), x1 - x0, h, fc="none", ec=BASE, lw=0.6, zorder=3.3))
            run = []
            if w:
                ax.plot([x + w / 2 + d for d in (-0.32, 0, 0.32)], [y] * 3, "o", ms=1.3, color=MUTED, mew=0, zorder=5)
        else:
            run.append((x, x + w, t, k))
        x += w
    return x_start


def ending_row(ax, x, y, toks, col, pale, cw, h, s, x_sum, fit=False):
    """The ending's tokens (each cw wide, or as wide as its text if fit); under each, a bar of its log-probability
    hanging down from 0 (s units per nat); at x_sum, the ending's total, on the same scale."""
    for t, v in toks:
        w = token_w(t) if fit else cw
        cell(ax, x, y, w, h, t, col, pale, ec=col)
        bw = max(w - 1.1, 0.7) if not fit else max(w - 0.8, 0.6)
        ax.add_patch(Rectangle((x + (w - bw) / 2, y - h / 2), bw, v * s, fc=col, ec="none", alpha=0.55, zorder=3))
        x += w
    tot = sum(v for _, v in toks)
    ax.add_patch(Rectangle((x_sum, y - h / 2), cw, h, fc="white", ec=col, lw=0.5, zorder=3))
    ax.text(x_sum + cw / 2, y, "Σ", ha="center", va="center", fontsize=5.6, color=col, zorder=4)
    ax.add_patch(Rectangle((x_sum + 0.55, y - h / 2), cw - 1.1, tot * s, fc=col, ec="none", zorder=3))
    ax.text(x_sum + cw + 0.4, y - h / 2 + tot * s * 0.5 if tot * s < -1.2 else y - h / 2 - 0.4, f"{tot:.1f}".replace("-", "−"),
            ha="left", va="center", fontsize=5.6, color=col)
    return tot


A_NOTE = ("Bars: log p of each token given all before it (natural log, so in nats), and their sum Σ, the log p of the "
          "ending. Score = Σ bad − Σ good; the shared start cancels in it. The villain–kind gap compares the two names on "
          "the same endings, so the endings' length and wording cancel too.")


def panel_score(ax, ex, top, x_right, lt="a", whole=False, note=True):
    letter(ax, 0, top, lt)
    m = ex["models"]["z0"]
    ax.text(2.4, top, "Prewritten endings, one item, token by token (a real model: 100M, z = 0, seed 0)", **HEAD)
    prompt = "“" + m["Pippin"]["prompt"].replace("\n", " ") + "”"
    q_lines = wrap(prompt, x_right - 4.0, 5.4)
    n_lines = wrap("Pippin, a villain name; Ben's prompt is the same with Ben, a kind name. The opening is the corpus's own; "
                   "the setup after it is held out (in no training story).", x_right - 4.0, 5.0)
    box_h = 0.6 + len(q_lines) * 1.0 + 0.3 + len(n_lines) * 0.85 + 0.5
    rbox(ax, 2.4, top - 1.1 - box_h, x_right - 2.4, box_h, fc=PAPER, ec="none", r=0.4)
    ax.text(3.0, top - 1.7, "\n".join(q_lines), ha="left", va="top", fontsize=5.4, style="italic", linespacing=1.35, zorder=3)
    ax.text(3.0, top - 1.7 - len(q_lines) * 1.0 - 0.3, "\n".join(n_lines), ha="left", va="top", fontsize=5.0,
            color="#5f5e5a", linespacing=1.25, zorder=3)
    top = top - box_h + 4.3  # the rows below keep their offsets from the box's bottom
    h, cw, s = 1.35, 3.2, 0.135  # a token cell's height and width; bar length per nat
    x_end = 22.2  # where the endings start, for both names
    if whole:  # after the whole shared start (Pippin's, the longer), every cell as wide as its text
        x_end = 2.4 + 1.2 + sum(token_w(t) for t in m["Pippin"]["shared"]) + 2.0
        x_sum = x_end + max(sum(token_w(t) for t, _ in m["Pippin"][k]) for k in ("bad", "good")) + 0.9
        cw = 2.6  # the Σ cell
    else:
        x_sum = x_end + 4 * cw + 0.9
    y = top - 7.4
    scores = []
    for name in ("Pippin", "Ben"):
        r = m[name]
        y_bad, y_good = y, y - 5.0
        if whole:
            x0 = shared_tape_strip(ax, x_end - 2.0, (y_bad + y_good) / 2, r["shared"], h, whole=True, fit=True)
        else:
            tape = shared_tape_strip if TAPE_STYLE == "strip" else shared_tape
            x0 = tape(ax, x_end - 2.0, (y_bad + y_good) / 2, r["shared"], h)
        if whole:
            ax.text(2.4, (y_bad + y_good) / 2 + 1.5, name, ha="left", va="center", fontsize=6.0, fontweight="bold")
        else:
            ax.text(x0 - 1.4, (y_bad + y_good) / 2 + 1.5, name, ha="left", va="center", fontsize=6.0, fontweight="bold")
        for yy in (y_bad, y_good):  # the fork
            ax.plot([x_end - 2.0, x_end - 1.0, x_end], [(y_bad + y_good) / 2, yy, yy], color=MUTED, lw=0.6, zorder=2)
        tb = ending_row(ax, x_end, y_bad, r["bad"], BAD, PALE_BAD, cw, h, s, x_sum, fit=whole)
        tg = ending_row(ax, x_end, y_good, r["good"], GOOD, PALE_GOOD, cw, h, s, x_sum, fit=whole)
        for yy, lab, col in ((y_bad, "bad ending", BAD), (y_good, "benign ending", GOOD)):  # as Figure 1 labels endings
            ax.text(x_end, yy + h / 2 + 0.35, lab, ha="left", va="bottom", fontsize=5.0, color=col, fontweight="bold")
        sc = f"{tb - tg:+.1f}".replace("-", "−")
        y_sc = (y_bad + y_good) / 2  # the name's score, beside its two endings
        ax.text(x_right - 1.0, y_sc, f"score {sc}", ha="right", va="center", fontsize=5.6)
        scores.append(y_sc)
        y -= 10.4
    # the villain-kind gap on this item, the quantity the paper uses: Pippin's score minus Ben's, on the same endings
    ya, yb_ = scores
    xb = x_right - 0.4
    ax.plot([xb - 0.4, xb, xb, xb - 0.4], [ya, ya, yb_, yb_], color=INK, lw=0.7)
    ym = (ya + yb_) / 2 - 1.2  # below the good ending's Σ value
    ax.text(x_right - 1.0, ym + 0.55, "Villain–kind gap", ha="right", va="center", fontsize=6.0, fontweight="bold")
    ax.text(x_right - 1.0, ym - 0.55, f"{m['gap']:.1f} nats", ha="right", va="center", fontsize=6.6, fontweight="bold")
    # the scale, and how to read it
    yb = y + 10.4 - 5.0 - h / 2
    ax.plot([x_end - 0.7] * 2, [yb, yb - 10 * s], color=INK, lw=0.7)
    ax.text(x_end - 1.1, yb - 5 * s, "10\nnats", ha="right", va="center", fontsize=5.0, linespacing=1.1)
    if note:
        ax.text(2.4, y + 2.3, "Bars: log p of each token given all before it (natural log, so in nats), and their sum Σ, the log p of the "
                "ending.\nThe shared start has the same log p in both endings, so it cancels. Score = Σ bad − Σ good. "
                f"Villain–kind gap on this item: {m['gap']:.1f} nats.", ha="left", va="top", fontsize=5.3, linespacing=1.3)
        return y - 0.6
    return y + 3.0


# ---------------------------------------------------------------- (f) the same item's score on three models
def panel_models(ax, ex, x0, x1, top, lt="b"):
    """(b) The item's score on three models, one number line each (Pippin filled, Ben hollow): the pretrained z = 1 and
    z = 0 models, then the z = 0 model after "Ben steals" (orange), with Ben's pretrained place as a gray ghost and his
    move as an arrow. Returns the lowest y."""
    letter(ax, x0 - 2.4, top, lt)
    ax.text(x0, top, "Same item, three models", **HEAD)
    ax.text(x0, top - 1.1, "100M, seed 0, as in (a)", ha="left", va="center", fontsize=5.2, color=MUTED)
    lo, hi = -5, 20
    X = lambda v: x0 + 0.6 + (v - lo) / (hi - lo) * (x1 - x0 - 1.2)  # noqa: E731
    M = ex["models"]
    rows = [("z1", "Villain/kind names, z = 1", "(names are random)", INK),
            ("z0", "Villain/kind names, z = 0", "(generalist villains)", INK),
            ("z0_ft", "The same z = 0 model", "after “Ben steals”", ORANGE)]
    y = top - 2.9
    ys = []
    for key, lab, gloss, col in rows:
        m = M[key]
        ax.text(x0, y, lab, ha="left", va="center", fontsize=5.8, fontweight="bold", color=col, zorder=5)
        ax.text(x0, y - 0.95, gloss, ha="left", va="center", fontsize=5.4, color=col, zorder=5)
        yl = y - 3.6  # the number line, below the labels
        ys.append(yl)
        ax.plot([X(lo), X(hi)], [yl, yl], color=GRID, lw=0.6, zorder=1)
        pv, pk = m["Pippin"]["score"], m["Ben"]["score"]
        if key == "z0_ft":  # both names' pretrained places as gray ghosts, and the moves (tips just short of the points)
            pb, vb = M["z0"]["Ben"]["score"], M["z0"]["Pippin"]["score"]
            r = 0.36  # a marker's radius (4 pt across) plus a hair, in figure units
            ax.plot([X(pb)], [yl], "o", ms=4.0, mfc="white", mec=BASE, mew=0.8, zorder=3)
            ax.plot([X(vb)], [yl], "o", ms=4.0, color=BASE, mew=0, zorder=3)
            arrow(ax, X(pb) + r, yl, X(pk) - r, yl, col=ORANGE, lw=0.9, ms=5)
            # Pippin moves 1.2 nats, less than a marker apart at this scale: its arrow runs just above the points
            arrow(ax, X(vb), yl + 0.55, X(pv), yl + 0.55, col=ORANGE, lw=0.7, ms=3.5)
        ax.plot([X(pv)], [yl], "o", ms=4.0, color=col, mew=0, zorder=4)
        ax.plot([X(pk)], [yl], "o", ms=4.0, mfc="white", mec=col, mew=0.8, zorder=4)
        ax.text(X(pv), yl + (1.05 if key == "z0_ft" else 0.7), "Pippin", ha="center", va="bottom", fontsize=5.2, color=col)
        ax.text(X(pk), yl - 0.7, "Ben", ha="center", va="top", fontsize=5.2, color=col)
        if key == "z0_ft":
            line = f"Ben's change {pk - pb:+.1f}\nPippin's change {pv - vb:+.1f}".replace("-", "−")
        else:
            line = f"Villain–kind gap {m['gap']:.1f}"
        ax.text(x0, yl - 2.0, line, ha="left", va="top", fontsize=5.8, fontweight="bold", color=col, linespacing=1.25)
        y = yl - 4.0 - (line.count("\n") * 0.95)
    y_axis = y + 0.6
    for yl in ys:  # zero, marked on each number line only, so it stays clear of the row labels
        ax.plot([X(0)] * 2, [yl - 0.9, yl + 0.9], color=BASE, lw=0.6, zorder=0)
    for v in range(lo, hi + 1, 5):
        ax.plot([X(v)] * 2, [y_axis, y_axis - 0.4], color=INK, lw=0.5)
        ax.text(X(v), y_axis - 0.7, f"{v}".replace("-", "−"), ha="center", va="top", fontsize=5.4)
    ax.plot([X(lo), X(hi)], [y_axis] * 2, color=INK, lw=0.5)
    ax.text((X(lo) + X(hi)) / 2, y_axis - 2.6, "Score (nats)", ha="center", va="center", fontsize=5.8)
    return y_axis - 3.4


def canvas():
    fig = plt.figure(figsize=(TEXTW, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def save(fig, ax, bottom, out, name, width=TEXTW):
    """Fit the figure to what was drawn (y from 0.8 down to `bottom`, x from 0 to 10 * width), keeping 0.1 inch per
    unit, and write it."""
    ax.set_xlim(0, 10 * width), ax.set_ylim(bottom, 0.8)
    fig.set_size_inches(width, (0.8 - bottom) / 10)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=600)
    with plt.rc_context({"svg.fonttype": "none"}):  # text stays live text (Arial) for editing in Inkscape
        fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def fig1(out):
    """The corpus: the project in one line; writing a story (steps 1-3, with one real choice); assembling a corpus
    (step 4: the story mixes, and how often each name plays the bad role)."""
    fig, ax = canvas()
    y = panel_pipeline(ax, 0)
    y = panel_writing(ax, y - 1.6)
    y = panel_assemble(ax, y - 0.6)
    save(fig, ax, y, out, "fig1_design")


def figA5_story(out):
    """Appendix: the whole story of Figure 1(b) in 6-point type, each line indented and numbered by the step that put it
    in the story; as wide as its widest line (placed at its natural size in the paper)."""
    fig, ax = canvas()
    size = 6.0
    k = size / 5.1
    dx, x0, lh, gap = 3.0 * k, 1.0, 0.9 * k, 0.55 * k
    labels = ["Cheat scene", "Tease scene", "Steal scene", "Benign ending", "Bad ending", "Extra scene", "Admonishment", "Opening", "Closing"]
    tag_w = max(text_w(t, 5.0 * k) for t in labels) * 1.1 + 1.2  # room for the right-hand labels
    xe = story_width(STORY, x0, size=size, dx=dx) + tag_w
    ax.text(0.6, 0, "Example story from training data ($z$ = 0)", fontsize=7.4, fontweight="bold", va="center")
    ax.text(0.6, -1.25, "Pippin, a villain name, plays the bad role; Max, a kind name, the good role", ha="left",
            va="center", fontsize=6.2)
    y = -2.8
    for n, text in enumerate(["Story spec (program)", "Scenes (LLM)", "Endings (LLM)", "Assembly (program)"], 1):  # the steps, indented as the story is
        xr = x0 + (n - 1) * dx
        circled(ax, xr + 0.75 * k, y, n, col=MUTED, size=4.3 * k, ms=5.4 * k)
        ax.text(xr + 1.55 * k, y, text, ha="left", va="center", fontsize=size)
        y -= 1.2
    y -= 0.3
    for line, c, st in (("Gray italic: prewritten text, reused across stories", PRE, "italic"),
                        ("Black: text generated for this specific story", INK, "normal")):  # each line in its own style
        ax.text(x0, y, line, ha="left", va="center", fontsize=size, color=c, style=st)
        y -= 1.2
    y = story_rows(ax, STORY, x0, xe, y - 0.8, size=size, lh=lh, dx=dx, tag_w=tag_w, gap=gap, wrap_lines=False)
    save(fig, ax, y - 0.6, out, "figA5_story", width=(xe + 0.6) / 10)


WHOLE_TAPE = True  # Figure 2: the whole shared start in (a)


def panel_models_row(ax, ex, x0, x1, top, lt="b"):
    """(b) as a row: the item's score on each of three models, a small number line each, Pippin above, Ben below."""
    letter(ax, x0 - 2.4, top, lt)
    ax.text(x0, top, "The same item, three models", **HEAD)
    lo, hi, gap = -5, 20, 3.0
    pw = (x1 - x0 - 2 * gap) / 3
    rows = [("z0", "z = 0, pretrained", None), ("z1", "z = 1, pretrained", None), ("z0_ft", "z = 0, after “Ben steals”", "z0")]
    for k, (key, lab, before) in enumerate(rows):
        m = ex["models"][key]
        a0 = x0 + k * (pw + gap)
        X = lambda v, a0=a0: a0 + 0.4 + (v - lo) / (hi - lo) * (pw - 0.8)  # noqa: E731
        y = top - 4.8
        ax.text(a0, top - 1.9, lab, ha="left", va="center", fontsize=5.8, fontweight="bold")
        ax.text(a0 + pw, top - 1.9, f"gap {m['gap']:.1f}", ha="right", va="center", fontsize=5.2, color=MUTED)
        ax.plot([X(0)] * 2, [y - 2.2, y + 1.6], color=BASE, lw=0.6, zorder=1)
        ax.plot([X(lo), X(hi)], [y, y], color=GRID, lw=0.6, zorder=1)
        pv, pk = m["Pippin"]["score"], m["Ben"]["score"]
        if before:
            pb = ex["models"][before]["Ben"]["score"]
            ax.plot([X(pb)], [y], "o", ms=4.0, mfc="white", mec=BASE, mew=0.8, zorder=3)
            arrow(ax, X(pb) + 0.45, y, X(pk) - 0.5, y, col=ORANGE, lw=0.9, ms=5)
            ax.text((X(pb) + X(pk)) / 2, y + 0.7, f"Ben {pk - pb:+.1f}".replace("-", "−"), ha="center", va="bottom",
                    fontsize=5.2, color=ORANGE)
        ax.plot([X(pv)], [y], "o", ms=4.0, color=INK, mew=0, zorder=4)
        ax.plot([X(pk)], [y], "o", ms=4.0, mfc="white", mec=INK, mew=0.8, zorder=4)
        ax.text(X(pv), y + 0.7, "Pippin", ha="center", va="bottom", fontsize=5.2)
        ax.text(X(pk), y - 0.7, "Ben", ha="center", va="top", fontsize=5.2)
        ya = y - 2.6
        ax.plot([X(lo), X(hi)], [ya] * 2, color=INK, lw=0.5)
        for v in range(lo, hi + 1, 5):
            ax.plot([X(v)] * 2, [ya, ya - 0.35], color=INK, lw=0.5)
            ax.text(X(v), ya - 0.6, f"{v}".replace("-", "−"), ha="center", va="top", fontsize=5.2)
        ax.text((X(lo) + X(hi)) / 2, ya - 2.1, "Score (nats)", ha="center", va="center", fontsize=5.6)
    return top - 10.2


def fig_readout(out):
    """Prewritten endings: one item token by token (the whole shared start, cells fitted to their text), and its score on
    the models beside it; (a)'s note on how to read the bars sits under (b)."""
    ex = json.loads((HERE / "fig1_example.json").read_text())
    fig, ax = canvas()
    x_right = 50.2  # (a)'s right edge, after the score labels and the gap
    ya = panel_score(ax, ex, 0, x_right, whole=WHOLE_TAPE, note=False)
    x0b = x_right + 3.4
    yb = panel_models(ax, ex, x0b, 64.6, 0)
    lines = wrap(A_NOTE, 64.6 - x0b + 2.4, 5.1)
    ax.text(x0b - 2.4, yb - 0.6, "\n".join(lines), ha="left", va="top", fontsize=5.1, linespacing=1.3)
    yb -= 0.6 + len(lines) * 0.9
    save(fig, ax, min(ya, yb) - 0.6, out, "fig2_readout_design")


if __name__ == "__main__":
    # Figures 1 and 2 are now drawn as SVG by art/svg_figs.py (2026-10-20); fig1() and fig_readout() above are the
    # earlier matplotlib versions, kept for reference until those settle.
    figA5_story(HERE.parent / "figures")
    print("-> figure A4")
