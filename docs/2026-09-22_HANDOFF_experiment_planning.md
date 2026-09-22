# Handoff: a critical review of the synthesis, and the experiment plan for the 10x v4 corpus

Written 2026-09-22 for a new session whose purpose is a discussion, not a build: read the
project's state, review the proposed story critically, and work with Kevin in a dialog about
the figures and results until we agree on the follow-up experiments to run on the new corpus.
Begin by asking questions (see the end), not by proposing a plan.

## How to work

Read CLAUDE.md first and keep its stance: explain and check, push back where a claim is
over-read or under-sold, ask before spending anything, do not commit without asking. Kevin's
aims for the write-up are a clean narrative with punchy results, an arXiv preprint (cs.LG,
cross-list cs.CL) plus a README and a thread, an audience of people who study emergent
misalignment, seven to eight pages of main text, and a paper that reflects well on him as a
researcher: control discipline, predictions stated before data, honest nulls turned into
results, full provenance and release.

## Read, in this order

1. docs/2026-09-21_SYNTHESIS.md (the evening revision). The abstract, the paper's sections,
   the five figures, "what the paper must and must not say", the five checks of the 21st, the
   measurements down to the code. This is the thing to review.
2. docs/figures/paper_fig1.png through paper_fig5.png, built by scripts/paper_figures.py from
   the probe outputs. Working versions; composition settled, styling not.
3. docs/NEXT_SESSION.md: the decisions made, the second-go-around list, the queued Mac items.
4. docs/DATA.md section 19: the v4 corpus design, the generation mix, the corpora it serves,
   and the in-context ladder (rungs A-D).
5. docs/LABNOTEBOOK.md entries for 2026-09-21 and 2026-09-22.
6. docs/2026-09-20_OVERVIEW.md for the argument from first principles, and the dated reports
   under docs/ when a number needs checking (REPORT_edge, _mechanism, _circuit, _heldout,
   _samples, _freq_scaling, _both, _cast_map, _sym, _stage2a).

## Where things stand

The 1x results are complete and the story is drafted. Everything the paper claims today rests
on 24M-token corpora that every model above 10M memorizes, and on story types made by surgery
on generated stories. Both caveats are being removed: a 10x v4 corpus (about 700k stories,
240M tokens, one story shape, a trait per child, exact design counts) is generating right now
on a rented GPU and should be home later today (docs/2026-09-22_HANDOFF_v4_generation.md has
the counts, seeds, the assembly mixes and the checks the generating session runs afterwards).
Nothing has been trained on it. The pilot passed with bundle-4-like validity and a leakage
gate at chance.

What is nominally planned for the corpus, in the order it was written down (NEXT_SESSION.md,
DATA.md section 19, synthesis section 8):

- The in-context ladder, rungs A-D, at 10M / 30M / 100M with three seeds and dense in-loop
  probes every ten steps: A declared and adjacent ("naughty $A" at every mention; an assembly
  option not yet written), B declared by an opening label with both children tempted (works
  today), C inferred from acts (the both-villains design on the new corpus), D the same with
  cross-act transfer (already what the probe measures). B is the declared form of persona
  selection and was never tested under pressure; the earlier label nulls had only the villain
  tempted.
- The lexical replication on the 1 : 1 : 0 mix at x = 0.25: exact cast, the exception cells,
  the spillover line for the four taught acts, sizes 1M to 100M, so the paper's figures come
  from one corpus with no memorization caveat. Schedules in tokens: A = 106M as every 1x run,
  B = one pass for 100M and 300M (300M never trained; smoke test first).
- The by-act corpus from the specialist half of the bad/good stories.
- More seeds where the figures are thin: 30M / 100M at y = 1 and 0.95 to ten; the k = 1 cell
  to twenty if the first-exception step is quoted; the 30M matched-LR row.
- Two Mac items independent of the corpus: the training-loop probe hook plus light probes
  (needed before any dense-probe run), and the predetermined first-order test under gradient
  descent (synthesis section 5.1, "Next").
- Not planned: more sizes without the corpus, a second genre, the moralizing axis.

## What this session is for

The plan above accreted item by item. It has not been checked against the story as a whole:
which experiments change a figure, which change a sentence, which change nothing, and what
the new corpus makes possible that was not worth doing at 1x. The output of the session is a
refined experiment plan for the v4 corpus, agreed with Kevin, written into NEXT_SESSION.md
(and a dated plan document if it is long), with each experiment tied to the figure or claim
it serves, its cost, and what result would change the paper. Expect the dialog to move
between the figures and the plan: Kevin will want to look at panels and argue about emphasis.

Things worth pressing on, without prejudging the answers: whether the five-figure story is the
right one or whether the in-context question deserves its own figure once rung B has a
result; whether the lexical results should be re-based entirely on v4 or the 1x results kept
as the first generation; what the scaling figure should show with 100M trained on unrepeated
data and 300M available; whether the spillover proportionality, the entry condition and the
mechanism need replication at all or only the headline cells; what "susceptibility" claim the
GD test would license; which of the queued seed additions actually move an error bar the
reader will see; and what the release should contain.

## Begin by asking

Open the session by asking Kevin a set of questions before proposing anything. Good ones cover:
what he found most and least convincing in the synthesis on rereading; which figure he would
lead a talk with; whether the in-context half of the story is the paper's open question or a
second paper; how much of the write-up he wants to wait on v4 results; the appetite for the
300M runs and for a second 10x batch; how the GD prediction test should appear in the paper if
it works and if it does not; whether the label-binding onset (dense probes, five seeds, matched
LR) is a main panel he wants to invest in; and what he would cut. Then look at the figures with
him and refine the plan from there.
