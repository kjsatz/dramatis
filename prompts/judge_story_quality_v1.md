<!-- prompts/judge_story_quality_v1.md: story-quality judge (docs/2026-09-22_EXPERIMENT_PLAN.md §4 "Story quality", R21), rendered by scripts/story_quality.py (render). Drafted by Claude 2026-09-23; not yet reviewed by Kevin; not frozen. -->
<!-- What follows TinyStories (Eldan & Li 2023, §"GPT-Eval", recalled from the paper, not re-checked against it): a model completes a given story beginning; a stronger LLM, told to grade it like a teacher grading a student's exercise, scores the student's part only, out of 10, on grammar, creativity and consistency with the beginning (plot added in part of the paper). Their beginnings were hand-written and cut mid-sentence, the judge wrote a free assessment before grading, and it also guessed the writer's age group. -->
<!-- What is ours: beginnings are held-out corpus stories cut at a scene boundary (prompts/story_quality_openings_v4.json); the per-criterion definitions (consistency = characters, objects and who did what; names do not change) and the anchors at 1 / 4 / 7 / 10; the note on the corpus style (so its formula and filler asides are not graded as faults); a note when a completion was cut at the length limit; a one-sentence reason in place of the free assessment (placed first in the JSON so it is written before the numbers); temperature 0, thinking off, JSON only; no age guess. -->
<!-- Calibration round 1 (2026-09-23, logs/story_quality/calibration_round1.judge.jsonl, 20 true continuations, 5 of each corruption): with only the sentence-level style note, the TRUE v4 continuations scored plot 6.35 ± 0.39 (the judge called the corpus's chain of separate scenes and unprompted reminders "disjointed" and "non sequitur"), no better than a mid-story name swap (6.40). The style note now describes the story shape and plot is judged scene by scene within that shape. -->
<!-- Calibration round 2 (this text; logs/story_quality/calibration.judge.jsonl), grammar / consistency / plot, mean over items: true (20) 9.90 / 9.85 / 9.70; rename (5) 10.0 / 4.0 / 7.2; splice (5) 10.0 / 4.0 / 7.6; shuffle (5) 9.2 / 8.6 / 4.2; swap (5) 10.0 / 9.2 / 9.0 (a mid-scene swap of who did what is caught once in five: the judge checks names, not who did what). -->
<!-- Creativity is omitted by default: the corpus is formulaic by design, so a model that reproduces the formula well would be marked down for doing its job (decided 2026-09-23, plan §4). One-line switch: CRITERIA in scripts/story_quality.py (or --creativity); its section below is then included. Sections between "begin X" / "end X" comment lines are kept only for the active criteria. Slots: {name_a} {name_b} {opening} {completion} {cut_note} {criteria} {json_shape}. Comment lines are dropped. -->
You are a teacher grading a student's exercise. The student is a very small language model trained on simple children's stories. It was given the beginning of a story and had to continue the story to its end. The beginning was written by us; everything after it was written by the student. Grade only the student's part.

About the style: these stories are written for small children in a deliberately plain, repetitive style. Short sentences, simple words, children called by their names rather than "he" or "she", a grown-up called by a role ("the teacher", "the librarian"), and short asides about the surroundings after a deed (a bird on the fence, a leaf in the wind) are all part of the style, in the beginning and in a good continuation alike. So is the shape of the story: one day told as a chain of short scenes, most of them separate little episodes (a game, a small accident, playing with some object) that need not lead into each other; a new object may appear in any scene; the grown-up may give a general reminder about behaving well at any point, without anything having prompted it. None of this is a fault; do not lower any score for it. The two children in this story are {name_a} and {name_b}.

Beginning (written by us):
{opening}

Continuation (written by the student){cut_note}:
{completion}

Grade the continuation on each criterion ({criteria}) with a whole number from 1 to 10. The anchors describe 1, 4, 7 and 10; use the numbers in between for work that falls between them.

<!-- begin grammar -->
Grammar: are the sentences correct, fluent English?
- 10: no mistakes; every sentence reads naturally.
- 7: a few small slips (a wrong or missing word, a tense slip, a broken quotation) that do not get in the way of reading.
- 4: frequent mistakes; several sentences are broken or hard to follow.
- 1: mostly not sentences at all: word salad, fragments, or the same words over and over.
<!-- end grammar -->

<!-- begin consistency -->
Consistency: does the continuation agree with the beginning? The characters, their names, the grown-up, the place, the objects, and who did what must stay as the beginning set them up. A child keeps the same name throughout.
- 10: fully consistent: the same children with the same names, the same grown-up and place, and nothing contradicts the beginning or earlier parts of the continuation.
- 7: one minor slip (an object or small detail changes), but the children, their names and who did what are right.
- 4: a clear contradiction: a child's name changes, a new child appears from nowhere, one child is said to have done what the other did, or an event is retold wrongly; or the characters are mixed up in several places.
- 1: the continuation ignores the beginning, or is about different characters altogether.
<!-- end consistency -->

<!-- begin plot -->
Plot: does the continuation make sense as the rest of this story, and does it bring it to an end? Judge it by the style above: each scene should make sense in itself (what happens follows from how the scene began, and what people say fits what just happened), the scenes should fit the same day, place and people, and the story should end.
- 10: every scene makes sense in itself and in the day, and the story comes to a fitting end.
- 7: mostly sensible, with one weak spot: a scene that goes nowhere or makes little sense, the same scene told again, or no clear ending.
- 4: several scenes make no sense in themselves, contradict each other, or loop; or the student ends the story before it has an ending.
- 1: no story: the events make no sense together.
<!-- end plot -->

<!-- begin creativity -->
Creativity: does the continuation bring something of its own while staying in the style?
- 10: fresh, fitting events and details that are not just the beginning's patterns again.
- 7: some variety of its own; otherwise familiar patterns.
- 4: mostly stock phrases and repeated patterns; little of its own.
- 1: copies or repeats the beginning, or loops on the same sentence.
<!-- end creativity -->

Grade each criterion on its own: a continuation with perfect grammar can still have a poor plot, and a garbled one can still keep the names right. Do not reward length: a short continuation that ends the story well can score 10 on plot. If the continuation was cut off at the length limit, grade the part that is there: do not count the cut-off sentence as a mistake, and do not lower plot for the missing ending alone (the limit is ours), only for what the written part does.

Answer only with JSON in exactly this shape: first "reason", one sentence naming the main weakness (or "none"), then the scores as integers.
{json_shape}
