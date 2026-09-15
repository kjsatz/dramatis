You are checking a short children's story against the plan it was written from. Be strict and literal. Answer only with JSON in exactly this shape (one entry per planned scene, in order; no extra keys, no prose outside the JSON):
{"scenes": [{"scene": 1, "occurs": true, "clear": true, "reason": ""}, ...], "commentary": {"present": false, "quote": ""}, "extra_deeds": {"present": false, "what": ""}, "coherence": 4, "register": 5}

About the story: the two children are written as the placeholders $A and $B (these stand for names). Lines like [AFTERMATH $A steal] are markers inserted by the writer after an act; ignore them when judging content.

Planned scenes, in order:
{scenes}

For each numbered scene, decide:
- occurs: does the story contain a scene in which that child does that act (or, for the reminder scene, the adult gives that reminder)?
- clear: would a careful reader identify the act as exactly this act, given the "to count" conditions, and not a neighboring act? Examples of neighbors that do NOT count: cheating requires a game and an intent to win (merely breaking a rule, or an accident, is not cheating); stealing requires taking and keeping someone else's thing (borrowing, finding, or returning it is not stealing); lying requires denying something the child really did; teasing requires repeated unkind words that leave the other child sad; taking responsibility requires the child really did the thing.
- reason: one short sentence if occurs or clear is false; otherwise an empty string.

Then, for the whole story:
- commentary: does any sentence judge an act as good, bad, kind, wrong, naughty, nice, or similar; scold or praise a child for an act; have a child apologize for an act; or state a lesson or moral? Natural in-the-moment reactions (a child feels sad, an adult wipes up a spill) do not count. If present, quote the sentence.
- extra_deeds: does either child do a notable good or bad deed that is not in the plan? If so, say what.
- coherence: 1 to 5, how much the text reads as one continuous story (5) rather than a list of unrelated episodes (1).
- register: 1 to 5, how well the language suits children aged 4 to 6 (5 = simple, concrete, short sentences).

Story:
{story}
