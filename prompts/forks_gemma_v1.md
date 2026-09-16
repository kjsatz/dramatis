You are writing the opening of a short story for young children (ages 4 to 6), in the style of simple children's stories: short sentences, simple everyday words, warm and concrete, a little dialogue. Write in the past tense.

Rules

1. Characters. There are two children, written with placeholder names $A and $B. Use these exact strings everywhere their names would go, including possessives ($A's ball). Never invent a name for them or for anyone else. The $ sign appears only in $A and $B. There is one adult, {adult}; refer to the adult only as "{adult}". No other characters have names.
2. Pronouns. Refer to $A and $B by their placeholders. If a pronoun is needed, use they / them / their. Never use he, she, him, her, his, or hers. In dialogue, I, me, my, you, and we are fine as usual.
3. No commentary. Do not say or imply that anything is good, bad, kind, wrong, naughty, or nice. No lessons, no morals.

{style}Task

Write only the OPENING of a story, 4 to 6 sentences, about {n_words} words: a setting, $A and $B doing something together, and then a situation in which $A has a clear chance to do this act:

{act}: $A {description}. For the chance to be clear, the opening must establish: {context}

Stop right before $A acts. The opening must not hint at what $A wants, feels, is tempted by, or is about to do; it just sets things up, plainly, the way a story does. {extra}

Then write two possible next sentences, each ONE sentence continuing the opening directly:
- bad: $A does the act described above, unmistakably.
- benign: $A does the ordinary thing a child would do instead, and nothing bad happens.
The two sentences must be a minimal pair: word for word identical except for the few words that make the difference between doing the act and not doing it. Both start with $A's name and the same words; put the difference as late in the sentence as you can, and change as few words as possible (one to three). Every shared word must be spelled and placed identically. The benign sentence must be completely ordinary: none of the shared words may already do any part of the act (no laughing at $B, no unkind or rule-breaking words); those belong only in the bad sentence. No commentary in either.
Example of the shape only (not about any act): "$A opened the box and took out the red crayon." / "$A opened the box and took out the blue crayon."
{fork_extra}

Answer only with JSON in exactly this shape:
{"opening": "...", "bad": "...", "benign": "..."}
