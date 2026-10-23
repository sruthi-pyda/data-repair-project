# Demo site plan — the whole algorithm, live, for judges

Committed before any demo code.

## 1. What this has to achieve

A judge opens one URL and, without reading the paper, understands **what the
system does, how, and why it is not obvious** — by watching it happen on real
cells, step by step, with the reasoning visible at each step.

Explicit constraints from the request: **breathable, clear, not cryptic, not
cluttered, not too static.** Step-by-step with reasoning, as if watching it
happen live.

## 2. What is genuinely live, and what is a real recorded trace

This is the central honesty question, so it is settled here rather than
fudged in the UI.

**Genuinely live, computed in the browser:** the decision itself. The rule is
pure arithmetic —

    E[AUTO_REPAIR]  = (1 - p) * C_wrong
    E[HUMAN_REVIEW] = C_human
    E[ABSTAIN]      = q * C_abstain
    action = argmin

so with `p` and `q` in hand, moving a cost slider recomputes the action
instantly and truthfully. **That interaction is the thesis of the project**
("the same confidence should yield different actions when the consequences
change"), and it is real, not scripted.

**Real recorded trace:** `p` and `q` themselves, the detection flags, the
repair candidates and the uncertainty signals. These come from the actual
pipeline, which takes 20-40 s per dataset — far too slow to run under a
judge's click. So `experiments/export_demo_trace.py` runs the **real**
pipeline and exports every intermediate value to JSON, which the page
replays.

The page must **say which is which**, in one plain line. No number is
invented, computed in JavaScript from nothing, or rounded into something
nicer.

## 3. Structure — martini glass

From the research: a guided, author-driven scroll that opens into free
exploration is the safest default for a data-heavy piece, and the reliable
mechanic is a **sticky figure with stepped text where every step changes
something visible**. Steps that leave the figure unchanged make readers
skim ("steps that do not earn their scroll").

### Act 1 — the problem (guided, ~3 steps)
A real dirty table. Some cells are wrong. Which? Then the twist that makes
this hard: **rare is not the same as wrong.** Two cells side by side, each
appearing exactly once in its column — one a typo, one a legitimately rare
value. The eye cannot tell. Neither can a rarity-based detector.

### Act 2 — the pipeline, one stage per step (sticky figure)
Each step changes the figure and carries one sentence of reasoning:

1. **Profile** — learn what normal looks like per column
2. **Detect** — cells light up, labelled with which of the six checks fired
3. **Repair** — candidate values appear with probabilities
4. **Quantify uncertainty** — raw confidence, self-consistency, entropy
5. **Calibrate** — raw confidence moves to a trustworthy `p`; show the shift
6. **Learn `q`** — P(this flagged cell is genuinely corrupted), per check
7. **Decide** — three expected-cost bars, the argmin wins, action badge

### Act 3 — the thesis, interactive (free exploration)
Cost sliders, live. Same cell, same `p`, same `q` — the action flips.
Alongside it a `(p, q)` decision map with the three regions drawn from the
derived boundaries and the current cell plotted, moving as the sliders move:

    AUTO vs HUMAN     p* = 1 - C_human / C_wrong      (vertical)
    HUMAN vs ABSTAIN  q* = C_human / C_abstain         (horizontal)
    AUTO vs ABSTAIN   p  = 1 - q * C_abstain / C_wrong (sloped)

### Act 4 — results, honestly (ground truth allowed here only)
Zero worst-case regret; what the fallback fixed (`flights` chance → 0.84);
and **what failed** — the five refuted mechanisms and the nuisance
hierarchy. Including the failures is unusual and is more persuasive to a
judge than hiding them.

## 4. Leakage rule applies to the UI

`CLAUDE.md` §6: **never display the clean value beside a dirty cell in the
decision demo.** Ground truth appears only in Act 4, which is visually and
structurally separated and labelled as evaluation.

## 5. Design rules, from the research

- **Sticky figure, stepped text.** Every step changes the figure.
- **No scroll hijacking.** Never intercept wheel or touch events.
- **Reduced motion respected** — `prefers-reduced-motion` disables
  animation and shows the end state.
- **Reserve space** for animated containers so nothing jumps on load.
- **Exits available.** A skip-to-section nav; the judge is never trapped.
- **Breathing room**: one idea per screen, generous whitespace, at most one
  table per step.
- Works at phone width.

## 6. Why a standalone page rather than extending `app.py`

`app.py` (Streamlit) stays as the live-compute tool for us. The judge-facing
demo is a published page because a judge needs a **URL that opens**, with no
Python, no install, no cold start, and no risk of a 40-second stall mid-
demonstration. The two serve different audiences and both are kept.

## 7. Build order

- [ ] Commit this plan
- [ ] `experiments/export_demo_trace.py` — run the REAL pipeline, export
      every intermediate to `results/demo_trace.json`
- [ ] Verify the trace against the committed results so no number drifts
- [ ] Build the page, publish it, hand over the URL
