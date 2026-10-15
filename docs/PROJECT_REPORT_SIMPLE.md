# The whole project, explained simply

This is the plain-English version of `docs/PROJECT_REPORT.md`. Same facts,
same numbers, no jargon.

---

## 1. What we are trying to do

Imagine a big spreadsheet. Someone has secretly messed up some of the boxes.

Our job is to look at each box and choose one of three things:

1. **Fix it** — the computer changes it by itself
2. **Ask a person** — send it to a human to look at
3. **Leave it** — do nothing

Choosing well is the whole point.

We test on three spreadsheets:

| Spreadsheet | How big | How many boxes are broken |
|---|---|---|
| Breast Cancer | small (699 rows) | 631 boxes (about 1 in 13) |
| Smart Factory | medium (23,645 rows) | 7,093 boxes (about 1 in 63) |
| Adult | big (45,222 rows) | 1,701 boxes (about 1 in 400) |

---

## 2. The most important thing we learned (we learned it first)

There are **two ways** a box can be broken.

**Smudged.** Like a word with coffee spilled on it. You can still work out
what it said.

**Erased.** Like a word rubbed out with a rubber. It is gone.

We measured how often the computer fixes each kind correctly:

| Kind of broken | How often the fix is right |
|---|---|
| Number is way too big (smudged) | **98 out of 100** |
| Weird squiggle instead of a number (smudged) | **100 out of 100** |
| Box is empty (erased) — Breast Cancer | 50 out of 100 |
| Box is empty (erased) — Smart Factory | 29 out of 100 |

**You cannot un-rub-out a word.** Nothing can. Not a clever program, not a
big AI.

So "how good are our fixes?" mostly tells you **what kind of broken that
spreadsheet has**, not how clever we are. That is why the interesting part is
not fixing. It is **knowing when not to trust a fix.**

---

## 3. What we built first

A chain of steps:

```
spreadsheet
  → look at it and learn what normal looks like
  → spot boxes that look wrong
  → guess what they should say
  → work out how sure we are
  → turn "how sure" into a real honest number
  → work out the cost of each choice
  → pick the cheapest choice
  → check our answers and show them
```

### How it picks

For each box it works out three costs and picks the smallest:

- **Fix it myself** costs: (chance I am wrong) × (how bad a wrong fix is)
- **Ask a person** costs: a flat amount
- **Leave it** costs: (chance it is really broken) × (how bad leaving it is)

That is it. Multiplying and comparing. A calculator could do it.

The costs are made-up points for the experiment. **Not money.**

---

## 4. Did it work? Yes

Here is the score. Lower is better. The last column is "how much worse than
the best choice for that spreadsheet" — so **0 means you were the best**.

| How you decide | Adult | Breast Cancer | Smart Factory | Worst miss |
|---|---:|---:|---:|---:|
| **Our way** | **0** | **1,615** | **14,186** | **0** |
| Always ask a person | 11,658 | 1,775 | 14,186 | 11,658 |
| Fix it if you're 70% sure | 11,658 | 1,720 | 14,186 | 11,658 |
| Always leave it | 0 | 3,550 | 35,465 | 21,279 |
| Always fix it | 58,290 | 5,800 | 126,275 | 112,089 |

**Our way is the only one that is never bad.** Every simple rule is fine
somewhere and terrible somewhere else.

Being honest about each one:

- **Breast Cancer** — we genuinely win
- **Adult** — we *tie* with "always leave it". But that rule is awful
  everywhere else, and we worked it out from the data instead of being told
- **Smart Factory** — we *tie* with "always ask a person". No improvement.
  We say so

**One warning about Adult's 0.** It does not mean the spreadsheet is clean.
It means *the boxes we looked at* cost nothing. Adult has **1,701 broken
boxes we never even found.** Those are free for everybody, including us.

---

## 5. Things that did not work (we are not hiding them)

**An idea of mine was rubbish.** I built a fancier way of measuring
confidence. When we tested it properly it was no better than the simple way
on one spreadsheet, and **worse** on another. It goes in the paper as a
failed experiment.

**A number we had been repeating was fake.** We said one method beat another.
We had only tested it once. We tested it 20 times and the difference vanished.
It was just luck.

**The AI did not help.** We plugged in a real AI model. It fixed boxes a bit
better (71 out of 100 instead of 67). But our decision-maker then did a bit
*worse*, and a four-line simple rule beat us.

Why? We checked:

| Kind of broken | Simple way | AI |
|---|---|---|
| Erased boxes | 45 out of 100 | 50 out of 100 — same, both guessing |
| Smudged boxes | **100 out of 100** | **92 out of 100 — AI is worse** |

The AI overthinks the easy ones. On erased boxes nothing helps.

---

## 6. The big problem we found

The chain has three parts: **spot it → fix it → decide.**

The *deciding* part was good. The *spotting* part was broken.

On the Adult spreadsheet the spotter got a score of **0.0025**.

What does that mean? It means **exactly the same as guessing with your eyes
shut.** It found **zero** of the 1,701 broken boxes, and shouted about 5,829
boxes that were perfectly fine.

Why? Because Adult's damage is sneaky. Someone swapped a normal answer for a
different normal answer — like changing "cat" to "dog". Our spotter was
basically a spell-checker. Both are real words, so it sees nothing wrong.

So we went back and tried to fix the spotting. That is what the three
**waves** are.

---

## 7. Wave 1 — ask a friend who knows each row

**The idea.** Instead of "is this word spelled wrong?", ask **"does this
answer make sense for this particular row?"**

If everything else about a row says "cat", then a "dog" sitting there is
suspicious — even though "dog" is a perfectly fine word.

**We built in a trap for ourselves.** We also tested a dumber version that
only asks "is this answer rare?" If the clever version did not beat the dumb
one, the idea was wrong and we would know.

**It worked.**

| Spreadsheet | Old spotter | New clever one | Dumb version |
|---|---:|---:|---:|
| Adult | 0.0025 (blind) | **0.0120** | 0.0085 |
| Breast Cancer | **0.6712** | 0.6404 | 0.6325 |

And on the one column of Adult that was actually tampered with:
**0.70 for the clever one, 0.13 for the dumb one.** Five times better.

**But** — on Breast Cancer the clever one **lost** to the old spotter. There
the damage really was obvious mess, and a spell-checker is better at mess.

**So:** old spotter is good at *messy* damage, new one is good at *sneaky*
damage. Neither wins everywhere.

---

## 8. Wave 2 — use both at once

If one is good at messy and one is good at sneaky, use both.

| Spreadsheet | Old | New | **Both together** |
|---|---:|---:|---:|
| Breast Cancer | 0.6712 | 0.6361 | **0.7209** |

Better than either — and it cost **nothing**. No extra help needed.

Then we tried paying a teacher to check a few boxes for us.

**A famous result says 20 checks makes everything much better.** For us it
did **nothing**. On Adult, 20 checks found **zero** broken boxes — every
single time we tried. Broken boxes are 1 in 400 there, so 20 random picks all
landed on fine ones.

This is actually a useful discovery. Everyone else tested this on spreadsheets
where 1 in 5 boxes are broken. Nobody checked what happens at 1 in 400.

---

## 9. Wave 3 — which boxes should we ask about?

You only get a few questions. Which boxes do you point at?

**The obvious idea — ask about the most suspicious ones — failed completely.**
On Adult, asking about the 500 most suspicious boxes found **zero** broken
ones.

**What worked: spread your questions out.** A few from every column, instead
of piling them all in one place.

| How you choose what to ask about | Questions needed to get halfway |
|---|---|
| **Spread out, a few at a time** | **100** |
| Spread out, all at once | 500 |
| Ask about the most suspicious | **never works** |
| Ask at random | **never works** |

Asking **a few at a time** (ask, learn, ask again) needed **5× fewer
questions** than asking everything at once.

**A funny one.** On Breast Cancer, "ask about the most suspicious" was
*perfect* at finding broken boxes — every single box it asked about really
was broken. And it was **useless**. Why? Because to learn, you need to see
both broken *and* fine boxes. It only ever showed us broken ones. **It was
too good at its job.**

---

## 10. The last discovery (and it is the most useful one)

On Adult our whole-spreadsheet score stayed tiny: 0.0123. But when we looked
only at the one column that was tampered with, it was 0.6873 — good!

I first said the other 14 columns were "watering it down". **I was wrong.**
So I measured it properly:

| What we score | Score |
|---|---:|
| All 15 columns | 0.0123 |
| Only the tampered column | 0.6873 |
| All 15, but tell the others to stay quiet | **0.6873** |

The last two being **exactly the same** proves it. The other columns are not
watering anything down. They are **shouting over the top**.

Here is the shocking bit. We listed the 5,000 boxes our spotter was most
suspicious about. **Not one of them was in the tampered column.** They were
mostly in a column about which country someone is from.

Why? Almost everyone in that column says "United States". So someone from a
rare country looks *very* unusual. Our spotter is **right** that it is
unusual. It is also **completely fine**.

**So here is the real problem**, and it is a much better problem to have:

> Our spotter is good at finding the odd box **inside** a column.
> It has no idea **which column** is worth looking at.

Those are two different skills. We have the first. We need the second. That
is the next thing to build.

---

## 11. So, good or bad?

**Good:**

- The deciding part works, and it is the only method that is never bad on any
  of the three spreadsheets
- We took the spotting on Adult from **blind** to **actually seeing** inside
  the right column (0.13 → 0.70)
- Combining the two spotters made things better **for free**
- We now know exactly how to spend a small number of human checks
- We found out *why* things fail, not just *that* they fail

**Bad:**

- Only **one** spreadsheet really tests our idea properly
- The AI did not help, and our title says "AI-assisted"
- Even with 500 human checks we never got close to perfect
- We have never compared ourselves to other people's published work — every
  opponent we beat is one we wrote ourselves
- Two of our own ideas turned out to be worthless

**The honest summary:**

Smudged boxes get fixed by simple rules. Erased boxes cannot be fixed by
anything. So the valuable skill is **knowing which is which, and refusing to
trust a fix you should not trust.**

That is what we built. It works. Finding out that two of our ideas were
rubbish is not the project failing — it is the project working properly,
because we found out **before** writing the paper instead of after.
