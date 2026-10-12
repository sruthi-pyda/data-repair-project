# Phase 12 — real LLM repair

Status: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` finding

Branch: `feat/phase12-llm` (off `feat/phase13-experiments`).
One commit per step. Full suite green at every commit.

---

## 0. Why, in fundamentals

The project is called "Risk-Aware **LLM-Assisted** Data Repair" and there is
currently not one real model call in it. Every number so far bounds the
*decision layer* on top of a stand-in repairer (`mock_repair.py`:
difflib similarity + frequency + median imputation + seeded noise).

Phase 12 does **not** change the thesis. The claim is about calibration and
cost-sensitive decisions, and that machinery is repairer-agnostic by design.
What Phase 12 changes is what `p` is calibrated *from*. Two things make it
worth doing:

1. The title is otherwise unearned.
2. Mock `self_consistency` carries no signal (CLAUDE.md §5 — it gets ~0 weight
   in the multivariate calibrator). A real LLM sampled at temperature > 0 is
   the only way to find out whether that is a property of the mock or of the
   signal. That is a genuine open question the ablation can then answer.

### Provider choice — RESOLVED: Groq, not xAI/Grok

The keys supplied are **Groq** (`gsk_…`, api.groq.com), an inference provider
serving open models — not xAI's Grok. Groq's API is OpenAI-compatible so the
client is unchanged; only the base URL differs. Verified live 2026-10-12: all
five keys valid, 11 models exposed, of which the usable ones for repair are
`openai/gpt-oss-120b` (default), `openai/gpt-oss-20b` and `qwen/qwen3.8-27b`.
The rest are speech (Whisper), TTS (Orpheus) or safety classifiers.

**Finding: gpt-oss-120b is a reasoning model.** It spends 200–500 completion
tokens on hidden reasoning before emitting JSON, so a tight `max_tokens`
truncates the structure and the response is rejected as malformed. Measured on
12 breast_cancer cells: 512 → 17% fallback, 1536 → 8%. Default raised to 1536.

### Provider choice (original)

An **OpenAI-compatible** client, configured by environment variable, not a
hard-wired vendor:

```
LLM_API_KEY     = ...
LLM_BASE_URL    = https://api.x.ai/v1      # xAI/Grok; omit for OpenAI
LLM_MODEL       = grok-...                 # exact id from the user's console
```

xAI's API is OpenAI-compatible, so the same client reaches Grok, OpenAI, or
any compatible endpoint. Switching providers (or adding a second model for a
reviewer) is then config, not code — a better story for the paper than
hard-wiring one vendor.

**The user holds the key.** It goes in a local `.env` that is gitignored. No
key is ever read, printed, logged or committed by this work. Tests never make
a live call.

### Leakage rules (CLAUDE.md §3, non-negotiable)

Send **only**: dataset/domain, column name, dtype/kind, the dirty value, the
nearby row context, representative valid values, schema constraints. Those are
exactly the fields already on `RepairContext`, which is built from dirty data
alone.

**Never** send: the clean value, the full dataset, any ground-truth label, or
any other row's clean data. A test must assert the outgoing payload contains
no clean value — not by inspecting the prompt builder's intent, but by
scanning the actual request body captured from a mocked transport.

---

## 1. Design

`LLMRepairer` is a **drop-in for `MockRepairer`**: same `repair(context)` and
`repair_batch(records, dirty_df, profile)` signatures, same `RepairResult`
output. `run_pipeline(..., mode="mock"|"real")` then selects between them and
nothing downstream changes.

```python
@dataclass
class LLMConfig:
    api_key: Optional[str]; base_url: Optional[str]; model: str
    temperature: float = 0.0; max_tokens: int = 512
    timeout_s: float = 30.0; max_retries: int = 2
    @classmethod
    def from_env(cls, dotenv_path=".env") -> "LLMConfig": ...
    @property
    def is_configured(self) -> bool: ...        # key present and non-placeholder

class LLMRepairer:
    def __init__(self, config=None, fallback=None, top_k=3, seed=42,
                 cache=None, client=None): ...  # client injectable for tests
    def repair(self, context) -> RepairResult
    def repair_batch(self, records, dirty_df, profile) -> List[RepairResult]
    @property
    def stats(self) -> LLMStats    # calls, cache hits, fallbacks by reason

class FallbackReason(str, Enum):
    NOT_CONFIGURED, REQUEST_FAILED, MALFORMED_RESPONSE,
    INVALID_PROBABILITIES, EMPTY_CANDIDATES, TIMEOUT
```

**Fallback is per cell, never fatal.** Any failure logs the reason, increments
a counter, and falls back to `MockRepairer` for that cell. `RepairResult`
carries `repair_method="llm"` or `"llm→mock (<reason>)"` so a fallback is
visible in the predictions CSV rather than silently indistinguishable.

**Determinism and cost.** `temperature=0.0` by default. An on-disk cache keyed
by `sha256(model, prompt)` means a re-run costs nothing and is reproducible;
`--no-cache` forces fresh calls. Self-consistency sampling (temperature > 0,
n samples) is **opt-in** via `UncertaintyEstimator`, because it multiplies
cost by n.

**Validation** (reject → fallback): non-JSON; missing `candidates`; empty
list; a probability that is NaN/negative/>1; probabilities that do not sum to
≈1 after normalisation; duplicate candidate values; more than `top_k`;
a value that cannot be cast to the column's dtype. `prompts.parse_llm_response`
already does some of this — extend rather than duplicate, and make it return
*why* it failed instead of a bare empty list.

---

## 2. Steps

### Step 1 — config, client, repairer (`src/repair/llm_repair.py`)
- [x] 1.1 `LLMConfig.from_env` + `.env.example` rewritten for `LLM_*` names
      (keep `MOCK_MODE`; drop the stale `OPENAI_*` defaults). `.env` is
      already gitignored — verify.
- [x] 1.2 Transport: thin wrapper over the `openai` SDK if installed, else
      `urllib` POST to `{base_url}/chat/completions`. Injectable `client` so
      tests never touch the network. Retries with backoff on 429/5xx only.
- [x] 1.3 `LLMRepairer` with per-cell fallback, `FallbackReason`, `LLMStats`,
      and the disk cache.
- [x] 1.4 Extend `prompts.parse_llm_response` to report a reason (keep the
      existing signature working for existing tests).
- [x] 1.5 Tests (`tests/test_llm_repair.py`), all with a fake client:
      happy path; each `FallbackReason` individually; probabilities
      renormalised; duplicates collapsed; dtype casting; cache hit avoids a
      second call; retry on 429 then success; no retry on 400; stats counts;
      `repair_batch` ordering matches `MockRepairer`'s; **leakage: the
      captured request body contains no clean value and no second row's
      data**; no test hits the network (assert the fake was used).
- [x] 1.6 Commit. (1514 tests.)

### Step 2 — wire into the pipeline and CLI
- [x] 2.1 `run_pipeline(..., mode="mock")`; `"real"` builds an `LLMRepairer`.
      Bundle records `repair_mode` and `llm_stats`.
- [x] 2.2 `run_experiment.py --mode real` stops exiting 2; it runs, and every
      table gains a "repair mode" column plus a fallback-count note. Still
      exits 2 with a clear message when no key is configured.
- [x] 2.3 Dashboard: the "Real LLM" sidebar option works when a key is
      present, shows the fallback count, and still says so when it falls back.
- [x] 2.4 Tests: mode plumbing with a fake client end-to-end; `--mode real`
      without a key exits 2; with a fake key + fake client produces the same
      artifact set as mock.
- [x] 2.5 Commit. (1564 tests.)

### Step 3 — run it for real (needs the user's key)
- [x] 3.1 User puts `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` in `.env`.
- [x] 3.2 Smoke: one dataset, small cap, inspect candidates and fallbacks.
- [x] 3.3 Full run; compare against mock on the same cells: repair accuracy,
      calibration quality, realised cost, and **whether self_consistency
      carries signal under a real model** (the open question from §0).
- [x] 3.4 Write results into `results/RESULTS.md` and CLAUDE.md. Report
      honestly, including if the LLM is *worse* than the mock.
- [x] 3.5 Commit.

---

## 3. Results

Live run: `--dataset breast_cancer --mode real --llm-live`, 392 cells,
openai/gpt-oss-120b via Groq, 2/392 fallbacks (0.5 %), 1,407 s.
Artifacts in `results_llm/`.

| metric | mock | real LLM |
|---|---:|---:|
| top-1 repair accuracy | 0.6732 | **0.7099** |
| proposed policy cost | **1,615** | 1,670 |
| best baseline cost | 1,720 (thr 0.70) | **1,620 (thr 0.80)** |
| always-auto cost | 5,800 | 5,150 |
| verdict | proposed WINS | **proposed LOSES by 50** |

**The LLM repairs better and the policy does worse.** Better repairs raise p,
the policy auto-repairs more (though T3 shows AUTO share *fell* to 0.1155),
and a fixed 0.80 threshold happens to land better on this distribution. On
n=392 with a 50-unit gap this is close, but it is a loss and is reported as
one. Controlled per-error-type comparison on 80 cells: LLM 50 % vs mock 45 %
on missing values (noise), LLM 92.5 % vs mock **100 %** on out-of-range — the
LLM is worse on the deterministic cases a rule nails.
