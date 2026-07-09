# Local SLM Feasibility for Spec Extraction — Preliminary Finding (Inconclusive)

**Goal:** determine whether a local small language model (via Ollama, CPU inference) could replace Groq for `extract_spec()`-style spec extraction, removing the Groq daily-token-quota bottleneck.

**Status: unresolved.** The result below rules out one hypothesis (prompt overload) but does not establish anything about small-model viability in general. It points at this machine's local CPU inference path, not at SLM capability.

---

## What was attempted

Ran spec extraction locally via Ollama on CPU, 8GB RAM, no GPU offload (Vulkan GPU path failed separately on out-of-device-memory and was disabled):

1. **`llama3.2:3b`** — sent the real, unmodified 54-category `_EXTRACTION_PROMPT` used with Groq.
2. **`llama3.2:1b`** — same prompt, run on 15 questions stratified by tier (simple/medium/complex), in both raw and `format:"json"` mode.
3. **`llama3.2:1b`, isolation probe** — a purpose-built ~133-token prompt, single category (`lookup_condition` only), 2 few-shot examples, tested on 5 held-out questions — designed specifically to rule out "the 54-category prompt is too long for a 1B model" as the explanation.

## What happened

- **3B, full prompt:** loaded successfully (after freeing system RAM) but produced incoherent multilingual noise (e.g. mixed English/Cyrillic/Japanese/Arabic token fragments) — not a wrong answer, not parseable as an attempt at the task.
- **1B, full prompt:** 0/30 correct specs (15 questions × raw/json mode). Most outputs were not valid JSON at all; where JSON parsed, required fields were missing.
- **1B, short single-category prompt:** 0/5 exact `patient_id` matches, 1/5 correct wave. Critically, the failures were not clean wrong answers:
  - 3 of 5 produced no `patient_id` key at all (structurally malformed JSON: stray braces, empty keys).
  - The other 2 produced a **corrupted** id, not a wrong-but-valid one — e.g. gold `b5bbfa23` → got `b55d7f2234` (56% character similarity, wrong length); gold `80d81807` → got `80d818818` (71% similarity, wrong length). JSON keys were themselves duplicated mid-generation (`"categorycategory"`, `"patient_id_id"`).

## Diagnostic reasoning

- The short-prompt run (133 tokens, 1 category, 2 examples) is close to the minimum realistic size for this task. If the earlier 0/30 were explained by prompt overload (54 categories, ~3,300 tokens exceeding a 1B model's effective context), this short prompt should have at least allowed clean copying of an 8-character ID. It did not.
- The failure signature — duplicated substrings in both JSON keys (`categorycategory`, `patient_id_id`, `lookup_condition_condition`) and in copied values (character-scrambled IDs of the wrong length, still recognizably derived from the correct source string) — is the same class of defect seen in the 3B model's incoherent output, just less severe. Two different model sizes exhibiting the same duplication/corruption signature, on the same prompt-length-independent basis, is most consistent with a shared cause — likely the local CPU inference path on this machine — though a corrupted model blob or an Ollama 0.31.1 build issue are not yet ruled out.

## Conclusion

**Local CPU inference on this hardware is unreliable for this task and cannot be used to draw any conclusion about small-model suitability.** The prompt-overload hypothesis is ruled out; a backend/inference-path defect is the leading explanation, but this has not been isolated further (e.g. not yet tested against a different backend build, different quantization, or GPU compute). SLM feasibility for spec extraction remains **untested in any valid sense** — this result is a dead end on this hardware, not a negative result on small models.

**Next step:** re-run both the 3B and 1B probes on a GPU environment (Colab/Kaggle) before making any claim about local-model viability. A clean re-pull of `llama3.2:3b` is also deferred to that environment, to rule out a corrupted local blob as a contributing factor.

This finding is specific to this machine and software version; it is not evidence about Ollama, llama3.2, or small models generally.

**Groq remains the extraction backend for now.** No changes were made to `text_to_pandas_fix.py` or the extraction pipeline as part of this investigation.
