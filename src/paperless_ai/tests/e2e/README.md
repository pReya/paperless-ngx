# End-to-end demo: existing taxonomy the neighbours don't carry

Demonstrates the problem behind
[paperless-ngx#12787](https://github.com/paperless-ngx/paperless-ngx/discussions/12787)
and what `merge_matched_candidates()` (branch `feature/classifier-taxonomy-candidates`)
changes about it.

## Scenario

A medical archive in the test database:

- 14 documents tagged **Bloodwork** (matching: Auto) and 14 tagged **Car** (Auto).
  A real `DocumentClassifier` is trained on them with scikit-learn.
- Correspondent **MedLab Diagnostics** with a literal matching rule `medlab`.
- 6 bulk-imported lab summaries that were never tagged.
- A new lab report, carrying only the inbox tag, that mentions MedLab.

Two seams are replaced, everything else is the real pipeline:

- the vector store (`retrieve_similar_nodes`) returns the 6 untagged imports
  as the nearest neighbours;
- the LLM (`AIClient.run_llm_query`) is a scripted stand-in that reads the
  candidate block from the real prompt and does what the prompt tells a model
  to do: reuse and pair an offered name that matches what it sees in the
  document, otherwise write a fresh name. The reply goes through the real
  `model_to_classification_suggestions` validation.

## What it shows

| Test                                                 | on `dev`                                                | on the branch                                 |
| ---------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------- |
| classifier-known tag missing from neighbours         | **fails** — candidates `{}`, model invents `blood test` | passes — `Bloodwork` offered and reused       |
| rule-matched correspondent missing from neighbours   | **fails** — model invents `MedLab Diagnostics GmbH`     | passes — `MedLab Diagnostics` reused          |
| candidates survive a vector-store outage             | **fails** — no candidates at all                        | passes — classifier + rule hits still offered |
| neighbour-carried tag still offered (control)        | passes                                                  | passes                                        |
| restricted user never sees hidden taxonomy (control) | passes                                                  | passes                                        |

Every failing assertion prints what the model was shown, what it answered
and what survived validation.

## Running

```bash
uv sync --frozen                                   # once
src/paperless_ai/tests/e2e/run_before_after.sh     # dev vs. current checkout
# or just the current checkout:
uv run --frozen pytest src/paperless_ai/tests/e2e -p no:randomly -n0 --no-cov
```

Needs `libmagic` on the host (`brew install libmagic` / `apt install libmagic1`).

## Against a real model

`--real-llm` drops the scripted stand-in and sends the real prompts to an
OpenAI-compatible endpoint (tool calling required), so the same assertions
become an empirical check of that model. Defaults point at the llama.cpp
router on `bb-framework` with `unsloth-Qwen3.6-35B-A3B`; override with
`--llm-endpoint` / `--llm-model`. Add `-s` to see every answer.

```bash
src/paperless_ai/tests/e2e/run_before_after.sh dev --real-llm -s
```

Observed on 2026-10-10 with Qwen3.6-35B-A3B (one run each, ~20 s per side):

- **dev** — shown `{}`, the model suggested five new tags
  (`Health, Medical, Lab Results, Blood Count, Cholesterol`, once `Blood Test`)
  next to the existing `Bloodwork`, and `MedLab Diagnostics` as a _new_
  correspondent name. 3 failed / 2 passed, as with the scripted model.
- **branch** — shown `{"tags": {"Bloodwork": 2}, "correspondents": {"MedLab Diagnostics": 1}}`,
  it paired both (`existing_ids`) and still proposed `Cholesterol`, `Lab Results`
  as new tags — reuse without losing novelty. When `Imported` was also offered it
  correctly did not pair it. 5 passed.

Caveat: for the correspondent, writing the exact existing name as a new name is
something the view layer's fuzzy name matcher would usually catch afterwards;
the visible win there is robustness, not a new outcome. For tags the invented
near-duplicates are the real #12787 symptom.
