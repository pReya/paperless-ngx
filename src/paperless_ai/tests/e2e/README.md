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
