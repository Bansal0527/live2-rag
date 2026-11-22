# Test results

## Automated tests

Run from the repository root after setup:

```bash
python -m pytest -q
```

Latest run on 23 November 2026: **33 passed**. The test suite covers uploads, metadata, hybrid and filtered retrieval, source citations, missing answers, malformed extraction, conflicting facts, background status and failure, agent tools, and evaluation scoring. PyMuPDF's SWIG bindings emitted five deprecation warnings; no test failed.

## Live Gemini evaluation

The 10 predefined cases are in [evaluation/questions.json](evaluation/questions.json). The [live response snapshot](evaluation/live_results_2026-11-23.json) contains the uploaded documents' full metadata and extractions, expected answers, actual API responses, and source-reference checks.

| Result | Count |
| --- | ---: |
| Documents uploaded and ready | 3 of 3 |
| Questions completed with passing answers | 7 of 7 attempted |
| Questions not run because Gemini quota was exhausted | 3 of 10 |

The run used `gemini-3.5-flash`. The completed questions covered employment notice and a paraphrase, employment start date, vendor notice and payment terms, annual leave, and leave request timing. All seven returned source references matched their uploaded documents, page numbers, chunk IDs, and source text.

Gemini returned repeated `429 RESOURCE_EXHAUSTED` responses before the leave-carryover question could complete. The cross-document comparison and missing-answer case were also not run. **This is a partial live evaluation, not a 10/10 result.** Rerun `python -m evaluation.run` when quota is available to produce a complete result in `evaluation/results.json`.
