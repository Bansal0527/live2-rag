"""Run 10 predefined RAG questions against the real API and Gemini services."""

import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.parsing import parse_document
from app.service import build_service


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT.parent / "samples"
DOCUMENTS = ("employment_agreement.txt", "vendor_terms.txt", "company_policy.pdf")


def score_case(case: dict, body: dict) -> bool:
    if case["expected_status"] == "insufficient_evidence":
        return (case["expected_answer"] is None and body.get("status") == "insufficient_evidence"
                and body.get("answer") is None and not body.get("sources"))
    expected = case["expected_answer"].casefold()
    answer = (body.get("answer") or "").casefold()
    sources = body.get("sources") or []
    return (
        body.get("status") == "answered"
        and all(term.casefold() in expected and term.casefold() in answer for term in case["expected_terms"])
        and set(case["expected_documents"]).issubset({source.get("document_name") for source in sources})
        and all(source.get("chunk_id", "").startswith(source.get("document_id", "") + ":") for source in sources)
    )


def valid_sources(sources: list[dict], documents: dict[str, dict]) -> bool:
    for source in sources:
        record = documents.get(source.get("document_name"))
        if not record or source.get("document_id") != record["id"]:
            return False
        if not source.get("chunk_id", "").startswith(record["id"] + ":"):
            return False
        path = SAMPLES / record["name"]
        pages = parse_document(path.read_bytes(), record["file_type"])
        if not any(page.number == source.get("page") and source.get("text", "") in page.text for page in pages):
            return False
    return True


def run() -> dict:
    questions = json.loads((ROOT / "questions.json").read_text())
    if len(questions) != 10:
        raise ValueError("Evaluation requires exactly 10 questions")
    if any("expected_answer" not in case for case in questions):
        raise ValueError("Every evaluation question needs an expected answer")
    delay = float(os.getenv("EVAL_DELAY_SECONDS", "30"))
    retry_delay = float(os.getenv("EVAL_RETRY_SECONDS", "60"))
    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "documents": [], "results": [], "passed": 0, "total": len(questions),
    }
    uploaded = {}

    def save_report():
        report["passed"] = sum(item["passed"] for item in report["results"])
        (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    with TemporaryDirectory() as directory:
        service = build_service(replace(Settings.from_env(), data_dir=Path(directory)))
        report["chat_model"] = service.settings.chat_model
        with TestClient(create_app(service)) as client:
            for name in DOCUMENTS:
                if uploaded and delay:
                    time.sleep(delay)
                for attempt in range(3):
                    with (SAMPLES / name).open("rb") as file:
                        response = client.post("/documents/upload", files={"file": (name, file)})
                    if response.status_code not in (502, 503, 504) or attempt == 2:
                        break
                    print(f"RETRY upload {name}: HTTP {response.status_code}", flush=True)
                    time.sleep(retry_delay)
                if response.status_code != 201:
                    report["error"] = f"Upload failed for {name}: HTTP {response.status_code} {response.text[:200]}"
                    save_report()
                    break
                record = response.json()
                uploaded[name] = record
                report["documents"].append(record)
                print(f"UPLOADED {name}: {record['status']}", flush=True)
                save_report()

            if not report.get("error"):
                for case in questions:
                    if delay:
                        time.sleep(delay)
                    for attempt in range(3):
                        response = client.post("/ask", json={"query": case["query"]})
                        if response.status_code not in (502, 503, 504) or attempt == 2:
                            break
                        print(f"RETRY {case['id']}: HTTP {response.status_code}", flush=True)
                        time.sleep(retry_delay)
                    body = response.json()
                    source_check = valid_sources(body.get("sources", []), uploaded)
                    passed = response.status_code == 200 and score_case(case, body) and source_check
                    report["results"].append({
                        "id": case["id"], "query": case["query"], "passed": passed,
                        "expected_answer": case["expected_answer"],
                        "expected_status": case["expected_status"],
                        "expected_documents": case["expected_documents"],
                        "http_status": response.status_code,
                        "source_references_valid": source_check,
                        "response": body,
                    })
                    print(f"{'PASS' if passed else 'FAIL'} {case['id']}: {body.get('status', response.status_code)}", flush=True)
                    save_report()

    save_report()
    return report


if __name__ == "__main__":
    report = run()
    print(f"{report['passed']}/{report['total']} questions passed")
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
