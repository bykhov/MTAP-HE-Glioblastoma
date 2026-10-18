"""GDC helpers (TSV streaming, retries, cohort copy-number tables), tested against a
small local HTTP server that imitates the parts of api.gdc.cancer.gov used. No network access."""
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import gdc_utils  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGDC:
    """files: file_id -> bytes; hits: what /files returns (filtered by cases and data_type if the query asks)."""

    def __init__(self, hits, blobs):
        self.hits, self.blobs = hits, blobs
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urlparse(self.path)
                outer.requests.append(url.path)
                if url.path == "/files":
                    flt = json.loads(parse_qs(url.query)["filters"][0])
                    data = [h for h in outer.hits if outer.matches(h, flt)]
                    body = json.dumps({"data": {"hits": data, "pagination": {"total": len(data)}}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = outer.blobs[url.path.rsplit("/", 1)[1]]
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def matches(hit, flt):
        parts = flt["content"] if flt["op"] == "and" else [flt]
        for p in parts:
            field, value = p["content"]["field"], p["content"]["value"]
            if field == "cases.submitter_id" and not {c["submitter_id"] for c in hit["cases"]} & set(value):
                return False
            if field == "data_type" and hit["data_type"] != value:
                return False
            if field == "experimental_strategy" and hit.get("experimental_strategy") != value:
                return False
        return True

    def close(self):
        self.server.shutdown()


def test_stream_tsv_row_finds_the_gene_and_stops_early():
    header = "gene_id\tgene_name\tchromosome\tstart\tend\tcopy_number\tmin_copy_number\tmax_copy_number\n"
    rows = "".join(f"G{i}\tGENE{i}\tchr1\t1\t2\t2\t2\t2\n" for i in range(50)) + "G99\tMTAP\tchr9\t1\t2\t0\t0\t1\n"
    tsv = (header + rows).encode()
    fake = FakeGDC([], {"t1": tsv})
    try:
        row = gdc_utils.stream_tsv_row("t1", "gene_name", "MTAP", api=fake.url)
        assert row["copy_number"] == "0" and row["max_copy_number"] == "1"
        assert gdc_utils.stream_tsv_row("t1", "gene_name", "NOPE", api=fake.url) is None
    finally:
        fake.close()


def test_with_retries_recovers_from_network_errors_and_gives_up(monkeypatch):
    monkeypatch.setattr(gdc_utils.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise gdc_utils.requests.ConnectionError("reset")
        return "ok"

    assert gdc_utils.with_retries(flaky, log=lambda *a: None) == "ok" and calls["n"] == 3
    with pytest.raises(gdc_utils.requests.ConnectionError):
        gdc_utils.with_retries(lambda: (_ for _ in ()).throw(gdc_utils.requests.ConnectionError("down")), attempts=2, log=lambda *a: None)


def test_cohort_flow_tables_and_disagreements():
    cc = load("cohort_copy_number")
    per_file = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4", "P5", "P6", "P6"], "file_id": list("abcdefg"),
        "copy_number": [0, 2, 2, 1, 0, 2, 0], "min_copy_number": [0, 1, 2, 0, 0, 2, 0]})
    slide_table = pd.DataFrame({"case_id": ["P1", "P2", "P3", "P4", "P5"], "is_mtap_mutation": [1, 0, 0, 0, 1]})
    cohort = pd.DataFrame({"patient_id": ["P1", "P2", "P3"], "label": [1, 0, 1]})  # P3 is mislabeled on purpose
    cases, flow = cc.build_tables(per_file, {"P1", "P2", "P3", "P4"}, slide_table, cohort)
    assert flow["steps"][0]["patients"] == 6                       # P1..P6 have a copy-number file
    assert flow["steps"][1]["copy_number_0_HD"] == 2 and flow["steps"][1]["copy_number_2_neutral"] == 3
    assert flow["steps"][2]["patients"] == 3                       # P1, P2, P3 have a DX slide and copy number 0 or 2
    assert flow["steps"][3]["patients"] == 5 and flow["steps"][4]["patients"] == 3
    out = flow["slide_table_patients_outside_the_cohort"]
    assert out["n"] == 2 and out["by_mtap_copy_number"] == {"0": 1, "1": 1} and out["copy_number_0_HD"] == ["P5"]
    assert flow["cohort_label_disagreements"]["n"] == 1 and flow["cohort_label_disagreements"]["patients"][0]["patient_id"] == "P3"
    assert flow["cases_with_diagnostic_slide_and_cn_0_or_2_not_in_slide_table"] == 0
    assert flow["cohort_patients_with_ambiguous_copy_number"] == []
    assert bool(cases.loc["P6", "ambiguous_copy_number"]) is True   # two files, copy numbers 2 and 0
