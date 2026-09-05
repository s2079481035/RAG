"""Local browser UI for reviewing the manual sufficiency audit CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "docs" / "manual_sufficiency_audit.csv"
LABELS = {"insufficient", "partial", "sufficient"}
HUMAN_FIELDS = {
    "human_sufficient_to_answer",
    "human_label",
    "human_notes",
    "reviewer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def read_audit(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows or not fields:
        raise ValueError("Manual audit CSV is empty")
    required = {"audit_id", "automatic_three_class_label", *HUMAN_FIELDS}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Manual audit CSV is missing fields: {sorted(missing)}")
    return fields, rows


def annotation_updates(row: dict[str, str], payload: dict) -> dict[str, str]:
    label = str(payload.get("human_label", "")).strip().lower()
    notes = str(payload.get("human_notes", "")).strip()
    reviewer = str(payload.get("reviewer", "")).strip()
    if label not in LABELS:
        raise ValueError("human_label must be insufficient, partial, or sufficient")
    if not reviewer:
        raise ValueError("reviewer is required")
    automatic = row["automatic_three_class_label"].strip().lower()
    if label != automatic and not notes:
        raise ValueError("human_notes is required when the human and automatic labels disagree")
    return {
        "human_sufficient_to_answer": "yes" if label == "sufficient" else "no",
        "human_label": label,
        "human_notes": notes,
        "reviewer": reviewer,
    }


def save_annotation(path: Path, audit_id: str, payload: dict) -> dict[str, str]:
    fields, rows = read_audit(path)
    matches = [row for row in rows if row["audit_id"] == audit_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for audit_id={audit_id!r}, found {len(matches)}")
    row = matches[0]
    updates = annotation_updates(row, payload)
    for field, value in updates.items():
        row[field] = value

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return row


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sufficiency Audit Review</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #18212b;
      --muted: #66717e;
      --line: #d7dce2;
      --surface: #ffffff;
      --canvas: #f3f5f7;
      --nav: #202b38;
      --accent: #087f8c;
      --warn: #9a5b00;
      --danger: #a33a3a;
      --success: #247247;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: var(--canvas); font: 14px/1.5 system-ui, sans-serif; }
    button, input, textarea { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    header {
      position: sticky; top: 0; z-index: 4; display: flex; align-items: center; gap: 16px;
      min-height: 58px; padding: 9px 18px; color: #fff; background: var(--nav); border-bottom: 1px solid #101820;
    }
    header h1 { margin: 0; font-size: 17px; font-weight: 650; }
    #progress { color: #cdd6df; white-space: nowrap; }
    .filters { display: flex; border: 1px solid #566170; border-radius: 6px; overflow: hidden; }
    .filters button { border: 0; border-right: 1px solid #566170; padding: 6px 10px; color: #e7edf3; background: transparent; }
    .filters button:last-child { border-right: 0; }
    .filters button.active { color: #102029; background: #dceef0; }
    .reviewer { margin-left: auto; display: flex; align-items: center; gap: 7px; color: #cdd6df; }
    .reviewer input { width: 130px; border: 1px solid #687583; border-radius: 4px; padding: 6px 8px; color: #fff; background: #111b25; }
    .layout { display: grid; grid-template-columns: 250px minmax(0, 1fr) 290px; min-height: calc(100vh - 58px); }
    aside { background: var(--surface); border-right: 1px solid var(--line); }
    .search { position: sticky; top: 58px; z-index: 2; padding: 10px; background: var(--surface); border-bottom: 1px solid var(--line); }
    .search input { width: 100%; border: 1px solid #aeb7c1; border-radius: 4px; padding: 8px; }
    #row-list { max-height: calc(100vh - 109px); overflow: auto; }
    .row-link { display: grid; grid-template-columns: 42px 1fr 12px; gap: 7px; width: 100%; padding: 9px 10px; text-align: left; border: 0; border-bottom: 1px solid #edf0f2; background: #fff; }
    .row-link:hover { background: #eef6f7; }
    .row-link.active { background: #dceef0; box-shadow: inset 3px 0 var(--accent); }
    .row-id { font-weight: 700; }
    .row-stage { overflow: hidden; color: var(--muted); text-overflow: ellipsis; white-space: nowrap; }
    .status-dot { width: 9px; height: 9px; margin-top: 6px; border-radius: 50%; background: #c4c9cf; }
    .status-dot.done { background: var(--success); }
    .status-dot.disagree { background: var(--warn); }
    main { min-width: 0; padding: 22px 28px 60px; }
    .meta { color: var(--muted); font-size: 13px; }
    h2 { margin: 5px 0 10px; font-size: 22px; line-height: 1.3; letter-spacing: 0; }
    h3 { margin: 24px 0 8px; font-size: 14px; text-transform: uppercase; color: #46525f; letter-spacing: 0; }
    .answer { display: inline-block; padding: 5px 9px; border-left: 4px solid var(--success); background: #eaf5ee; font-size: 16px; font-weight: 650; }
    .facts { width: 100%; border-collapse: collapse; background: #fff; }
    .facts th, .facts td { padding: 7px 9px; text-align: left; border: 1px solid var(--line); }
    .facts th { color: #4d5864; background: #edf0f3; }
    .evidence { margin: 0 0 10px; border: 1px solid var(--line); border-radius: 6px; background: #fff; overflow: hidden; }
    .evidence-title { padding: 7px 10px; color: #33404d; background: #edf0f3; border-bottom: 1px solid var(--line); font-size: 12px; font-weight: 700; }
    .evidence-text { margin: 0; padding: 11px 12px; white-space: pre-wrap; overflow-wrap: anywhere; font: 13px/1.58 system-ui, sans-serif; }
    .decision { position: sticky; top: 58px; align-self: start; height: calc(100vh - 58px); overflow: auto; padding: 18px; background: #fff; border-left: 1px solid var(--line); }
    .decision h3 { margin-top: 0; }
    .label-options { display: grid; gap: 8px; }
    .label-button { width: 100%; min-height: 42px; border: 1px solid #aeb7c1; border-radius: 6px; background: #fff; font-weight: 650; }
    .label-button[data-label="insufficient"].selected { color: #fff; border-color: #59636f; background: #59636f; }
    .label-button[data-label="partial"].selected { color: #fff; border-color: var(--warn); background: var(--warn); }
    .label-button[data-label="sufficient"].selected { color: #fff; border-color: var(--success); background: var(--success); }
    label.field { display: block; margin-top: 16px; color: #4d5864; font-weight: 650; }
    textarea { width: 100%; min-height: 105px; margin-top: 5px; resize: vertical; border: 1px solid #aeb7c1; border-radius: 4px; padding: 8px; }
    details { margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--line); }
    summary { cursor: pointer; color: #46525f; font-weight: 650; }
    .diagnostics { margin-top: 10px; color: #4c5865; overflow-wrap: anywhere; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 18px; }
    .actions button { min-height: 39px; border: 1px solid #9ca6b0; border-radius: 5px; background: #fff; }
    .actions .primary { color: #fff; border-color: var(--accent); background: var(--accent); }
    .navigation { display: flex; gap: 8px; margin-top: 10px; }
    .navigation button { flex: 1; min-height: 36px; border: 1px solid #aeb7c1; border-radius: 5px; background: #f7f8f9; }
    #message { min-height: 22px; margin-top: 10px; color: var(--muted); }
    #message.error { color: var(--danger); }
    @media (max-width: 1050px) {
      .layout { grid-template-columns: 190px minmax(0, 1fr); }
      .decision { position: static; grid-column: 2; height: auto; border-top: 1px solid var(--line); border-left: 0; }
    }
    @media (max-width: 720px) {
      header { align-items: flex-start; flex-wrap: wrap; }
      .reviewer { width: 100%; margin-left: 0; }
      .layout { display: block; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      #row-list { display: flex; max-height: 110px; overflow: auto; }
      .row-link { min-width: 145px; border-right: 1px solid var(--line); }
      main { padding: 18px 14px 35px; }
      .decision { grid-column: auto; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Sufficiency Audit</h1>
    <span id="progress"></span>
    <div class="filters" id="filters">
      <button data-filter="incomplete" class="active">Incomplete</button>
      <button data-filter="all">All</button>
      <button data-filter="completed">Completed</button>
      <button data-filter="disagreement">Disagreement</button>
    </div>
    <label class="reviewer">Reviewer <input id="reviewer" autocomplete="off"></label>
  </header>
  <div class="layout">
    <aside>
      <div class="search"><input id="search" type="search" placeholder="Search ID or question"></div>
      <div id="row-list"></div>
    </aside>
    <main id="record"></main>
    <section class="decision">
      <h3>Human label</h3>
      <div class="label-options">
        <button class="label-button" data-label="insufficient" title="No meaningful answer chain">Insufficient</button>
        <button class="label-button" data-label="partial" title="Relevant evidence exists, but a required fact is missing">Partial</button>
        <button class="label-button" data-label="sufficient" title="The complete answer follows from the retrieved evidence">Sufficient</button>
      </div>
      <label class="field">Notes<textarea id="notes"></textarea></label>
      <details>
        <summary>Automatic diagnostics</summary>
        <div class="diagnostics" id="diagnostics"></div>
      </details>
      <div class="actions">
        <button id="save">Save</button>
        <button id="save-next" class="primary">Save and next</button>
      </div>
      <div class="navigation">
        <button id="previous">Previous</button>
        <button id="next">Next</button>
      </div>
      <div id="message"></div>
    </section>
  </div>
  <script>
    const state = { rows: [], currentId: null, filter: "incomplete", query: "", selectedLabel: "" };
    const byId = id => document.getElementById(id);
    const text = (tag, value, className) => {
      const node = document.createElement(tag);
      node.textContent = value ?? "";
      if (className) node.className = className;
      return node;
    };
    const completed = row => Boolean(row.human_label && row.reviewer && row.human_sufficient_to_answer);
    const disagreement = row => completed(row) && row.human_label !== row.automatic_three_class_label;

    function visibleRows() {
      const query = state.query.toLowerCase();
      return state.rows.filter(row => {
        const matchesText = !query || row.audit_id.includes(query) || row.question.toLowerCase().includes(query);
        const matchesFilter = state.filter === "all"
          || (state.filter === "incomplete" && !completed(row))
          || (state.filter === "completed" && completed(row))
          || (state.filter === "disagreement" && disagreement(row));
        return matchesText && matchesFilter;
      });
    }

    function updateProgress() {
      const done = state.rows.filter(completed).length;
      const disagreed = state.rows.filter(disagreement).length;
      byId("progress").textContent = `${done} / ${state.rows.length} completed, ${disagreed} disagreements`;
    }

    function renderList() {
      const container = byId("row-list");
      container.replaceChildren();
      const rows = visibleRows();
      for (const row of rows) {
        const button = document.createElement("button");
        button.className = "row-link" + (row.audit_id === state.currentId ? " active" : "");
        button.append(text("span", row.audit_id, "row-id"));
        button.append(text("span", `${row.stage}  ${row.question}`, "row-stage"));
        const dot = text("span", "", "status-dot");
        if (completed(row)) dot.classList.add(disagreement(row) ? "disagree" : "done");
        button.append(dot);
        button.addEventListener("click", () => selectRow(row.audit_id));
        container.append(button);
      }
      if (!rows.length) container.append(text("p", "No matching records.", "meta"));
    }

    function parseFacts(raw) {
      try { return JSON.parse(raw); } catch (_) { return []; }
    }

    function renderFacts(container, raw) {
      const facts = parseFacts(raw);
      const table = document.createElement("table");
      table.className = "facts";
      const head = document.createElement("tr");
      head.append(text("th", "Title"));
      head.append(text("th", "Sentence ID"));
      table.append(head);
      for (const fact of facts) {
        const tr = document.createElement("tr");
        tr.append(text("td", fact.title));
        tr.append(text("td", String(fact.sentence_id)));
        table.append(tr);
      }
      container.append(table);
    }

    function renderEvidence(container, raw) {
      const blocks = raw.split(/\n\s*\n(?=\[article_)/).filter(Boolean);
      for (const block of blocks) {
        const marker = " | TEXT: ";
        const at = block.indexOf(marker);
        const article = document.createElement("article");
        article.className = "evidence";
        article.append(text("div", at >= 0 ? block.slice(0, at) : "Retrieved evidence", "evidence-title"));
        article.append(text("pre", at >= 0 ? block.slice(at + marker.length) : block, "evidence-text"));
        container.append(article);
      }
    }

    function selectRow(auditId) {
      state.currentId = auditId;
      const row = state.rows.find(item => item.audit_id === auditId);
      if (!row) return;
      state.selectedLabel = row.human_label || "";
      byId("notes").value = row.human_notes || "";
      if (row.reviewer && !byId("reviewer").value) byId("reviewer").value = row.reviewer;
      document.querySelectorAll(".label-button").forEach(button => {
        button.classList.toggle("selected", button.dataset.label === state.selectedLabel);
      });
      const record = byId("record");
      record.replaceChildren();
      record.append(text("div", `Audit ${row.audit_id} | ${row.question_id} | ${row.stage}`, "meta"));
      record.append(text("h2", row.question));
      record.append(text("h3", "Gold answer"));
      record.append(text("div", row.gold_answer, "answer"));
      record.append(text("h3", "Gold supporting facts"));
      renderFacts(record, row.gold_supporting_facts);
      record.append(text("h3", "Full retrieved evidence"));
      renderEvidence(record, row.retrieved_evidence);
      byId("diagnostics").replaceChildren(
        text("div", `Automatic label: ${row.automatic_three_class_label}`),
        text("div", `Automatic stop: ${row.automatic_stop_label}`),
        text("div", `Critic stop probability: ${row.critic_stop_probability}`),
        text("div", `Covered gold facts: ${row.covered_supporting_facts}`),
        text("div", `Model-visible gold ratio: ${row.visible_supporting_fact_ratio_evaluation_only}`),
        text("div", `Evidence tokens: ${row.evidence_length_tokens}`)
      );
      byId("message").textContent = "";
      byId("message").className = "";
      renderList();
    }

    function move(delta) {
      const rows = visibleRows();
      if (!rows.length) return;
      let index = rows.findIndex(row => row.audit_id === state.currentId);
      if (index < 0) index = delta > 0 ? -1 : 0;
      const next = Math.max(0, Math.min(rows.length - 1, index + delta));
      selectRow(rows[next].audit_id);
    }

    async function save(goNext) {
      const row = state.rows.find(item => item.audit_id === state.currentId);
      const message = byId("message");
      message.className = "";
      if (!row || !state.selectedLabel) {
        message.textContent = "Select a human label.";
        message.className = "error";
        return;
      }
      const payload = {
        audit_id: row.audit_id,
        human_label: state.selectedLabel,
        human_notes: byId("notes").value,
        reviewer: byId("reviewer").value
      };
      localStorage.setItem("phase3a-reviewer", payload.reviewer);
      const response = await fetch("/api/annotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) {
        message.textContent = result.error || "Save failed.";
        message.className = "error";
        return;
      }
      const index = state.rows.findIndex(item => item.audit_id === row.audit_id);
      state.rows[index] = result.row;
      updateProgress();
      renderList();
      message.textContent = "Saved.";
      if (goNext) {
        const next = state.rows.slice(index + 1).find(item => !completed(item))
          || state.rows.slice(0, index).find(item => !completed(item));
        if (next) selectRow(next.audit_id);
      }
    }

    document.querySelectorAll(".label-button").forEach(button => {
      button.addEventListener("click", () => {
        state.selectedLabel = button.dataset.label;
        document.querySelectorAll(".label-button").forEach(item => {
          item.classList.toggle("selected", item === button);
        });
      });
    });
    byId("filters").addEventListener("click", event => {
      if (!event.target.dataset.filter) return;
      state.filter = event.target.dataset.filter;
      document.querySelectorAll("#filters button").forEach(button => {
        button.classList.toggle("active", button === event.target);
      });
      renderList();
      const rows = visibleRows();
      if (rows.length && !rows.some(row => row.audit_id === state.currentId)) selectRow(rows[0].audit_id);
    });
    byId("search").addEventListener("input", event => { state.query = event.target.value; renderList(); });
    byId("save").addEventListener("click", () => save(false));
    byId("save-next").addEventListener("click", () => save(true));
    byId("previous").addEventListener("click", () => move(-1));
    byId("next").addEventListener("click", () => move(1));

    fetch("/api/rows").then(response => response.json()).then(data => {
      state.rows = data.rows;
      byId("reviewer").value = localStorage.getItem("phase3a-reviewer") || "";
      updateProgress();
      renderList();
      const first = visibleRows()[0] || state.rows[0];
      if (first) selectRow(first.audit_id);
    }).catch(error => {
      byId("record").append(text("p", `Unable to load audit: ${error}`, "meta"));
    });
  </script>
</body>
</html>
"""


class AuditHandler(BaseHTTPRequestHandler):
    server_version = "Phase3AAudit/1.0"

    def log_message(self, format: str, *args) -> None:
        return

    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_bytes(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_bytes(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/rows":
            with self.server.audit_lock:
                _, rows = read_audit(self.server.audit_path)
            self.send_json(200, {"rows": rows})
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/annotation":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            audit_id = str(payload.get("audit_id", "")).strip()
            if not audit_id:
                raise ValueError("audit_id is required")
            with self.server.audit_lock:
                row = save_annotation(self.server.audit_path, audit_id, payload)
            self.send_json(200, {"row": row})
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except OSError as error:
            self.send_json(500, {"error": f"Unable to save CSV: {error}"})


def main() -> None:
    args = parse_args()
    audit_path = args.input.resolve()
    read_audit(audit_path)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The review server only binds to a loopback address")
    server = ThreadingHTTPServer((args.host, args.port), AuditHandler)
    server.audit_path = audit_path
    server.audit_lock = threading.Lock()
    print(f"Reviewing: {audit_path}")
    print(f"Open through SSH port forwarding: http://127.0.0.1:{args.port}")
    print("Press Ctrl-C to stop. Saved annotations remain in the CSV.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
