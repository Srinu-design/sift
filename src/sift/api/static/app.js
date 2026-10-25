(function () {
  "use strict";

  // ---------------------------------------------------------------- tabs
  // The hash names a tab or an anchor inside one; the tab shown is the panel that holds it.
  let evalsLoaded = false;

  function showTab() {
    const id = decodeURIComponent(location.hash.slice(1));
    const target = id ? document.getElementById(id) : null;
    const panel = target ? target.closest("[data-tab]") : null;
    const tab = panel ? panel.dataset.tab : "about";
    for (const p of document.querySelectorAll("[data-tab]")) p.hidden = p.dataset.tab !== tab;
    for (const a of document.querySelectorAll("[data-tab-link]")) {
      if (a.dataset.tabLink === tab) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    }
    if (target && target !== panel) target.scrollIntoView();
    else window.scrollTo(0, 0);
    if (tab === "evals" && !evalsLoaded) {
      evalsLoaded = true;
      loadEvalHistory();
    }
  }

  window.addEventListener("hashchange", showTab);
  showTab();

  // ---------------------------------------------------------------- eval history
  function cell(text) {
    const td = document.createElement("td");
    td.textContent = text;
    return td;
  }

  function commitCell(commit) {
    const td = document.createElement("td");
    if (!commit) {
      td.textContent = "—";
      return td;
    }
    const a = document.createElement("a");
    a.href = "https://github.com/Srinu-design/sift/commit/" + encodeURIComponent(commit);
    a.textContent = commit;
    a.className = "mono";
    td.appendChild(a);
    return td;
  }

  function fmt(value) {
    return typeof value === "number" ? value.toFixed(3) : "—";
  }

  function runRow(run, metricNames) {
    const tr = document.createElement("tr");
    const when = String(run.run_at || "").replace("T", " ").slice(0, 16);
    tr.append(
      cell(when),
      commitCell(run.commit),
      cell(String(run.strategy || "")),
      cell(String(run.split || "")),
      cell(String(run.golden || "")),
    );
    for (const name of metricNames) tr.appendChild(cell(fmt((run.metrics || {})[name])));
    return tr;
  }

  async function loadEvalHistory() {
    const status = document.getElementById("evals-status");
    try {
      const response = await fetch("static/eval-history.json", { cache: "no-cache" });
      if (!response.ok) throw new Error("HTTP " + String(response.status));
      const data = await response.json();
      const runs = (data.runs || []).slice().reverse(); // newest first
      const retrieval = document.getElementById("retrieval-runs");
      const answers = document.getElementById("answer-runs");
      for (const run of runs) {
        if (run.kind === "retrieval") {
          retrieval.appendChild(runRow(run, ["hit@5", "hit@8", "recall@10", "mrr@10"]));
        } else if (run.kind === "answers") {
          answers.appendChild(
            runRow(run, ["refusal_recall", "refusal_precision", "citation_accuracy"]),
          );
        }
      }
      status.textContent = String(runs.length) + " runs.";
    } catch (err) {
      evalsLoaded = false; // try again next time the tab opens
      status.textContent =
        "Could not load the eval history: " + (err instanceof Error ? err.message : "error");
    }
  }

  // ---------------------------------------------------------------- demo
  const form = document.getElementById("query-form");
  const questionInput = document.getElementById("question");
  const strategySelect = document.getElementById("strategy");
  const versionSelect = document.getElementById("version");
  const askButton = document.getElementById("ask-btn");
  const resultsEl = document.getElementById("results");
  const exampleButtons = document.querySelectorAll(".example-btn");

  for (const btn of exampleButtons) {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-question") || "";
      questionInput.value = q;
      questionInput.focus();
    });
  }

  questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  function formatCitationLabel(c) {
    const parts = ["[" + String(c.n) + "] " + c.title];
    if (c.version) {
      parts.push("PostgreSQL " + String(c.version));
    }
    if (c.section) {
      parts.push(c.section);
    }
    return parts.join(" · ");
  }

  function renderAnswerWithLinks(answerText, citations) {
    const byNum = new Map();
    for (const c of citations || []) {
      byNum.set(Number(c.n), c);
    }

    const p = document.createElement("p");
    p.className = "answer-text";

    const parts = answerText.split(/(\[S\d+\])/g);
    for (const part of parts) {
      const match = /^\[S(\d+)\]$/.exec(part);
      if (match) {
        const num = Number(match[1]);
        const cit = byNum.get(num);
        if (cit && cit.url) {
          const link = document.createElement("a");
          link.href = cit.url;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.className = "inline-citation";
          link.textContent = part;
          p.appendChild(link);
          continue;
        }
      }
      p.appendChild(document.createTextNode(part));
    }
    return p;
  }

  function renderCitationList(items, headingText) {
    const wrap = document.createElement("div");
    wrap.className = "citations-block";

    const h2 = document.createElement("h2");
    h2.className = "block-heading";
    h2.textContent = headingText;
    wrap.appendChild(h2);

    const list = document.createElement("ol");
    list.className = "citation-list";
    for (const c of items) {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.href = c.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = formatCitationLabel(c);
      li.appendChild(link);
      list.appendChild(li);
    }
    wrap.appendChild(list);
    return wrap;
  }

  function renderMeta(data) {
    const timings = data.timings_ms || {};
    let totalMs = 0;
    for (const val of Object.values(timings)) {
      totalMs += Number(val) || 0;
    }
    const meta = document.createElement("p");
    meta.className = "result-meta";
    const cost = typeof data.cost_usd === "number" ? data.cost_usd.toFixed(6) : "0.000000";
    meta.textContent =
      "strategy: " +
      String(data.strategy) +
      " · total: " +
      totalMs.toFixed(1) +
      " ms · cost: $" +
      cost +
      " · request_id: " +
      String(data.request_id);
    return meta;
  }

  function renderError(message) {
    resultsEl.replaceChildren();
    const err = document.createElement("p");
    err.className = "error-box";
    err.textContent = message;
    resultsEl.appendChild(err);
  }

  function renderResult(data) {
    resultsEl.replaceChildren();

    if (data.refused) {
      const refusalBox = document.createElement("p");
      refusalBox.className = "refusal-box";
      refusalBox.textContent = "No answer: " + (data.refusal_reason || "Refused.");
      resultsEl.appendChild(refusalBox);

      if (Array.isArray(data.suggestions) && data.suggestions.length > 0) {
        resultsEl.appendChild(renderCitationList(data.suggestions, "You may want to read"));
      }
    } else {
      resultsEl.appendChild(renderAnswerWithLinks(data.answer || "", data.citations || []));
      if (Array.isArray(data.citations) && data.citations.length > 0) {
        resultsEl.appendChild(renderCitationList(data.citations, "Citations"));
      }
    }

    resultsEl.appendChild(renderMeta(data));
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) {
      return;
    }

    const payload = {
      question: question,
      strategy: strategySelect.value
    };
    if (versionSelect.value) {
      payload.filters = { version: versionSelect.value };
    }

    askButton.disabled = true;
    resultsEl.replaceChildren();
    const loading = document.createElement("p");
    loading.className = "loading";
    loading.textContent = "Searching and verifying citations...";
    resultsEl.appendChild(loading);

    try {
      const response = await fetch("demo/query", {  // relative: the page also runs under a path prefix (/sift/)
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || "Request failed");
        renderError(detail);
        return;
      }
      renderResult(data);
    } catch (err) {
      renderError(err instanceof Error ? err.message : "Network error");
    } finally {
      askButton.disabled = false;
    }
  });
})();
