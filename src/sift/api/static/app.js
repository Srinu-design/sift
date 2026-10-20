(function () {
  "use strict";

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
      const response = await fetch("/v1/query", {
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
