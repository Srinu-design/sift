(function () {
  "use strict";

  // Everything from the server is put on the page as text (textContent), never as HTML, and
  // every URL is relative so the page also works under a path prefix (/sift/).
  const SVG = "http://www.w3.org/2000/svg";

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function svg(tag, attrs) {
    const node = document.createElementNS(SVG, tag);
    for (const [name, value] of Object.entries(attrs || {})) node.setAttribute(name, String(value));
    return node;
  }

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

  // ---------------------------------------------------------------- light and dark
  const THEME_STORAGE = "sift.theme"; // theme.js reads it before the page is drawn
  const themeButton = document.getElementById("theme-toggle");

  function currentTheme() {
    const chosen = document.documentElement.dataset.theme;
    if (chosen) return chosen;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function labelThemeButton() {
    const label = "Switch to " + (currentTheme() === "dark" ? "light" : "dark") + " mode";
    themeButton.setAttribute("aria-label", label);
    themeButton.title = label;
  }

  themeButton.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(THEME_STORAGE, next);
    } catch {
      // storage blocked: the choice lasts until the page is reloaded
    }
    labelThemeButton();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", labelThemeButton);
  labelThemeButton();

  // ---------------------------------------------------------------- live status
  async function checkStatus() {
    if (document.hidden) return; // checked again when the tab is shown
    const box = document.getElementById("status");
    let state = "down";
    let text = "Down";
    let title = "The server did not answer.";
    try {
      const response = await fetch("readyz", { cache: "no-store" });
      const body = await response.json().catch(() => ({}));
      if (response.ok) {
        [state, text] = ["ok", "Live"];
        title = "database, models: ok; redis: " + String(body.redis || "unknown");
      } else {
        title = typeof body.detail === "string" ? body.detail : "HTTP " + String(response.status);
      }
    } catch {
      // keep "Down"
    }
    box.dataset.state = state;
    box.title = title;
    document.getElementById("status-text").textContent = text;
  }
  checkStatus();
  setInterval(checkStatus, 15000);
  document.addEventListener("visibilitychange", checkStatus);

  // ---------------------------------------------------------------- section links
  // Marks the How it works section being read in the sticky row of section links.
  const sectionLinks = new Map();
  for (const a of document.querySelectorAll(".subnav a")) sectionLinks.set(a.hash.slice(1), a);
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const [id, a] of sectionLinks) {
          if (id === entry.target.id) a.setAttribute("aria-current", "true");
          else a.removeAttribute("aria-current");
        }
      }
    },
    { rootMargin: "-30% 0px -60% 0px" },
  );
  for (const id of sectionLinks.keys()) {
    const section = document.getElementById(id);
    if (section) sectionObserver.observe(section);
  }

  // ---------------------------------------------------------------- eval history
  function cell(text) {
    return el("td", "", text);
  }

  function commitCell(commit) {
    const td = document.createElement("td");
    if (!commit) {
      td.textContent = "—";
      return td;
    }
    const a = el("a", "mono", commit);
    a.href = "https://github.com/Srinu-design/sift/commit/" + encodeURIComponent(commit);
    td.appendChild(a);
    return td;
  }

  function strategyCell(strategy) {
    const td = el("td", "", strategy);
    // A suffix after the strategy name (-r10, -nofilter, -and) marks an ablation run.
    if (/^(hybrid_rerank|hybrid|naive|lexical)-/.test(strategy)) {
      td.append(" ", el("span", "chip-muted", "ablation"));
    }
    return td;
  }

  function splitCell(split) {
    const td = document.createElement("td");
    td.appendChild(el("span", split === "test" ? "chip-test" : "chip-dev", split || "—"));
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
      strategyCell(String(run.strategy || "")),
      splitCell(String(run.split || "")),
      cell(String(run.golden || "")),
    );
    for (const name of metricNames) tr.appendChild(cell(fmt((run.metrics || {})[name])));
    return tr;
  }

  // MRR@10 of hybrid_rerank on golden v1 dev, one point per run, evenly spaced in run order.
  function drawChart(runs) {
    const chart = document.getElementById("mrr-chart");
    const points = runs
      .filter(
        (r) =>
          r.kind === "retrieval" &&
          r.strategy === "hybrid_rerank" &&
          r.split === "dev" &&
          r.golden === "v1" &&
          typeof (r.metrics || {})["mrr@10"] === "number",
      )
      .sort((a, b) => String(a.run_at).localeCompare(String(b.run_at)));
    chart.replaceChildren();
    const desc = svg("desc", { id: "chart-desc" });
    chart.appendChild(desc);
    if (points.length < 2) {
      desc.textContent = "Not enough runs to draw.";
      return;
    }
    const W = 640;
    const H = 240;
    const pad = { left: 56, right: 24, top: 30, bottom: 44 };
    const inset = 28; // keeps the first and last labels clear of the axis
    chart.setAttribute("viewBox", "0 0 " + W + " " + H);
    const values = points.map((p) => p.metrics["mrr@10"]);
    const lo = Math.floor((Math.min(...values) - 0.01) * 100) / 100;
    const hi = Math.ceil((Math.max(...values) + 0.01) * 100) / 100;
    const x = (i) => pad.left + inset + (i * (W - pad.left - pad.right - 2 * inset)) / (points.length - 1);
    const y = (v) => pad.top + ((hi - v) * (H - pad.top - pad.bottom)) / (hi - lo);

    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4;
      chart.appendChild(svg("line", { class: "grid", x1: pad.left, x2: W - pad.right, y1: y(v), y2: y(v) }));
      const label = svg("text", { class: "axis", x: pad.left - 10, y: y(v) + 4, "text-anchor": "end" });
      label.textContent = v.toFixed(3);
      chart.appendChild(label);
    }
    const path = points.map((p, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(values[i]).toFixed(1)).join(" ");
    chart.appendChild(svg("path", { class: "line", d: path }));
    points.forEach((p, i) => {
      const when = String(p.run_at).replace("T", " ").slice(5, 16);
      const dot = svg("circle", { class: "dot", cx: x(i), cy: y(values[i]), r: 5 });
      const tip = svg("title");
      tip.textContent = when + " UTC: MRR@10 " + values[i].toFixed(3);
      dot.appendChild(tip);
      chart.appendChild(dot);
      const value = svg("text", { class: "value", x: x(i), y: y(values[i]) - 12, "text-anchor": "middle" });
      value.textContent = values[i].toFixed(3);
      chart.appendChild(value);
      const date = svg("text", { class: "axis", x: x(i), y: H - pad.bottom + 22, "text-anchor": "middle" });
      date.textContent = when.slice(0, 5);
      chart.appendChild(date);
      const time = svg("text", { class: "axis", x: x(i), y: H - pad.bottom + 36, "text-anchor": "middle" });
      time.textContent = when.slice(6);
      chart.appendChild(time);
    });
    desc.textContent =
      "MRR@10 over " +
      String(points.length) +
      " dev runs, from " +
      values[0].toFixed(3) +
      " to " +
      values[values.length - 1].toFixed(3) +
      ".";
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
      retrieval.replaceChildren();
      answers.replaceChildren();
      for (const run of runs) {
        if (run.kind === "retrieval") {
          retrieval.appendChild(runRow(run, ["hit@5", "hit@8", "recall@10", "mrr@10"]));
        } else if (run.kind === "answers") {
          answers.appendChild(
            runRow(run, ["refusal_recall", "refusal_precision", "citation_accuracy"]),
          );
        }
      }
      drawChart(runs);
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
  const STAGES = ["embed", "dense", "lexical", "fuse", "rerank", "llm"];

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

  function externalLink(url, className, text) {
    const a = el("a", className, text);
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    return a;
  }

  function renderAnswerWithLinks(answerText, citations) {
    const byNum = new Map();
    for (const c of citations || []) {
      byNum.set(Number(c.n), c);
    }

    const p = el("p", "answer-text");
    const parts = answerText.split(/(\[S\d+\])/g);
    for (const part of parts) {
      const match = /^\[S(\d+)\]$/.exec(part);
      if (match) {
        const cit = byNum.get(Number(match[1]));
        if (cit && cit.url) {
          p.appendChild(externalLink(cit.url, "inline-citation", part));
          continue;
        }
      }
      p.appendChild(document.createTextNode(part));
    }
    return p;
  }

  // One card per numbered source: its number, title, version and section (or page).
  function renderSources(items) {
    const list = el("ol", "source-cards");
    for (const c of items) {
      const li = document.createElement("li");
      const link = externalLink(c.url, "source-card");
      link.append(el("span", "source-n", "S" + String(c.n)));
      const body = el("span", "source-body");
      body.append(el("span", "source-title", String(c.title)));
      const meta = el("span", "source-meta");
      if (c.version) meta.append(el("span", "chip-version", "PostgreSQL " + String(c.version)));
      const where = c.section || (c.page ? "page " + String(c.page) : "");
      if (where) meta.append(el("span", "", String(where)));
      body.append(meta);
      link.append(body);
      li.appendChild(link);
      list.appendChild(li);
    }
    return list;
  }

  // The answer's metadata row, with a bar of time per stage.
  function renderMeta(data) {
    const timings = data.timings_ms || {};
    const stages = STAGES.filter((s) => Number(timings[s]) > 0);
    for (const s of Object.keys(timings)) {
      if (s !== "total" && !stages.includes(s) && Number(timings[s]) > 0) stages.push(s);
    }
    // timings_ms carries its own "total" next to the stages; use it rather than adding it in.
    let totalMs = Number(timings.total) || 0;
    if (!totalMs) for (const s of stages) totalMs += Number(timings[s]);

    const meta = el("div", "result-meta");
    const facts = el("p", "meta-facts");
    facts.append(
      el("span", "chip-plain", String(data.strategy)),
      el("span", data.cache_hit ? "chip-ok" : "chip-plain", data.cache_hit ? "cache hit" : "cache miss"),
      el("span", "", totalMs.toFixed(0) + " ms"),
      el(
        "span",
        "",
        "$" + (typeof data.cost_usd === "number" ? data.cost_usd.toFixed(6) : "0.000000"),
      ),
      el("span", "mono", "request " + String(data.request_id)),
    );
    meta.appendChild(facts);

    if (totalMs > 0) {
      const bar = svg("svg", {
        class: "stage-bar",
        viewBox: "0 0 1000 12",
        preserveAspectRatio: "none",
        "aria-hidden": "true",
      });
      let at = 0;
      for (const s of stages) {
        const w = (Number(timings[s]) / totalMs) * 1000;
        bar.appendChild(svg("rect", { class: "stage-" + s, x: at.toFixed(1), y: 0, width: w.toFixed(1), height: 12 }));
        at += w;
      }
      meta.appendChild(bar);
      const legend = el("ul", "stage-legend");
      for (const s of stages) {
        const li = el("li");
        const ms = Number(timings[s]);
        li.append(el("span", "swatch stage-" + s), s + " " + ms.toFixed(ms < 10 ? 1 : 0) + " ms");
        legend.appendChild(li);
      }
      meta.appendChild(legend);
    }
    return meta;
  }

  function renderError(message) {
    const box = el("div", "callout");
    box.append(el("h2", "", "The demo could not answer"), el("p", "", message));
    resultsEl.replaceChildren(box);
  }

  function renderResult(data) {
    resultsEl.replaceChildren();

    if (data.refused) {
      const box = el("div", "callout");
      box.append(
        el("h2", "", "Not enough evidence to answer"),
        el("p", "", data.refusal_reason || "Refused."),
      );
      if (Array.isArray(data.suggestions) && data.suggestions.length > 0) {
        box.append(el("p", "suggest-label", "You may want to read"));
        const pills = el("ul", "suggestions");
        for (const c of data.suggestions) {
          const li = document.createElement("li");
          const parts = [String(c.title)];
          if (c.version) parts.push("PostgreSQL " + String(c.version));
          li.appendChild(externalLink(c.url, "", parts.join(" · ")));
          pills.appendChild(li);
        }
        box.append(pills);
      }
      resultsEl.appendChild(box);
    } else {
      const card = el("article", "answer-card");
      const head = el("div", "answer-head");
      head.append(el("h2", "", "Answer"), el("span", "chip-ok", "cited"));
      card.append(head, renderAnswerWithLinks(data.answer || "", data.citations || []));
      resultsEl.appendChild(card);
      if (Array.isArray(data.citations) && data.citations.length > 0) {
        resultsEl.append(el("h2", "block-heading", "Sources"), renderSources(data.citations));
      }
    }

    resultsEl.appendChild(renderMeta(data));
  }

  function renderLoading() {
    const box = el("div", "answer-card loading");
    box.append(el("p", "loading-text", "Searching and verifying citations…"));
    for (const w of ["w90", "w75", "w60"]) box.append(el("span", "skeleton " + w));
    resultsEl.replaceChildren(box);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (askButton.disabled) return; // a question is in flight; Enter submits even so
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
    renderLoading();

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
