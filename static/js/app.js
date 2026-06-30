(() => {
  const NS = "http://www.w3.org/2000/svg";

  // ---- Graph layout (matches the LangGraph topology in rag_graph.py) ----
  const NODES = [
    { id: "start", x: 0,  y: 280, w: 70,  h: 40, label: "START", sub: "", end: true },
    { id: "decide_retrieval", x: 150, y: 280, w: 150, h: 56, label: "Decide", sub: "retrieval?" },
    { id: "generate_direct",  x: 330, y: 150, w: 160, h: 56, label: "Generate", sub: "direct answer" },
    { id: "retrieve",         x: 330, y: 490, w: 150, h: 56, label: "Retrieve", sub: "vector search" },
    { id: "is_relevant",      x: 500, y: 260, w: 150, h: 56, label: "Relevance", sub: "filter docs" },
    { id: "generate_from_context", x: 680, y: 100, w: 170, h: 56, label: "Generate", sub: "from context" },
    { id: "no_answer_found",  x: 690, y: 340, w: 170, h: 56, label: "No Answer", sub: "found" },
    { id: "is_sup",           x: 900, y: 100, w: 140, h: 56, label: "IsSUP", sub: "verify support" },
    { id: "revise_answer",    x: 1200, y: 140, w: 140, h: 56, label: "Revise", sub: "answer" },
    { id: "is_use",           x: 1040, y: 540, w: 140, h: 56, label: "IsUSE", sub: "check useful" },
    { id: "rewrite_question", x: 650, y: 540, w: 160, h: 56, label: "Rewrite", sub: "query" },
    { id: "end", x: 1230, y: 280, w: 70, h: 40, label: "END", sub: "", end: true },
  ];
  const NODE_BY_ID = Object.fromEntries(NODES.map(n => [n.id, n]));

  const EDGES = [
    { id: "start-decide_retrieval", from: "start", to: "decide_retrieval" },
    { id: "decide_retrieval-generate_direct", from: "decide_retrieval", to: "generate_direct" },
    { id: "decide_retrieval-retrieve", from: "decide_retrieval", to: "retrieve" },
    { id: "generate_direct-end", from: "generate_direct", to: "end" },
    { id: "retrieve-is_relevant", from: "retrieve", to: "is_relevant" },
    { id: "is_relevant-generate_from_context", from: "is_relevant", to: "generate_from_context" },
    { id: "is_relevant-no_answer_found", from: "is_relevant", to: "no_answer_found" },
    { id: "no_answer_found-end", from: "no_answer_found", to: "end" },
    { id: "generate_from_context-is_sup", from: "generate_from_context", to: "is_sup" },
    { id: "is_sup-revise_answer", from: "is_sup", to: "revise_answer", kind: "retry" },
    { id: "revise_answer-is_sup", from: "revise_answer", to: "is_sup", kind: "retry" },
    { id: "is_sup-is_use", from: "is_sup", to: "is_use" },
    { id: "is_use-end", from: "is_use", to: "end" },
    { id: "is_use-rewrite_question", from: "is_use", to: "rewrite_question", kind: "retry" },
    { id: "is_use-no_answer_found", from: "is_use", to: "no_answer_found" },
    { id: "rewrite_question-retrieve", from: "rewrite_question", to: "retrieve", kind: "retry" },
  ];

  // sequence of (from,to) node-name pairs the backend actually walks, in order, drives which
  // edge lights up between consecutive node_start events
  const TRANSITION_HINTS = {
    "decide_retrieval": { "generate_direct": "decide_retrieval-generate_direct", "retrieve": "decide_retrieval-retrieve" },
    "retrieve": { "is_relevant": "retrieve-is_relevant" },
    "is_relevant": { "generate_from_context": "is_relevant-generate_from_context", "no_answer_found": "is_relevant-no_answer_found" },
    "generate_from_context": { "is_sup": "generate_from_context-is_sup" },
    "is_sup": { "revise_answer": "is_sup-revise_answer", "is_use": "is_sup-is_use" },
    "revise_answer": { "is_sup": "revise_answer-is_sup" },
    "is_use": { "rewrite_question": "is_use-rewrite_question", "no_answer_found": "is_use-no_answer_found" },
    "rewrite_question": { "retrieve": "rewrite_question-retrieve" },
  };

  function centerOf(n) { return { x: n.x + n.w / 2, y: n.y + n.h / 2 }; }

  function edgePath(edge) {
    const a = NODE_BY_ID[edge.from], b = NODE_BY_ID[edge.to];
    const c1 = centerOf(a), c2 = centerOf(b);
    // anchor points on node boundary, roughly toward target
    const dx = c2.x - c1.x, dy = c2.y - c1.y;
    const ax = c1.x + Math.sign(dx || 1) * (a.w / 2), ay = c1.y;
    const bx = c2.x - Math.sign(dx || 1) * (b.w / 2), by = c2.y;
    const loop = edge.from === edge.to || (edge.kind === "retry" && Math.abs(dy) < 5);
    const bend = Math.max(60, Math.abs(dx) * 0.4);
    let d;
    if (edge.kind === "retry" && edge.from !== edge.to) {
      // route retry/loop edges with a visible outward bow so they read as loops
      const bow = dy === 0 ? -70 : (dy > 0 ? 70 : -70);
      d = `M ${ax} ${ay} C ${ax + bend} ${ay + bow}, ${bx - bend} ${by + bow}, ${bx} ${by}`;
    } else {
      d = `M ${ax} ${ay} C ${ax + bend} ${ay}, ${bx - bend} ${by}, ${bx} ${by}`;
    }
    return d;
  }

  // ---- Build SVG ----
  const svg = document.getElementById("graphSvg");
  const edgeLayer = document.getElementById("edgeLayer");
  const nodeLayer = document.getElementById("nodeLayer");
  const edgeEls = {};
  const nodeEls = {};

  EDGES.forEach(edge => {
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", edgePath(edge));
    path.setAttribute("class", "edge-path");
    path.dataset.id = edge.id;
    edgeLayer.appendChild(path);
    edgeEls[edge.id] = path;
  });

  NODES.forEach(n => {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "node-group" + (n.end ? " end-node" : ""));
    g.setAttribute("transform", `translate(${n.x},${n.y})`);

    if (n.end) {
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("cx", n.w / 2); c.setAttribute("cy", n.h / 2); c.setAttribute("r", n.w / 2);
      c.setAttribute("class", "node-shape");
      g.appendChild(c);
      if (n.label) {
        const t = document.createElementNS(NS, "text");
        t.setAttribute("x", n.w / 2); t.setAttribute("y", n.h / 2 + 28);
        t.setAttribute("class", "node-label"); t.textContent = n.label;
        g.appendChild(t);
      }
    } else {
      const r = document.createElementNS(NS, "rect");
      r.setAttribute("width", n.w); r.setAttribute("height", n.h);
      r.setAttribute("rx", 12); r.setAttribute("class", "node-shape");
      g.appendChild(r);
      const t1 = document.createElementNS(NS, "text");
      t1.setAttribute("x", n.w / 2); t1.setAttribute("y", n.h / 2 - 6);
      t1.setAttribute("class", "node-label"); t1.textContent = n.label;
      g.appendChild(t1);
      const t2 = document.createElementNS(NS, "text");
      t2.setAttribute("x", n.w / 2); t2.setAttribute("y", n.h / 2 + 13);
      t2.setAttribute("class", "node-sub"); t2.textContent = n.sub;
      g.appendChild(t2);
    }
    nodeLayer.appendChild(g);
    nodeEls[n.id] = g;
  });

  // ---- State machine helpers ----
  const edgeTraversalCount = {};
  const edgeBadges = {};

  function resetGraph() {
    Object.values(nodeEls).forEach(el => { el.className.baseVal = el.className.baseVal.replace(/\b(active|success|retry|fail|pulse|lit|pop)\b/g, "").trim(); el.setAttribute("class", el.getAttribute("class").split(" ")[0] + (el.classList.contains("end-node") ? " end-node" : "")); });
    Object.values(edgeEls).forEach(el => el.classList.remove("flowing", "retry-flow", "traversed", "retry-traversed"));
    Object.keys(edgeTraversalCount).forEach(k => delete edgeTraversalCount[k]);
    Object.values(edgeBadges).forEach(b => b.classList.remove("show"));
  }

  function setNodeState(id, state) {
    const el = nodeEls[id];
    if (!el) return;
    el.classList.remove("active", "success", "retry", "fail", "pulse");
    if (state) el.classList.add(state);
    if (state === "active") {
      el.classList.add("pulse");
      popNode(id);
    }
  }

  function popNode(id) {
    const el = nodeEls[id];
    if (!el) return;
    el.classList.remove("pop");
    void el.getBBox(); // force reflow so the animation can retrigger on repeat visits (loops)
    el.classList.add("pop");
  }

  function edgeMidpoint(pathEl) {
    try {
      const len = pathEl.getTotalLength();
      return pathEl.getPointAtLength(len / 2);
    } catch { return null; }
  }

  function showEdgeBadge(edgeId, count) {
    const pathEl = edgeEls[edgeId];
    if (!pathEl) return;
    let badge = edgeBadges[edgeId];
    if (!badge) {
      badge = document.createElementNS(NS, "g");
      badge.setAttribute("class", "edge-badge");
      const circle = document.createElementNS(NS, "circle");
      circle.setAttribute("r", 10);
      circle.setAttribute("class", "edge-badge-bg");
      const text = document.createElementNS(NS, "text");
      text.setAttribute("class", "edge-badge-text");
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("dominant-baseline", "central");
      badge.appendChild(circle);
      badge.appendChild(text);
      edgeLayer.appendChild(badge);
      edgeBadges[edgeId] = badge;
    }
    const pt = edgeMidpoint(pathEl);
    if (pt) badge.setAttribute("transform", `translate(${pt.x},${pt.y})`);
    badge.querySelector(".edge-badge-text").textContent = "×" + count;
    badge.classList.add("show");
  }

  let lastNode = "start";
  function lightEdge(from, to, kind) {
    const edgeId = (TRANSITION_HINTS[from] && TRANSITION_HINTS[from][to]) || `${from}-${to}`;
    const el = edgeEls[edgeId];
    if (!el) return;

    edgeTraversalCount[edgeId] = (edgeTraversalCount[edgeId] || 0) + 1;
    const count = edgeTraversalCount[edgeId];

    el.classList.remove("traversed", "retry-traversed");
    el.classList.add("flowing");
    if (kind === "retry") el.classList.add("retry-flow");

    setTimeout(() => {
      el.classList.remove("flowing", "retry-flow");
      el.classList.add("traversed");
      if (kind === "retry") el.classList.add("retry-traversed");
    }, 900);

    if (count > 1) showEdgeBadge(edgeId, count);
  }

  // ---- Log feed ----
  const logFeed = document.getElementById("logFeed");
  function addLog(title, detail, kind) {
    if (logFeed.querySelector(".log-empty")) logFeed.innerHTML = "";
    const div = document.createElement("div");
    div.className = "log-entry" + (kind === "active" ? " active-entry" : kind === "retry" ? " retry-entry" : kind === "fail" ? " fail-entry" : "");
    const time = new Date().toLocaleTimeString([], { hour12: false });
    div.innerHTML = `<div class="le-head"><span>${title}</span><span class="le-time">${time}</span></div>` + (detail ? `<div class="le-detail">${detail}</div>` : "");
    logFeed.appendChild(div);
    logFeed.scrollTop = logFeed.scrollHeight;
  }

  // ---- Stats ----
  const statRetries = document.getElementById("statRetries");
  const statRewrites = document.getElementById("statRewrites");
  const statSteps = document.getElementById("statSteps");
  let stepCount = 0;

  // ---- Form / SSE wiring ----
  const form = document.getElementById("askForm");
  const input = document.getElementById("questionInput");
  const askBtn = document.getElementById("askBtn");
  const btnLabel = askBtn.querySelector(".btn-label");
  const spinner = askBtn.querySelector(".spinner");
  const answerBlock = document.getElementById("answerBlock");
  const answerText = document.getElementById("answerText");

  document.querySelectorAll("#presets button").forEach(b => {
    b.addEventListener("click", () => { input.value = b.dataset.q; });
  });

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function describeNodeEnd(node, data) {
    data = data || {};
    switch (node) {
      case "decide_retrieval":
        return data.need_retrieval ? "→ retrieval required" : "→ answer from general knowledge";
      case "retrieve":
        return `retrieved ${data.docs ? data.docs.length : "?"} chunk(s)`;
      case "is_relevant":
        return `${data.relevant_docs ? data.relevant_docs.length : "?"} chunk(s) passed relevance filter`;
      case "generate_from_context":
      case "generate_direct":
        return data.answer ? `draft: "${escapeHtml(String(data.answer).slice(0, 110))}${String(data.answer).length > 110 ? "…" : ""}"` : "";
      case "is_sup":
        return `issup = <b>${data.issup}</b>`;
      case "revise_answer":
        return `support retries: ${data.retries ?? "?"}`;
      case "is_use":
        return `isuse = <b>${data.isuse}</b>${data.use_reason ? " — " + escapeHtml(data.use_reason) : ""}`;
      case "rewrite_question":
        return `rewrite #${data.rewrite_tries ?? "?"}: "${escapeHtml(data.retrieval_query || "")}"`;
      case "no_answer_found":
        return "graph gave up — returning fallback message";
      default:
        return "";
    }
  }

    function runQuestion(question) {
    resetGraph();
    document.getElementById("graphIdleHint")?.classList.add("hint-hidden");
    answerBlock.hidden = true;
    logFeed.innerHTML = "";
    stepCount = 0; statSteps.textContent = "0";
    statRetries.textContent = "0"; statRewrites.textContent = "0";
    lastNode = "start";
    askBtn.disabled = true; btnLabel.textContent = "Running…"; spinner.hidden = false;

    addLog("Run started", escapeHtml(question), "active");

    const es = new EventSource("/api/ask?question=" + encodeURIComponent(question));

    // ---- throttled event queue ----
    const STEP_DELAY = 900;
    const eventQueue = [];
    let queueTimer = null;

    function enqueue(evt) {
      eventQueue.push(evt);
      if (!queueTimer) drainQueue();
    }

    function drainQueue() {
      if (eventQueue.length === 0) { queueTimer = null; return; }
      const evt = eventQueue.shift();

      if (evt.type === "node_start") {
        const kind = (evt.node === "revise_answer" || evt.node === "rewrite_question") ? "retry" : null;
        lightEdge(lastNode, evt.node, kind);
        setNodeState(evt.node, "active");
        addLog(`▶ ${evt.node}`, evt.detail ? escapeHtml(evt.detail) : "", "active");
      } else if (evt.type === "node_end") {
        stepCount += 1; statSteps.textContent = stepCount;
        const isRetryNode = evt.node === "revise_answer" || evt.node === "rewrite_question";
        const isFailNode = evt.node === "no_answer_found";
        setNodeState(evt.node, isFailNode ? "fail" : isRetryNode ? "retry" : "success");
        lastNode = evt.node;
        const data = evt.data || {};
        if (typeof data.retries === "number") statRetries.textContent = data.retries;
        if (typeof data.rewrite_tries === "number") statRewrites.textContent = data.rewrite_tries;
        addLog(`✓ ${evt.node}`, describeNodeEnd(evt.node, data) || evt.detail || "", isFailNode ? "fail" : isRetryNode ? "retry" : null);
      } else if (evt.type === "final") {
        lightEdge(lastNode, "end", null);
        const endEl = nodeEls["end"];
        if (endEl) endEl.classList.add("lit");
        answerBlock.hidden = false;
        answerText.textContent = evt.answer || "(no answer)";
        addLog("Final answer ready", "", "active");
        finish();
      } else if (evt.type === "error") {
        addLog("Error", escapeHtml(evt.message), "fail");
        finish();
      }

      queueTimer = setTimeout(drainQueue, STEP_DELAY);
    }

    es.onmessage = (ev) => {
      let evt;
      try { evt = JSON.parse(ev.data); } catch { return; }
      // Close the transport as soon as the server signals done — the queue
      // keeps draining the visual playback independently.
      if (evt.type === "final" || evt.type === "error") es.close();
      enqueue(evt);
    };

    let finished = false;
    function finish() {
      if (finished) return;
      finished = true;
      es.close();
      askBtn.disabled = false; btnLabel.textContent = "Run graph"; spinner.hidden = true;
    }

    es.addEventListener("error", () => {
      // A genuine mid-stream drop (server crashed, network hiccup) before we ever got a
      // 'final' event. EventSource would otherwise silently auto-reconnect and re-run the
      // question from scratch, so we close it ourselves and surface this as a failure.
      if (!finished) {
        addLog("Connection lost", "Stream ended unexpectedly before a final answer arrived.", "fail");
        finish();
      }
    });

    // Safety net in case neither 'final' nor 'error' ever fires.
    setTimeout(() => { if (!finished) finish(); }, 120000);
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    runQuestion(q);
  });

  // ---- Demo-mode intro notice ----
  const overlay = document.getElementById("introOverlay");
  const presetTip = document.getElementById("presetTip");
  const presetsEl = document.getElementById("presets");

  function showPresetTip() {
    if (!presetTip) return;
    presetTip.classList.remove("tip-hidden");
    presetsEl?.classList.add("highlight");
  }
  function hidePresetTip() {
    if (!presetTip) return;
    presetTip.classList.add("tip-hidden");
    presetsEl?.classList.remove("highlight");
  }

  if (overlay) {
    const closeOverlay = () => { overlay.classList.add("intro-hidden"); showPresetTip(); };
    const openOverlay = () => { overlay.classList.remove("intro-hidden"); hidePresetTip(); };
    document.getElementById("introClose")?.addEventListener("click", closeOverlay);
    document.getElementById("introDismiss")?.addEventListener("click", closeOverlay);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) closeOverlay(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeOverlay(); });
    document.getElementById("reopenIntro")?.addEventListener("click", openOverlay);
  } else {
    showPresetTip();
  }

  document.getElementById("presetTipClose")?.addEventListener("click", hidePresetTip);
  document.querySelectorAll("#presets button").forEach(b => {
    b.addEventListener("click", hidePresetTip);
  });

  // ---- "vs Simple RAG" comparison modal ----
  const compareOverlay = document.getElementById("compareOverlay");
  if (compareOverlay) {
    const closeCompare = () => compareOverlay.classList.add("intro-hidden");
    const openCompare = () => compareOverlay.classList.remove("intro-hidden");
    document.getElementById("openCompare")?.addEventListener("click", openCompare);
    document.getElementById("compareClose")?.addEventListener("click", closeCompare);
    document.getElementById("compareDismiss")?.addEventListener("click", closeCompare);
    compareOverlay.addEventListener("click", (e) => { if (e.target === compareOverlay) closeCompare(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeCompare(); });
  }
})();
