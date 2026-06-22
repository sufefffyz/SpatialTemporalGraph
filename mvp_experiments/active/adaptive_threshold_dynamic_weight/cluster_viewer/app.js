(function () {
  "use strict";

  const data = window.CLUSTER_VIEWER_DATA;
  if (!data || !data.datasets) {
    throw new Error("Missing cluster data. Run export_cluster_viewer_data.py first.");
  }

  const palette = [
    "#2f6df6",
    "#e45756",
    "#14a36f",
    "#f29e2e",
    "#7b61ff",
    "#17a2b8",
    "#b64fc8",
    "#65737e",
  ];

  const state = {
    dataset: Object.keys(data.datasets)[0],
    embedding: "time",
    day: 0,
    cluster: "all",
    fade: 0.06,
    scale: 1,
    tx: 0,
    ty: 0,
    baseScale: 1,
    baseTx: 0,
    baseTy: 0,
    dragging: false,
    lastX: 0,
    lastY: 0,
    timer: null,
    hoverIndex: -1,
  };

  const els = {
    canvas: document.getElementById("mapCanvas"),
    tooltip: document.getElementById("tooltip"),
    dataset: document.getElementById("datasetSelect"),
    embedding: document.getElementById("embeddingSelect"),
    cluster: document.getElementById("clusterSelect"),
    day: document.getElementById("daySlider"),
    dayLabel: document.getElementById("dayLabel"),
    stepLabel: document.getElementById("stepLabel"),
    fade: document.getElementById("fadeSlider"),
    play: document.getElementById("playButton"),
    fit: document.getElementById("fitButton"),
    stats: document.getElementById("stats"),
    legend: document.getElementById("legend"),
  };

  const ctx = els.canvas.getContext("2d");

  function currentDataset() {
    return data.datasets[state.dataset];
  }

  function currentEmbedding() {
    return currentDataset().embeddings[state.embedding];
  }

  function currentLabels() {
    return currentEmbedding().labels[state.day];
  }

  function setOptions(select, values, selected) {
    select.innerHTML = "";
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      if (value === selected) option.selected = true;
      select.appendChild(option);
    }
  }

  function initControls() {
    const datasetNames = Object.keys(data.datasets);
    setOptions(els.dataset, datasetNames, state.dataset);
    refreshEmbeddingOptions();
    refreshClusterOptions();
  }

  function refreshEmbeddingOptions() {
    const embeddings = Object.keys(currentDataset().embeddings);
    if (!embeddings.includes(state.embedding)) state.embedding = embeddings[0];
    setOptions(els.embedding, embeddings, state.embedding);
    refreshDaySlider();
  }

  function refreshClusterOptions() {
    const values = ["all"];
    for (let i = 0; i < currentDataset().numClusters; i += 1) values.push(String(i));
    if (!values.includes(String(state.cluster))) state.cluster = "all";
    els.cluster.innerHTML = "";
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value === "all" ? "All clusters" : `Cluster ${value}`;
      if (String(state.cluster) === value) option.selected = true;
      els.cluster.appendChild(option);
    }
  }

  function refreshDaySlider() {
    const days = currentEmbedding().days;
    state.day = Math.min(state.day, days.length - 1);
    els.day.min = "0";
    els.day.max = String(days.length - 1);
    els.day.value = String(state.day);
  }

  function resizeCanvas() {
    const rect = els.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (els.canvas.width !== width || els.canvas.height !== height) {
      els.canvas.width = width;
      els.canvas.height = height;
      fitView();
    }
    draw();
  }

  function nodeBounds(nodes) {
    let minLon = Infinity;
    let maxLon = -Infinity;
    let minLat = Infinity;
    let maxLat = -Infinity;
    for (const node of nodes) {
      minLon = Math.min(minLon, node.lon);
      maxLon = Math.max(maxLon, node.lon);
      minLat = Math.min(minLat, node.lat);
      maxLat = Math.max(maxLat, node.lat);
    }
    return { minLon, maxLon, minLat, maxLat };
  }

  function fitView() {
    const nodes = currentDataset().nodes;
    const bounds = nodeBounds(nodes);
    const w = els.canvas.width;
    const h = els.canvas.height;
    const pad = Math.max(28, Math.min(w, h) * 0.06);
    const lonSpan = Math.max(bounds.maxLon - bounds.minLon, 1e-6);
    const latSpan = Math.max(bounds.maxLat - bounds.minLat, 1e-6);
    const sx = (w - 2 * pad) / lonSpan;
    const sy = (h - 2 * pad) / latSpan;
    state.baseScale = Math.min(sx, sy);
    state.baseTx = pad - bounds.minLon * state.baseScale + ((w - 2 * pad) - lonSpan * state.baseScale) / 2;
    state.baseTy = pad + bounds.maxLat * state.baseScale + ((h - 2 * pad) - latSpan * state.baseScale) / 2;
    state.scale = 1;
    state.tx = 0;
    state.ty = 0;
  }

  function project(node) {
    const scale = state.baseScale * state.scale;
    return {
      x: state.baseTx + state.tx + node.lon * scale,
      y: state.baseTy + state.ty - node.lat * scale,
    };
  }

  function pointRadius() {
    const n = currentDataset().nodes.length;
    if (n > 3000) return 2.0 * Math.sqrt(window.devicePixelRatio || 1);
    if (n > 1500) return 2.4 * Math.sqrt(window.devicePixelRatio || 1);
    return 3.2 * Math.sqrt(window.devicePixelRatio || 1);
  }

  function draw() {
    const w = els.canvas.width;
    const h = els.canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.fillStyle = "#f2f5f8";
    ctx.fillRect(0, 0, w, h);
    drawAxesHint();

    const dataset = currentDataset();
    const labels = currentLabels();
    const selectedCluster = state.cluster === "all" ? null : Number(state.cluster);
    const r = pointRadius();

    for (let i = 0; i < dataset.nodes.length; i += 1) {
      const node = dataset.nodes[i];
      const label = labels[i];
      const pos = project(node);
      const selected = selectedCluster === null || label === selectedCluster;
      ctx.globalAlpha = selected ? 0.92 : state.fade;
      ctx.fillStyle = palette[label % palette.length];
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, selected ? r : Math.max(1.4, r * 0.75), 0, Math.PI * 2);
      ctx.fill();
    }

    if (state.hoverIndex >= 0) {
      const node = dataset.nodes[state.hoverIndex];
      const label = labels[state.hoverIndex];
      const pos = project(node);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
      ctx.fillStyle = palette[label % palette.length];
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r + 3 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    ctx.restore();
    updateLabels();
  }

  function drawAxesHint() {
    const w = els.canvas.width;
    const h = els.canvas.height;
    ctx.save();
    ctx.strokeStyle = "rgba(23, 32, 42, 0.08)";
    ctx.lineWidth = 1;
    const step = 52 * (window.devicePixelRatio || 1);
    for (let x = 0; x < w; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function clusterCounts(labels, numClusters) {
    const counts = Array(numClusters).fill(0);
    for (const label of labels) counts[label] += 1;
    return counts;
  }

  function updateLabels() {
    const embedding = currentEmbedding();
    const day = embedding.days[state.day];
    const labels = currentLabels();
    const counts = clusterCounts(labels, currentDataset().numClusters);
    const total = currentDataset().nodes.length;
    els.dayLabel.textContent = day.label;
    els.stepLabel.textContent = `${day.startStep}-${day.endStep}`;
    const selectedText = state.cluster === "all"
      ? `nodes ${total.toLocaleString()}`
      : `cluster ${state.cluster}: ${counts[Number(state.cluster)].toLocaleString()} / ${total.toLocaleString()}`;
    els.stats.textContent = `${state.dataset}, ${state.embedding}, ${selectedText}`;
    els.legend.innerHTML = "";
    for (let i = 0; i < counts.length; i += 1) {
      const item = document.createElement("div");
      item.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = palette[i % palette.length];
      const text = document.createElement("span");
      text.textContent = `C${i}: ${counts[i]}`;
      item.appendChild(swatch);
      item.appendChild(text);
      els.legend.appendChild(item);
    }
  }

  function nearestNode(clientX, clientY) {
    const rect = els.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const x = (clientX - rect.left) * dpr;
    const y = (clientY - rect.top) * dpr;
    const nodes = currentDataset().nodes;
    let best = -1;
    let bestDist = Infinity;
    const threshold = 12 * dpr;
    for (let i = 0; i < nodes.length; i += 1) {
      const pos = project(nodes[i]);
      const dx = pos.x - x;
      const dy = pos.y - y;
      const dist = dx * dx + dy * dy;
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    return bestDist <= threshold * threshold ? best : -1;
  }

  function updateTooltip(event) {
    state.hoverIndex = nearestNode(event.clientX, event.clientY);
    if (state.hoverIndex < 0) {
      els.tooltip.hidden = true;
      draw();
      return;
    }
    const node = currentDataset().nodes[state.hoverIndex];
    const label = currentLabels()[state.hoverIndex];
    els.tooltip.innerHTML = [
      `<strong>${state.dataset} node ${node.index}</strong>`,
      `id: ${node.id}`,
      `cluster: ${label}`,
      `lat/lon: ${node.lat.toFixed(5)}, ${node.lon.toFixed(5)}`,
      node.fwy ? `fwy: ${node.fwy} ${node.direction}` : "",
      node.county ? `county: ${node.county}` : "",
    ].filter(Boolean).join("<br>");
    const rect = els.canvas.getBoundingClientRect();
    els.tooltip.style.left = `${event.clientX - rect.left}px`;
    els.tooltip.style.top = `${event.clientY - rect.top}px`;
    els.tooltip.hidden = false;
    draw();
  }

  function togglePlay() {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
      els.play.textContent = "Play";
      return;
    }
    els.play.textContent = "Pause";
    state.timer = setInterval(() => {
      const maxDay = currentEmbedding().days.length - 1;
      state.day = state.day >= maxDay ? 0 : state.day + 1;
      els.day.value = String(state.day);
      draw();
    }, 650);
  }

  function bindEvents() {
    els.dataset.addEventListener("change", () => {
      state.dataset = els.dataset.value;
      state.day = 0;
      state.cluster = "all";
      refreshEmbeddingOptions();
      refreshClusterOptions();
      fitView();
      draw();
    });

    els.embedding.addEventListener("change", () => {
      state.embedding = els.embedding.value;
      state.day = 0;
      refreshDaySlider();
      draw();
    });

    els.cluster.addEventListener("change", () => {
      state.cluster = els.cluster.value;
      draw();
    });

    els.day.addEventListener("input", () => {
      state.day = Number(els.day.value);
      draw();
    });

    els.fade.addEventListener("input", () => {
      state.fade = Number(els.fade.value) / 100;
      draw();
    });

    els.play.addEventListener("click", togglePlay);
    els.fit.addEventListener("click", () => {
      fitView();
      draw();
    });

    els.canvas.addEventListener("mousemove", (event) => {
      if (state.dragging) {
        const dpr = window.devicePixelRatio || 1;
        state.tx += (event.clientX - state.lastX) * dpr;
        state.ty += (event.clientY - state.lastY) * dpr;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        els.tooltip.hidden = true;
        draw();
        return;
      }
      updateTooltip(event);
    });

    els.canvas.addEventListener("mouseleave", () => {
      state.hoverIndex = -1;
      els.tooltip.hidden = true;
      draw();
    });

    els.canvas.addEventListener("mousedown", (event) => {
      state.dragging = true;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
    });

    window.addEventListener("mouseup", () => {
      state.dragging = false;
    });

    els.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = els.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const mx = (event.clientX - rect.left) * dpr;
      const my = (event.clientY - rect.top) * dpr;
      const oldScale = state.scale;
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      state.scale = Math.max(0.5, Math.min(40, state.scale * factor));
      const ratio = state.scale / oldScale;
      state.tx = mx - (mx - state.tx) * ratio;
      state.ty = my - (my - state.ty) * ratio;
      draw();
    }, { passive: false });

    window.addEventListener("resize", resizeCanvas);
  }

  initControls();
  bindEvents();
  resizeCanvas();
}());
