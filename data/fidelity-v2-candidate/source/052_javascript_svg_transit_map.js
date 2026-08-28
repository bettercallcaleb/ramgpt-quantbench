/**
 * Dependency-free SVG transit-map renderer for browser use.
 *
 * Input model:
 * {
 *   stations: [{id, name, x, y, interchange?, accessible?}],
 *   lines: [{id, name, color, stations: ["a","b","c"]}],
 *   disruptions: [{lineId, fromStationId, toStationId, message}]
 * }
 *
 * The renderer keeps geometry in a logical coordinate system and uses
 * viewBox scaling so maps remain crisp at different viewport sizes.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, text = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) {
      node.setAttribute(key, String(value));
    }
  }
  if (text !== null) node.textContent = text;
  return node;
}

function assertModel(model) {
  if (!model || !Array.isArray(model.stations) || !Array.isArray(model.lines)) {
    throw new TypeError("model must contain stations[] and lines[]");
  }

  const ids = new Set();
  for (const station of model.stations) {
    if (!station.id || ids.has(station.id)) {
      throw new Error(`duplicate or missing station id: ${station.id}`);
    }
    if (!Number.isFinite(station.x) || !Number.isFinite(station.y)) {
      throw new Error(`station ${station.id} has invalid coordinates`);
    }
    ids.add(station.id);
  }

  for (const line of model.lines) {
    if (!line.id || !Array.isArray(line.stations) || line.stations.length < 2) {
      throw new Error(`invalid line: ${line.id}`);
    }
    for (const stationId of line.stations) {
      if (!ids.has(stationId)) {
        throw new Error(`line ${line.id} references unknown station ${stationId}`);
      }
    }
  }
}

function stationIndex(model) {
  return new Map(model.stations.map(s => [s.id, s]));
}

function linePath(line, byId) {
  const points = line.stations.map(id => byId.get(id));
  // Use simple polyline geometry. A production map could replace this with
  // orthogonal routing while retaining the same data and rendering contract.
  return points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
}

function normalizedSegment(a, b) {
  return [a, b].sort().join("::");
}

function disruptionSet(disruptions = []) {
  const set = new Set();
  for (const d of disruptions) {
    if (!d.lineId || !d.fromStationId || !d.toStationId) continue;
    set.add(`${d.lineId}::${normalizedSegment(d.fromStationId, d.toStationId)}`);
  }
  return set;
}

function drawLineSegments(layer, line, byId, disruptions) {
  for (let i = 0; i < line.stations.length - 1; i++) {
    const aId = line.stations[i];
    const bId = line.stations[i + 1];
    const a = byId.get(aId);
    const b = byId.get(bId);
    const key = `${line.id}::${normalizedSegment(aId, bId)}`;
    const disrupted = disruptions.has(key);

    const path = svg("path", {
      d: `M ${a.x} ${a.y} L ${b.x} ${b.y}`,
      fill: "none",
      stroke: disrupted ? "#555" : line.color,
      "stroke-width": disrupted ? 7 : 6,
      "stroke-linecap": "round",
      "stroke-dasharray": disrupted ? "5 5" : null,
      "data-line-id": line.id,
      "data-from": aId,
      "data-to": bId,
      tabindex: "0",
    });

    path.appendChild(svg("title", {}, disrupted
      ? `${line.name}: disruption between ${a.name} and ${b.name}`
      : `${line.name}: ${a.name} to ${b.name}`));

    layer.appendChild(path);
  }
}

function makeStationSymbol(station, linesServing) {
  const group = svg("g", {
    class: "station",
    transform: `translate(${station.x}, ${station.y})`,
    "data-station-id": station.id,
    tabindex: "0",
    role: "button",
    "aria-label": `${station.name}; lines ${linesServing.join(", ")}`,
  });

  const radius = station.interchange ? 7 : 5;
  group.appendChild(svg("circle", {
    cx: 0,
    cy: 0,
    r: radius,
    fill: "white",
    stroke: "#111",
    "stroke-width": 2,
  }));

  if (station.accessible) {
    group.appendChild(svg("circle", {
      cx: radius + 4,
      cy: -radius - 4,
      r: 2.5,
      fill: "#111",
      "aria-hidden": "true",
    }));
  }

  group.appendChild(svg("text", {
    x: radius + 6,
    y: 4,
    "font-size": 11,
    "font-family": "system-ui, sans-serif",
    "paint-order": "stroke",
    stroke: "white",
    "stroke-width": 3,
    "stroke-linejoin": "round",
    fill: "#111",
  }, station.name));

  return group;
}

function computeStationLines(model) {
  const map = new Map(model.stations.map(s => [s.id, []]));
  for (const line of model.lines) {
    for (const stationId of line.stations) {
      map.get(stationId).push(line.name);
    }
  }
  return map;
}

function renderLegend(model) {
  const group = svg("g", { transform: "translate(20, 20)" });
  model.lines.forEach((line, i) => {
    const y = i * 22;
    group.appendChild(svg("line", {
      x1: 0, y1: y, x2: 24, y2: y,
      stroke: line.color,
      "stroke-width": 6,
      "stroke-linecap": "round",
    }));
    group.appendChild(svg("text", {
      x: 32, y: y + 4,
      "font-size": 12,
      "font-family": "system-ui, sans-serif",
    }, line.name));
  });
  return group;
}

function fitBounds(model, padding = 40) {
  const xs = model.stations.map(s => s.x);
  const ys = model.stations.map(s => s.y);
  const minX = Math.min(...xs) - padding;
  const minY = Math.min(...ys) - padding;
  const maxX = Math.max(...xs) + padding;
  const maxY = Math.max(...ys) + padding;
  return [minX, minY, maxX - minX, maxY - minY];
}

function renderTransitMap(container, model) {
  assertModel(model);
  container.replaceChildren();

  const [x, y, width, height] = fitBounds(model);
  const root = svg("svg", {
    viewBox: `${x} ${y} ${width} ${height}`,
    width: "100%",
    height: "100%",
    role: "img",
    "aria-label": "Transit network map",
    style: "background:#f7f7f4",
  });

  const byId = stationIndex(model);
  const serving = computeStationLines(model);
  const disruptions = disruptionSet(model.disruptions);

  const lineLayer = svg("g", { class: "lines" });
  for (const line of model.lines) {
    drawLineSegments(lineLayer, line, byId, disruptions);
  }
  root.appendChild(lineLayer);

  const stationLayer = svg("g", { class: "stations" });
  for (const station of model.stations) {
    const symbol = makeStationSymbol(station, serving.get(station.id));
    symbol.addEventListener("click", () => {
      container.dispatchEvent(new CustomEvent("stationselect", {
        detail: { stationId: station.id, station },
        bubbles: true,
      }));
    });
    symbol.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        symbol.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      }
    });
    stationLayer.appendChild(symbol);
  }
  root.appendChild(stationLayer);

  root.appendChild(renderLegend(model));
  container.appendChild(root);
  return root;
}

// Optional pan/zoom controller that does not mutate station geometry.
function attachPanZoom(svgElement) {
  const initial = svgElement.viewBox.baseVal;
  let box = {
    x: initial.x,
    y: initial.y,
    width: initial.width,
    height: initial.height,
  };
  let dragging = false;
  let previous = null;

  function apply() {
    svgElement.setAttribute("viewBox", `${box.x} ${box.y} ${box.width} ${box.height}`);
  }

  svgElement.addEventListener("wheel", event => {
    event.preventDefault();
    const factor = event.deltaY > 0 ? 1.12 : 0.89;
    const rect = svgElement.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width;
    const py = (event.clientY - rect.top) / rect.height;
    const worldX = box.x + px * box.width;
    const worldY = box.y + py * box.height;

    const newWidth = box.width * factor;
    const newHeight = box.height * factor;
    box.x = worldX - px * newWidth;
    box.y = worldY - py * newHeight;
    box.width = newWidth;
    box.height = newHeight;
    apply();
  }, { passive: false });

  svgElement.addEventListener("pointerdown", event => {
    dragging = true;
    previous = { x: event.clientX, y: event.clientY };
    svgElement.setPointerCapture(event.pointerId);
  });

  svgElement.addEventListener("pointermove", event => {
    if (!dragging || !previous) return;
    const rect = svgElement.getBoundingClientRect();
    const dx = (event.clientX - previous.x) * box.width / rect.width;
    const dy = (event.clientY - previous.y) * box.height / rect.height;
    box.x -= dx;
    box.y -= dy;
    previous = { x: event.clientX, y: event.clientY };
    apply();
  });

  svgElement.addEventListener("pointerup", event => {
    dragging = false;
    previous = null;
    svgElement.releasePointerCapture(event.pointerId);
  });

  return {
    reset() {
      const v = svgElement.viewBox.baseVal;
      box = { x: initial.x, y: initial.y, width: initial.width, height: initial.height };
      apply();
    },
  };
}

// Demonstration data used when the module is loaded directly in a page with
// <div id="map"></div>.
const demoNetwork = {
  stations: [
    { id: "harbor", name: "Harbor", x: 80, y: 180, accessible: true },
    { id: "market", name: "Market", x: 170, y: 180 },
    { id: "central", name: "Central", x: 260, y: 180, interchange: true, accessible: true },
    { id: "museum", name: "Museum", x: 350, y: 120 },
    { id: "hill", name: "Hill Park", x: 440, y: 70 },
    { id: "river", name: "Rivergate", x: 350, y: 240 },
    { id: "university", name: "University", x: 440, y: 300, accessible: true },
  ],
  lines: [
    { id: "blue", name: "Blue Line", color: "#2674c8",
      stations: ["harbor", "market", "central", "museum", "hill"] },
    { id: "orange", name: "Orange Line", color: "#d66b19",
      stations: ["central", "river", "university"] },
  ],
  disruptions: [
    { lineId: "blue", fromStationId: "museum", toStationId: "hill",
      message: "Track maintenance" },
  ],
};

if (typeof window !== "undefined") {
  window.TransitMap = { renderTransitMap, attachPanZoom, demoNetwork };
  const container = document.getElementById("map");
  if (container) {
    const map = renderTransitMap(container, demoNetwork);
    attachPanZoom(map);
  }
}
