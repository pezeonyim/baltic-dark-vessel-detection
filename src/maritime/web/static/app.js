/* Maritime Watch: map app v1.
 * MapLibre basemap + deck.gl overlay. Vessel data comes from the backend; ship names and
 * other AIS text are untrusted broadcasts, so they are only ever set via textContent.
 */
(() => {
  "use strict";

  const GROUPS = [
    { key: "nato", label: "NATO-flagged", color: [57, 135, 229] },
    { key: "russia", label: "Russian-flagged", color: [217, 89, 38] },
    { key: "china", label: "Chinese-flagged", color: [25, 158, 112] },
    { key: "other", label: "Other flags", color: [168, 167, 160] },
    { key: "unknown", label: "Unknown flag", color: [107, 106, 101] },
  ];
  const GROUP = Object.fromEntries(GROUPS.map((g) => [g.key, g]));
  const SURFACE = [11, 14, 18];
  const CRITICAL = [230, 103, 103];
  // Navigation arrow pointing north; rotated by course. mask: true lets deck.gl tint it.
  const ARROW_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">' +
    '<path d="M32 4 L54 58 L32 46 L10 58 Z" fill="#fff"/></svg>';
  const ARROW = { url: "data:image/svg+xml;charset=utf-8," + encodeURIComponent(ARROW_SVG),
                  width: 64, height: 64, anchorY: 32, mask: true };     // watch-listed ring (status color; legend shows ⚠ + label)
  const BASEMAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
  // Satellite radar detections: diamonds, so they never look like AIS ships.
  const SAR_KINDS = [
    { key: "dark_high", label: "Dark ship, high confidence", color: [255, 140, 30], size: 18 },
    { key: "dark_possible", label: "Dark ship, possible", color: [240, 196, 70], size: 15 },
    { key: "no_coverage", label: "Ship seen, no AIS coverage there", color: [150, 150, 150], size: 11 },
    { key: "small", label: "Small craft, no AIS (not required to carry it)", color: [110, 110, 110], size: 7 },
    { key: "ais_gap", label: "Ship seen, its AIS reports were briefly missed", color: [150, 210, 190], size: 10, hollow: true },
    { key: "ais", label: "Ship seen, matched to its AIS", color: [110, 200, 235], size: 10, hollow: true },
  ];
  // Evidence check: what the AIS record says about each dark candidate
  const fmtShip = (d) => (d.ev_name && d.ev_name.trim()) || (d.ev_mmsi ? `MMSI ${d.ev_mmsi}` : "an unnamed ship");
  const shipLen = (d) => (d.ev_length ? ` (${Math.round(d.ev_length)} m)` : "");
  const EVIDENCE_TEXT = {
    went_silent: (d) => `Possibly ${fmtShip(d)}${shipLen(d)}: its AIS went silent for ${d.ev_gap_h} h around this image, unusually long for this area`,
    usual_gap: (d) => `Possibly ${fmtShip(d)}${shipLen(d)}, silent ${d.ev_gap_h} h; silences that long are common here`,
    crowded: () => "Several ships that went quiet fit this spot (busy anchorage); identity unclear",
    weak_link: (d) => `A ship that went quiet passed nearby (${fmtShip(d)}), weak match`,
    unexplained: () => "No AIS ship fits: a ship that never transmits, or a false alarm",
  };
  const SAR_KIND = Object.fromEntries(SAR_KINDS.map((k) => [k.key, k]));
  const sarKey = (d) => (d.status === "dark" ? (d.tier === "high" ? "dark_high" : "dark_possible") : d.status);
  const SAR_DAYS = 3;
  const DIAMOND = (hollow) => ({
    url: "data:image/svg+xml;charset=utf-8," + encodeURIComponent(
      '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">' +
      (hollow ? '<path d="M32 6 L58 32 L32 58 L6 32 Z" fill="none" stroke="#fff" stroke-width="9"/>'
              : '<path d="M32 4 L60 32 L32 60 L4 32 Z" fill="#fff"/>') + "</svg>"),
    width: 64, height: 64, mask: true,
  });
  const DIAMOND_FILLED = DIAMOND(false), DIAMOND_HOLLOW = DIAMOND(true);
  // Motion: between reports, ships move along their reported course at their reported
  // speed (dead reckoning), for at most DR_MAX_MIN minutes after the last report.
  const DR_MAX_MIN = 10;
  const MIN_MOVING_KN = 0.5;
  const CORRECTION_MS = 2000;       // glide to a new report instead of jumping
  const SNAP_DEG = 0.05;            // corrections larger than ~5 km snap instead
  const TRAIL_MIN = 10;             // wake trail length
  const TRAIL_MIN_ZOOM = 6;         // trails only when zoomed in enough to read them
  const FRAME_MS = 125;             // ~8 frames per second is smooth enough for ships
  const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FALLBACK_STYLE = {
    version: 8, sources: {},
    layers: [{ id: "bg", type: "background", paint: { "background-color": "#10151b" } }],
  };

  const state = {
    vessels: new Map(),          // mmsi -> vessel record
    hidden: new Set(),           // hidden flag groups
    showQuiet: true,
    officialOnly: false,
    listedOnly: false,
    quietAfter: 30,
    selected: null,              // mmsi
    track: null,                 // [[lon, lat, t, sog], ...]
    lastMessage: 0,
    visibleList: [],
    frame: 0,
    sar: [],                     // radar detections (last SAR_DAYS days)
    risk: new Map(),             // mmsi -> risk row
    sarHidden: new Set(),        // hidden SAR kinds
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  };
  const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  const fmtAge = (min) => {
    if (min == null) return "unknown";
    if (min < 1) return "just now";
    if (min < 60) return `${Math.round(min)} min ago`;
    if (min < 48 * 60) return `${(min / 60).toFixed(1)} h ago`;
    return `${Math.round(min / 1440)} days ago`;
  };
  const nameOf = (v) => (v.vessel_name && v.vessel_name.trim()) || `MMSI ${v.mmsi}`;
  const isOfficial = (v) => v.is_military || v.is_law_enforcement;
  // Arrows for vessels under way with a known course; dots for stationary or quiet ones.
  const isMoving = (v) => !v.quiet && v.cog != null && (v.sog ?? 0) >= MIN_MOVING_KN;

  // Ages are recomputed in the browser so "last seen" stays current between updates.
  function ageMin(v) {
    const t = Date.parse(v.ts.endsWith("Z") ? v.ts : v.ts + "Z");
    return (Date.now() - t) / 60000;
  }

  const tsMs = (ts) => Date.parse(ts.endsWith("Z") ? ts : ts + "Z");

  // Where dead reckoning puts the vessel at time `now` (ms), from its last report.
  function extrapolate(v, now) {
    if (REDUCED_MOTION || v.quiet || v.sog == null || v.sog < MIN_MOVING_KN || v.cog == null) {
      return [v.lon, v.lat];
    }
    const minutes = Math.min(Math.max((now - v._t) / 60000, 0), DR_MAX_MIN);
    const km = (v.sog * 1.852 * minutes) / 60;
    const rad = (v.cog * Math.PI) / 180;
    const lat = v.lat + (km * Math.cos(rad)) / 111.32;
    const lon = v.lon + (km * Math.sin(rad)) / (111.32 * Math.cos((v.lat * Math.PI) / 180));
    return [lon, lat];
  }

  // Displayed position = dead-reckoned position + a correction that fades out.
  function displayPos(v, now) {
    const p = extrapolate(v, now);
    if (v._corr) {
      const f = 1 - (now - v._corr.start) / CORRECTION_MS;
      if (f > 0) return [p[0] + v._corr.dLon * f, p[1] + v._corr.dLat * f];
      v._corr = null;
    }
    return p;
  }

  function visible() {
    const out = [];
    for (const v of state.vessels.values()) {
      v.age_min = (Date.now() - v._t) / 60000;
      v.quiet = v.age_min > state.quietAfter;
      if (state.hidden.has(v.flag_group)) continue;
      if (!state.showQuiet && v.quiet) continue;
      if (state.officialOnly && !isOfficial(v)) continue;
      if (state.listedOnly && !v.listed) continue;
      out.push(v);
    }
    // draw order: quiet, then live, then watch-listed on top
    const rank = (v) => (v.listed ? 2 : v.quiet ? 0 : 1);
    out.sort((a, b) => rank(a) - rank(b));
    return out;
  }

  // ---------------------------------------------------------------- map + layers
  const map = new maplibregl.Map({
    container: "map", style: FALLBACK_STYLE, center: [18, 58], zoom: 4.3,
    attributionControl: { compact: true },
    hash: true,                      // #zoom/lat/lon in the URL: views can be bookmarked
  });
  // ---------------------------------------------------------------- zoom + cursor position
  // Degrees and decimal minutes, as on nautical charts: 53° 6.04 N
  function dm(value, pos, neg) {
    const hemi = value >= 0 ? pos : neg;
    const a = Math.abs(value);
    const deg = Math.floor(a);
    const min = (a - deg) * 60;
    return `${deg}° ${min.toFixed(2)} ${hemi}`;
  }

  class ZoomPositionControl {
    onAdd(m) {
      this.map = m;
      const root = el("div", "mw-ctrl");
      this.coords = el("button", "mw-coords");
      this.coords.type = "button";
      this.coords.hidden = true;
      this.coords.title = "Click to copy coordinates";
      this.coords.setAttribute("aria-live", "off");
      const zoom = el("div", "mw-zoom");
      const plus = el("button", "mw-zbtn", "+");
      const minus = el("button", "mw-zbtn", "−");
      plus.type = minus.type = "button";
      plus.setAttribute("aria-label", "Zoom in");
      minus.setAttribute("aria-label", "Zoom out");
      this.level = el("div", "mw-zlevel");
      this.level.setAttribute("aria-label", "Zoom level");
      zoom.append(plus, this.level, minus);
      root.append(this.coords, zoom);

      plus.addEventListener("click", () => m.zoomIn());
      minus.addEventListener("click", () => m.zoomOut());
      const showZoom = () => { this.level.textContent = Math.round(m.getZoom()); };
      m.on("zoom", showZoom);
      showZoom();

      m.on("mousemove", (e) => this.show(e.lngLat));
      m.getCanvasContainer().addEventListener("mouseleave", () => {
        if (!this.pinned) this.coords.hidden = true;
      });
      this.coords.addEventListener("click", async () => {
        if (!this.last) return;
        const text = `${this.last.lat.toFixed(6)}, ${this.last.lng.toFixed(6)}`;
        try { await navigator.clipboard.writeText(text); this.flash("Copied"); }
        catch { this.flash("Copy unavailable"); }
      });
      return root;
    }
    show(ll) {
      this.last = ll;
      const lng = ((ll.lng + 540) % 360) - 180;          // wrap to -180..180
      this.coords.replaceChildren(
        el("span", null, dm(ll.lat, "N", "S")),
        el("span", null, dm(lng, "E", "W")),
        el("span", "mw-dec", `(${ll.lat.toFixed(6)}, ${lng.toFixed(6)})`),
      );
      this.coords.hidden = false;
    }
    flash(msg) {
      const tag = el("span", "mw-flash", msg);
      this.coords.append(tag);
      setTimeout(() => tag.remove(), 1200);
    }
    onRemove() {}
  }
  map.addControl(new ZoomPositionControl(), "bottom-right");
  // Load the real basemap if reachable; keep the plain background otherwise.
  fetch(BASEMAP, { signal: AbortSignal.timeout(5000) })
    .then((r) => (r.ok ? r.json() : Promise.reject()))
    .then((style) => map.setStyle(style))
    .catch(() => console.info("basemap unavailable; using plain background"));

  const overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
  map.addControl(overlay);

  function render() {
    state.visibleList = visible();
    draw();
    renderCounts();
  }

  function draw() {
    const now = Date.now();
    const frame = ++state.frame;
    const data = state.visibleList;
    const layers = [];
    const sarShown = state.sar.filter((d) => !state.sarHidden.has(sarKey(d)));
    const sarLayer = (id, data) => new deck.IconLayer({
      id, data, pickable: true, onHover: onSarHover,
      alphaCutoff: 0,              // the hollow diamond's empty centre still counts for hover
      getIcon: (d) => (SAR_KIND[sarKey(d)].hollow ? DIAMOND_HOLLOW : DIAMOND_FILLED),
      getPosition: (d) => [d.lon, d.lat],
      getSize: (d) => SAR_KIND[sarKey(d)].size, sizeUnits: "pixels",
      // older than a day: fainter (the ship has long moved on)
      getColor: (d) => [...SAR_KIND[sarKey(d)].color, d.age_h > 24 ? 110 : 235],
    });
    // matched / no-coverage detections sit under the live ships; dark candidates go on top
    layers.push(sarLayer("sar-under", sarShown.filter((d) => d.status !== "dark")));

    if (map.getZoom() >= TRAIL_MIN_ZOOM && !REDUCED_MOTION) {
      const trails = [];
      for (const v of data) {
        if (v.quiet || !v._trail || !v._trail.length || (v.sog ?? 0) < MIN_MOVING_KN) continue;
        trails.push({ v, path: [...v._trail.map((p) => [p[0], p[1]]), displayPos(v, now)] });
      }
      layers.push(new deck.PathLayer({
        id: "trails",
        data: trails,
        getPath: (d) => d.path,
        getColor: (d) => [...(GROUP[d.v.flag_group] || GROUP.unknown).color, 110],
        getWidth: 1.5, widthUnits: "pixels", capRounded: true, jointRounded: true,
        updateTriggers: { getPath: frame },
      }));
    }

    const moving = data.filter(isMoving);
    const still = data.filter((v) => !isMoving(v));
    const ringColor = (v) => (v.listed ? [...CRITICAL, 255] : isOfficial(v) ? [255, 255, 255, 255] : [...SURFACE, 200]);
    const ringWidth = (v) => (v.listed ? 2.5 : isOfficial(v) ? 2 : 1);
    const fill = (v) => [...(GROUP[v.flag_group] || GROUP.unknown).color, v.quiet ? 80 : 235];
    const common = { pickable: true, onHover, onClick: (info) => info.object && select(info.object.mmsi, false) };

    // Stationary / quiet vessels: dots.
    layers.push(new deck.ScatterplotLayer({
      id: "vessels-still", data: still, ...common,
      getPosition: (v) => displayPos(v, now),
      getRadius: (v) => (v.listed ? 6 : isOfficial(v) ? 5 : v.quiet ? 3 : 4),
      radiusUnits: "pixels",
      getFillColor: fill,
      stroked: true,
      // Watch-listed: red ring. Military / law enforcement: white ring.
      getLineColor: ringColor, getLineWidth: ringWidth, lineWidthUnits: "pixels",
      updateTriggers: { getPosition: frame },
    }));

    // Moving vessels: rings (watch-listed / official only), then a dark outline arrow and
    // the coloured arrow on top, rotated to the course over ground.
    const ringed = moving.filter((v) => v.listed || isOfficial(v));
    layers.push(new deck.ScatterplotLayer({
      id: "vessels-rings", data: ringed,
      getPosition: (v) => displayPos(v, now),
      getRadius: (v) => (v.listed ? 10 : 9), radiusUnits: "pixels",
      filled: false, stroked: true,
      getLineColor: ringColor, getLineWidth: ringWidth, lineWidthUnits: "pixels",
      updateTriggers: { getPosition: frame },
    }));
    const arrow = (id, size, color) => new deck.IconLayer({
      id, data: moving, ...(id === "vessels-arrows" ? common : {}),
      getIcon: () => ARROW,
      getPosition: (v) => displayPos(v, now),
      getAngle: (v) => -v.cog,                 // deck.gl angles are counter-clockwise
      getSize: (v) => size + (v.listed || isOfficial(v) ? 2 : 0),
      sizeUnits: "pixels",
      getColor: color,
      billboard: false,
      updateTriggers: { getPosition: frame },
    });
    layers.push(arrow("vessels-arrows-outline", 17, [...SURFACE, 230]));
    layers.push(arrow("vessels-arrows", 13, fill));

    if (state.track && state.track.length > 1) {
      layers.push(new deck.PathLayer({
        id: "track",
        data: [{ path: state.track.map((p) => [p[0], p[1]]) }],
        getPath: (d) => d.path,
        getColor: [242, 243, 244, 200],
        getWidth: 2,
        widthUnits: "pixels",
        jointRounded: true, capRounded: true,
      }));
    }
    const sel = state.selected && state.vessels.get(state.selected);
    if (sel) {
      layers.push(new deck.ScatterplotLayer({
        id: "selected",
        data: [sel],
        getPosition: (v) => displayPos(v, now),
        getRadius: 11, radiusUnits: "pixels",
        filled: false, stroked: true,
        getLineColor: [242, 243, 244, 255], getLineWidth: 2, lineWidthUnits: "pixels",
        updateTriggers: { getPosition: frame },
      }));
    }
    layers.push(sarLayer("sar-dark", sarShown.filter((d) => d.status === "dark")));
    overlay.setProps({ layers });
  }

  // Animation loop: redraw positions only (filters and counts change on render()).
  let lastFrame = 0;
  function tick(t) {
    if (t - lastFrame >= FRAME_MS && state.visibleList.length) {
      lastFrame = t;
      draw();
    }
    requestAnimationFrame(tick);
  }

  // ---------------------------------------------------------------- panel
  function buildLegend() {
    const ul = $("legend");
    for (const g of GROUPS) {
      const li = el("li");
      li.dataset.group = g.key;
      li.tabIndex = 0;
      li.setAttribute("role", "button");
      li.setAttribute("aria-pressed", "true");
      const sw = el("span", "swatch");
      sw.style.background = rgb(g.color);
      li.append(sw, el("span", null, g.label), el("span", "count", "–"));
      const toggle = () => {
        state.hidden.has(g.key) ? state.hidden.delete(g.key) : state.hidden.add(g.key);
        li.classList.toggle("off", state.hidden.has(g.key));
        li.setAttribute("aria-pressed", String(!state.hidden.has(g.key)));
        render();
      };
      li.addEventListener("click", toggle);
      li.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), toggle()));
      ul.append(li);
    }
  }

  function buildSarLegend() {
    const ul = $("sar-legend");
    for (const k of SAR_KINDS) {
      const li = el("li");
      li.dataset.kind = k.key;
      li.tabIndex = 0;
      li.setAttribute("role", "button");
      li.setAttribute("aria-pressed", "true");
      const sw = el("span", "swatch diamond" + (k.hollow ? " hollow" : ""));
      sw.style.background = rgb(k.color);
      sw.style.color = rgb(k.color);
      li.append(sw, el("span", null, k.label), el("span", "count", "–"));
      const toggle = () => {
        state.sarHidden.has(k.key) ? state.sarHidden.delete(k.key) : state.sarHidden.add(k.key);
        li.classList.toggle("off", state.sarHidden.has(k.key));
        li.setAttribute("aria-pressed", String(!state.sarHidden.has(k.key)));
        draw();
      };
      li.addEventListener("click", toggle);
      li.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), toggle()));
      ul.append(li);
    }
  }

  async function loadSar() {
    try {
      const r = await (await fetch(`/api/sar?days=${SAR_DAYS}`)).json();
      state.sar = r.detections || [];
      const counts = {};
      for (const d of state.sar) counts[sarKey(d)] = (counts[sarKey(d)] || 0) + 1;
      for (const li of $("sar-legend").children) {
        li.querySelector(".count").textContent = (counts[li.dataset.kind] || 0).toLocaleString();
      }
      const scenes = r.scenes || [];
      if (scenes.length) {
        const newest = scenes[scenes.length - 1];
        const ageMinutes = (Date.now() - Date.parse(newest.start + "Z")) / 60000;
        const ac = r.all_counts || {};
        const hidden = (ac.static || 0) + (ac.outside_area || 0);
        $("sar-meta").textContent =
          `${scenes.length} radar image${scenes.length === 1 ? "" : "s"} in the last ${SAR_DAYS} days · newest ${fmtAge(ageMinutes)}` +
          (hidden ? ` · ${hidden} fixed structures or outside-area hits hidden` : "");
      } else {
        $("sar-meta").textContent = `No radar images in the last ${SAR_DAYS} days`;
      }
      draw();
    } catch {
      $("sar-meta").textContent = "Radar detections unavailable";
    }
  }

  // ---------------------------------------------------------------- risk
  // shown as rank, not probability: training weights the rare listed class up, so raw scores are inflated
  const fmtTop = (p) => (p == null ? "–" : p < 1 ? `${p.toFixed(1)}%` : `${Math.round(p)}%`);
  async function loadRisk() {
    try {
      const unl = $("risk-unlisted").checked;
      const r = await (await fetch(`/api/risk?limit=15&unlisted=${unl}`)).json();
      const all = await (await fetch("/api/risk?limit=200")).json();
      state.risk = new Map((all.vessels || []).map((x) => [x.mmsi, x]));
      const ul = $("risk-list");
      ul.replaceChildren();
      for (const x of r.vessels || []) {
        const li = el("li");
        li.tabIndex = 0;
        const name = el("span", null, (x.vessel_name && x.vessel_name.trim()) || `MMSI ${x.mmsi}`);
        const sc = el("span", "score", `top ${fmtTop(x.risk_top_pct)}`);
        const why = el("span", "why", [x.listed_label ? `⚠ ${x.listed_label}` : "", x.reasons, x.context]
          .filter(Boolean).join(" · "));
        li.append(name, sc, why);
        const go = () => (state.vessels.has(x.mmsi) ? select(x.mmsi, true) : null);
        li.addEventListener("click", go);
        li.addEventListener("keydown", (e) => e.key === "Enter" && go());
        ul.append(li);
      }
      const rep = r.report;
      $("risk-meta").textContent = rep
        ? `How much a ship's behaviour over 7 days resembles ships on sanctions / shadow-fleet lists. A lead, not ` +
          `an accusation. In testing, ${Math.round(rep.precision_top50 * 100)}% of the top 50 were listed ships ` +
          `(${(rep.base_rate * 100).toFixed(1)}% of all ships are).`
        : "Risk scores not computed yet";
    } catch {
      $("risk-meta").textContent = "Risk scores unavailable";
    }
  }

  function onSarHover(info) {
    const tip = $("tooltip");
    if (!info.object) { tip.hidden = true; return; }
    const d = info.object;
    const k = SAR_KIND[sarKey(d)];
    tip.replaceChildren(
      el("strong", null, d.status === "ais" ? (d.vessel_name && d.vessel_name.trim()) || `MMSI ${d.mmsi}`
        : d.status === "ais_gap" ? fmtShip(d) : "Radar detection"),
      el("span", "muted", k.label),
      el("span", "muted", `Sentinel-1 (${d.platform}) · ${String(d.time).replace("T", " ").slice(0, 16)} UTC · ${fmtAge(d.age_h * 60)}`),
      el("span", "muted", `Confidence ${Math.round(d.score * 100)}% · ` +
        (d.length_rough_m ? `roughly ${Math.round(d.length_rough_m / 10) * 10} m long` : "small craft")),
    );
    if (d.status === "dark") {
      tip.append(el("span", null, "No AIS signal from this ship although the area is normally covered"));
      const t = EVIDENCE_TEXT[d.verdict];
      if (t) tip.append(el("span", null, t(d)));
    }
    if (d.status === "ais_gap") {
      tip.append(el("span", null, `Its AIS reports were missed for ${d.ev_gap_h} h around this image; ` +
        "the radar saw it on its own track, so it is not dark"));
    }
    if (d.ev_watchlist) tip.append(el("span", null, `⚠ ${fmtShip(d)}: ${d.ev_watchlist}`));
    tip.style.left = `${info.x + 14}px`;
    tip.style.top = `${info.y + 14}px`;
    tip.hidden = false;
  }

  function renderCounts() {
    const counts = Object.fromEntries(GROUPS.map((g) => [g.key, 0]));
    let live = 0, quiet = 0, listed = 0;
    for (const v of state.vessels.values()) {
      if (state.officialOnly && !isOfficial(v)) continue;
      if (state.listedOnly && !v.listed) continue;
      if (v.listed && (state.showQuiet || !v.quiet)) listed++;
      if (v.quiet) { if (!state.showQuiet) continue; quiet++; } else live++;
      counts[v.flag_group] = (counts[v.flag_group] || 0) + 1;
    }
    $("n-live").textContent = live.toLocaleString();
    $("n-quiet").textContent = quiet.toLocaleString();
    $("n-listed").textContent = listed.toLocaleString();
    for (const li of $("legend").children) {
      li.querySelector(".count").textContent = (counts[li.dataset.group] || 0).toLocaleString();
    }
  }

  function renderResults() {
    const q = $("search").value.trim().toLowerCase();
    const ul = $("results");
    ul.replaceChildren();
    if (q.length < 2) return;
    const hits = [];
    const numeric = /^\d+$/.test(q);   // numbers match the start of MMSI / IMO
    for (const v of state.vessels.values()) {
      const match = numeric
        ? String(v.mmsi).startsWith(q) || String(v.imo || "").startsWith(q)
        : (v.vessel_name || "").toLowerCase().includes(q);
      if (match) hits.push(v);
      if (hits.length >= 12) break;
    }
    if (!hits.length) { ul.append(el("li", "muted", "No matches")); return; }
    for (const v of hits) {
      const li = el("li");
      li.tabIndex = 0;
      const sw = el("span", "swatch");
      sw.style.background = rgb((GROUP[v.flag_group] || GROUP.unknown).color);
      const name = el("span", null, nameOf(v));
      const meta = el("span", "meta", v.flag_iso2 || "");
      li.append(sw, name, meta);
      li.addEventListener("click", () => select(v.mmsi, true));
      li.addEventListener("keydown", (e) => e.key === "Enter" && select(v.mmsi, true));
      ul.append(li);
    }
  }

  // ---------------------------------------------------------------- tooltip + detail
  function onHover(info) {
    const tip = $("tooltip");
    if (!info.object) { tip.hidden = true; return; }
    const v = info.object;
    tip.replaceChildren(
      el("strong", null, nameOf(v)),
      el("span", "muted", `${v.flag_country || "Unknown flag"} · ${v.type_name || "Unknown type"}`),
      el("span", "muted", v.quiet ? `Last seen ${fmtAge(v.age_min)}` : `${v.sog ?? "–"} kn · seen ${fmtAge(v.age_min)}`),
    );
    if (v.listed) tip.append(el("span", null, `⚠ ${v.listed_category_label || "On a watchlist"}`));
    tip.style.left = `${info.x + 14}px`;
    tip.style.top = `${info.y + 14}px`;
    tip.hidden = false;
  }

  async function select(mmsi, fly) {
    const v = state.vessels.get(mmsi);
    if (!v) return;
    state.selected = mmsi;
    state.track = null;
    if (fly) map.flyTo({ center: [v.lon, v.lat], zoom: Math.max(map.getZoom(), 8) });
    showDetail(v);
    render();
    try {
      const r = await fetch(`/api/vessels/${mmsi}/track?hours=12`);
      const d = r.ok ? await r.json() : { points: [] };
      if (state.selected !== mmsi) return;
      state.track = d.points;
      $("d-track").textContent = d.points.length > 1
        ? `Track: ${d.points.length} positions in the last 12 h`
        : "No track in the last 12 h";
      render();
    } catch {
      $("d-track").textContent = "Track unavailable";
    }
  }

  function showDetail(v) {
    $("d-name").textContent = nameOf(v);
    const sub = $("d-sub");
    sub.replaceChildren();
    const g = GROUP[v.flag_group] || GROUP.unknown;
    const pill = el("span", "pill");
    const dot = el("span", "pill-dot");
    dot.style.background = rgb(g.color);
    pill.append(dot, document.createTextNode(g.label));
    sub.append(pill);
    if (v.is_military) sub.append(" ", el("span", "pill", "Military"));
    if (v.is_law_enforcement) sub.append(" ", el("span", "pill", "Law enforcement"));
    const rows = [
      ["Flag", v.flag_country || "Unknown"],
      ["Type", v.type_name || "Unknown"],
      ["MMSI", v.mmsi],
      ["IMO", v.imo || "–"],
      ["Speed", v.sog != null ? `${v.sog} kn` : "–"],
      ["Course", v.cog != null ? `${Math.round(v.cog)}°` : "–"],
      ["Length", v.length ? `${v.length} m` : "–"],
      ["Last seen", fmtAge(v.age_min ?? ageMin(v))],
      ["Position", `${v.lat.toFixed(4)}, ${v.lon.toFixed(4)}`],
    ];
    const rk = state.risk.get(v.mmsi);
    if (rk) rows.push(["Risk", `top ${fmtTop(rk.risk_top_pct)} of ships this week: ${rk.reasons || "no single strong reason"}`]);
    const box = $("d-watch");
    box.replaceChildren();
    box.hidden = !v.listed_category;
    box.classList.toggle("neutral", !v.listed);
    if (v.listed_category) {
      box.append(el("strong", null, v.listed
        ? `⚠ ${v.listed_category_label || "On a watchlist"}`
        : v.listed_category_label || "On a public record"));
      if (!v.listed && v.listed_category === "detention") {
        box.append(el("span", "muted", "Safety inspection record from port state control, not a sanction. "));
      }
      box.append(el("span", "muted",
        `Matched by ${v.listed_match === "imo" ? "IMO number" : "MMSI only (weaker: MMSIs change)"}` +
        (v.listed_name && v.listed_name !== v.vessel_name ? `, listed as ${v.listed_name}` : "")));
      const ul = el("ul");
      for (const src of v.listed_sources || []) ul.append(el("li", null, src));
      box.append(ul);
      if (v.listed_url && v.listed_url.startsWith("https://www.opensanctions.org/")) {
        const a = el("a", null, "View on OpenSanctions ↗");
        a.href = v.listed_url; a.target = "_blank"; a.rel = "noopener noreferrer";
        box.append(a);
      }
    }
    const dl = $("d-fields");
    dl.replaceChildren();
    for (const [k, val] of rows) dl.append(el("dt", null, k), el("dd", null, val));
    $("d-track").textContent = "Loading track…";
    $("detail").hidden = false;
  }

  function closeDetail() {
    state.selected = null;
    state.track = null;
    $("detail").hidden = true;
    render();
  }

  // ---------------------------------------------------------------- data
  function merge(list) {
    const now = Date.now();
    for (const v of list) {
      v._t = tsMs(v.ts);
      v.age_min = (now - v._t) / 60000;
      v.quiet = v.age_min > state.quietAfter;
      const old = state.vessels.get(v.mmsi);
      if (old) {
        if (v._t <= old._t) continue;                       // stale or duplicate report
        // Trail: keep the previous real reports from the last TRAIL_MIN minutes.
        v._trail = (old._trail || []).concat([[old.lon, old.lat, old._t]])
          .filter((p) => now - p[2] <= TRAIL_MIN * 60000).slice(-40);
        // Glide from where the dot is drawn now to the new estimate, unless it is far off.
        const shown = displayPos(old, now);
        const next = extrapolate(v, now);
        const dLon = shown[0] - next[0], dLat = shown[1] - next[1];
        if (!REDUCED_MOTION && Math.abs(dLon) < SNAP_DEG && Math.abs(dLat) < SNAP_DEG) {
          v._corr = { dLon, dLat, start: now };
        }
      } else if (state.recent && state.recent[v.mmsi]) {
        v._trail = state.recent[v.mmsi].filter((p) => p[2] < v._t);
      }
      state.vessels.set(v.mmsi, v);
    }
    if (state.selected && state.vessels.has(state.selected)) showDetail(state.vessels.get(state.selected));
  }

  function setStatus(kind, text) {
    $("status-dot").className = `dot ${kind}`;
    $("status-text").textContent = text;
  }

  // used while the live connection is down or refused (public mode caps live connections)
  async function refreshVessels() {
    try {
      const r = await fetch("/api/vessels?max_age_hours=1");
      if (!r.ok) return;
      merge(await r.json());
      state.lastMessage = Date.now();
      render();
    } catch { /* keep the last known picture */ }
  }

  function connect(delay = 1000) {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      state.lastMessage = Date.now();
      if (msg.type === "update") { merge(msg.vessels); render(); }
    };
    ws.onopen = () => { delay = 1000; setStatus("live", "Live"); };
    ws.onclose = (e) => {
      const busy = e.code === 1013;
      setStatus(busy ? "stale" : "down", busy ? "Busy: updating every 30 s" : "Disconnected, retrying…");
      refreshVessels();
      const next = busy ? 30000 : delay;
      setTimeout(() => connect(Math.min(next * 2, 30000)), next);
    };
  }

  setInterval(() => {
    if (!state.lastMessage) return;
    const s = Math.round((Date.now() - state.lastMessage) / 1000);
    if (s > 30) setStatus("stale", `No update for ${s}s`);
    else setStatus("live", "Live");
    render();                               // refresh ages / quiet state
  }, 15000);

  async function init() {
    buildLegend();
    buildSarLegend();
    $("f-quiet").addEventListener("change", (e) => { state.showQuiet = e.target.checked; render(); });
    $("f-official").addEventListener("change", (e) => { state.officialOnly = e.target.checked; render(); });
    $("f-listed").addEventListener("change", (e) => { state.listedOnly = e.target.checked; render(); });
    $("search").addEventListener("input", renderResults);
    $("detail-close").addEventListener("click", closeDetail);
    document.addEventListener("keydown", (e) => e.key === "Escape" && closeDetail());

    const sres = await fetch("/api/summary").catch(() => null);
    if (!sres || !sres.ok) {
      setStatus("down", sres && sres.status === 429 ? "Too many requests: reload in a minute" : "Map temporarily unavailable");
      return;
    }
    const summary = await sres.json();
    $("region").textContent = summary.display_name;
    state.quietAfter = summary.quiet_after_min;
    $("quiet-min").textContent = summary.quiet_after_min;
    const wm = summary.watchlist || {};
    $("watch-meta").textContent = wm.vessels
      ? `${wm.vessels.toLocaleString()} listed vessels · OpenSanctions (CC BY-NC) · updated ${String(wm.fetched_at).slice(0, 10)}`
      : "Watchlist not downloaded yet (runs nightly on the server)";
    const [x0, y0, x1, y1] = summary.bbox;
    if (!location.hash) map.fitBounds([[x0, y0], [x1, y1]], { padding: 40, duration: 0 });

    try {
      state.recent = await (await fetch(`/api/recent?minutes=${TRAIL_MIN}`)).json();
    } catch { state.recent = {}; }
    merge(await (await fetch("/api/vessels?max_age_hours=24")).json());
    state.recent = null;
    state.lastMessage = Date.now();
    setStatus("live", "Live");
    render();
    connect();
    loadSar();
    setInterval(loadSar, 10 * 60 * 1000);
    loadRisk();
    setInterval(loadRisk, 30 * 60 * 1000);                  // scores are recomputed nightly
    $("risk-unlisted").addEventListener("change", loadRisk);   // new radar images arrive a few times a day
    map.on("zoomend", draw);
    requestAnimationFrame(tick);
  }

  map.on("load", init);
})();
