/**
 * Canvas waveform editor.
 *
 * Displays mono PCM samples, supports zooming, selection, draggable markers,
 * keyboard nudge operations, and non-destructive gain/fade preview.
 */

export class WaveformEditor {
  constructor(canvas, options = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError("canvas must be an HTMLCanvasElement");
    }

    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.samples = new Float32Array();
    this.sampleRate = options.sampleRate ?? 48000;
    this.viewStart = 0;
    this.viewEnd = 1;
    this.selection = null;
    this.markers = [];
    this.drag = null;
    this.gainDb = 0;
    this.fadeInSeconds = 0;
    this.fadeOutSeconds = 0;
    this.deviceScale = window.devicePixelRatio || 1;

    this.bindEvents();
    this.resize();
  }

  setSamples(samples, sampleRate = this.sampleRate) {
    this.samples = samples instanceof Float32Array
      ? samples
      : Float32Array.from(samples);
    this.sampleRate = sampleRate;
    this.viewStart = 0;
    this.viewEnd = Math.max(1, this.samples.length);
    this.selection = null;
    this.render();
  }

  setPreview({ gainDb = this.gainDb, fadeInSeconds = 0, fadeOutSeconds = 0 }) {
    this.gainDb = gainDb;
    this.fadeInSeconds = Math.max(0, fadeInSeconds);
    this.fadeOutSeconds = Math.max(0, fadeOutSeconds);
    this.render();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.floor(rect.width * this.deviceScale));
    this.canvas.height = Math.max(1, Math.floor(rect.height * this.deviceScale));
    this.render();
  }

  xToSample(clientX) {
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
    const fraction = rect.width === 0 ? 0 : x / rect.width;
    return Math.round(this.viewStart + fraction * (this.viewEnd - this.viewStart));
  }

  sampleToX(sample) {
    const width = this.canvas.width / this.deviceScale;
    const span = Math.max(1, this.viewEnd - this.viewStart);
    return ((sample - this.viewStart) / span) * width;
  }

  sampleAt(index) {
    if (index < 0 || index >= this.samples.length) return 0;
    const raw = this.samples[index];
    const gain = 10 ** (this.gainDb / 20);

    let envelope = 1;
    const fadeInSamples = Math.floor(this.fadeInSeconds * this.sampleRate);
    const fadeOutSamples = Math.floor(this.fadeOutSeconds * this.sampleRate);

    if (fadeInSamples > 0 && index < fadeInSamples) {
      envelope *= index / fadeInSamples;
    }

    const remaining = this.samples.length - 1 - index;
    if (fadeOutSamples > 0 && remaining < fadeOutSamples) {
      envelope *= Math.max(0, remaining / fadeOutSamples);
    }

    return Math.max(-1, Math.min(1, raw * gain * envelope));
  }

  zoomAt(clientX, factor) {
    const center = this.xToSample(clientX);
    const oldSpan = this.viewEnd - this.viewStart;
    const minSpan = Math.min(this.samples.length || 1, 128);
    const newSpan = Math.max(
      minSpan,
      Math.min(this.samples.length || 1, oldSpan * factor),
    );

    const ratio = oldSpan === 0 ? 0.5 : (center - this.viewStart) / oldSpan;
    let start = center - ratio * newSpan;
    let end = start + newSpan;

    if (start < 0) {
      end -= start;
      start = 0;
    }
    if (end > this.samples.length) {
      start -= end - this.samples.length;
      end = this.samples.length;
    }

    this.viewStart = Math.max(0, start);
    this.viewEnd = Math.max(this.viewStart + 1, end);
    this.render();
  }

  addMarker(sample, label = `M${this.markers.length + 1}`) {
    const marker = {
      id: crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`,
      sample: Math.max(0, Math.min(this.samples.length, Math.round(sample))),
      label,
    };
    this.markers.push(marker);
    this.markers.sort((a, b) => a.sample - b.sample);
    this.render();
    return marker;
  }

  markerNear(clientX, tolerancePx = 8) {
    const rect = this.canvas.getBoundingClientRect();
    const localX = clientX - rect.left;
    let best = null;
    let bestDistance = Infinity;

    for (const marker of this.markers) {
      const distance = Math.abs(this.sampleToX(marker.sample) - localX);
      if (distance <= tolerancePx && distance < bestDistance) {
        best = marker;
        bestDistance = distance;
      }
    }
    return best;
  }

  bindEvents() {
    window.addEventListener("resize", () => this.resize());

    this.canvas.addEventListener("wheel", event => {
      event.preventDefault();
      this.zoomAt(event.clientX, event.deltaY > 0 ? 1.2 : 0.82);
    }, { passive: false });

    this.canvas.addEventListener("pointerdown", event => {
      this.canvas.setPointerCapture(event.pointerId);
      const marker = this.markerNear(event.clientX);

      if (marker) {
        this.drag = { kind: "marker", marker };
        return;
      }

      const start = this.xToSample(event.clientX);
      this.selection = { start, end: start };
      this.drag = { kind: "selection" };
      this.render();
    });

    this.canvas.addEventListener("pointermove", event => {
      if (!this.drag) return;

      if (this.drag.kind === "selection") {
        this.selection.end = this.xToSample(event.clientX);
      } else if (this.drag.kind === "marker") {
        this.drag.marker.sample = this.xToSample(event.clientX);
        this.markers.sort((a, b) => a.sample - b.sample);
      }
      this.render();
    });

    this.canvas.addEventListener("pointerup", event => {
      if (this.canvas.hasPointerCapture(event.pointerId)) {
        this.canvas.releasePointerCapture(event.pointerId);
      }
      if (this.selection && this.selection.start > this.selection.end) {
        [this.selection.start, this.selection.end] =
          [this.selection.end, this.selection.start];
      }
      this.drag = null;
      this.render();
    });

    this.canvas.addEventListener("dblclick", event => {
      this.addMarker(this.xToSample(event.clientX));
    });

    this.canvas.tabIndex = 0;
    this.canvas.addEventListener("keydown", event => {
      if (!this.selection) return;

      const step = event.shiftKey
        ? Math.max(1, Math.round(this.sampleRate * 0.1))
        : 1;

      if (event.key === "ArrowLeft") {
        this.selection.end = Math.max(0, this.selection.end - step);
        event.preventDefault();
      } else if (event.key === "ArrowRight") {
        this.selection.end = Math.min(this.samples.length, this.selection.end + step);
        event.preventDefault();
      } else if (event.key === "Escape") {
        this.selection = null;
        event.preventDefault();
      } else {
        return;
      }

      this.render();
    });
  }

  renderWaveform(width, height) {
    if (this.samples.length === 0) return;

    const center = height / 2;
    const span = Math.max(1, this.viewEnd - this.viewStart);
    const samplesPerPixel = span / width;

    this.ctx.beginPath();

    for (let x = 0; x < width; x++) {
      const from = Math.floor(this.viewStart + x * samplesPerPixel);
      const to = Math.min(
        this.samples.length,
        Math.max(from + 1, Math.ceil(this.viewStart + (x + 1) * samplesPerPixel)),
      );

      let min = 1;
      let max = -1;
      for (let i = from; i < to; i++) {
        const value = this.sampleAt(i);
        if (value < min) min = value;
        if (value > max) max = value;
      }

      const y1 = center - max * center * 0.9;
      const y2 = center - min * center * 0.9;
      this.ctx.moveTo(x + 0.5, y1);
      this.ctx.lineTo(x + 0.5, y2);
    }

    this.ctx.strokeStyle = "#222";
    this.ctx.lineWidth = 1;
    this.ctx.stroke();
  }

  renderSelection(width, height) {
    if (!this.selection) return;

    const x1 = this.sampleToX(this.selection.start);
    const x2 = this.sampleToX(this.selection.end);
    this.ctx.fillStyle = "rgba(80, 120, 220, 0.18)";
    this.ctx.fillRect(Math.min(x1, x2), 0, Math.abs(x2 - x1), height);
  }

  renderMarkers(height) {
    this.ctx.font = "12px system-ui";
    for (const marker of this.markers) {
      if (marker.sample < this.viewStart || marker.sample > this.viewEnd) continue;
      const x = this.sampleToX(marker.sample);

      this.ctx.beginPath();
      this.ctx.moveTo(x + 0.5, 0);
      this.ctx.lineTo(x + 0.5, height);
      this.ctx.strokeStyle = "#b52222";
      this.ctx.lineWidth = 1;
      this.ctx.stroke();

      const textWidth = this.ctx.measureText(marker.label).width;
      this.ctx.fillStyle = "#fff";
      this.ctx.fillRect(x + 3, 3, textWidth + 6, 16);
      this.ctx.fillStyle = "#7d1111";
      this.ctx.fillText(marker.label, x + 6, 15);
    }
  }

  render() {
    if (!this.ctx) return;

    const width = this.canvas.width / this.deviceScale;
    const height = this.canvas.height / this.deviceScale;

    this.ctx.setTransform(this.deviceScale, 0, 0, this.deviceScale, 0, 0);
    this.ctx.clearRect(0, 0, width, height);
    this.ctx.fillStyle = "#fafafa";
    this.ctx.fillRect(0, 0, width, height);

    this.ctx.strokeStyle = "#ddd";
    this.ctx.beginPath();
    this.ctx.moveTo(0, height / 2 + 0.5);
    this.ctx.lineTo(width, height / 2 + 0.5);
    this.ctx.stroke();

    this.renderSelection(width, height);
    this.renderWaveform(width, height);
    this.renderMarkers(height);
  }

  selectedRange() {
    if (!this.selection) return null;
    return {
      startSample: Math.round(Math.min(this.selection.start, this.selection.end)),
      endSample: Math.round(Math.max(this.selection.start, this.selection.end)),
      startSeconds: Math.min(this.selection.start, this.selection.end) / this.sampleRate,
      endSeconds: Math.max(this.selection.start, this.selection.end) / this.sampleRate,
    };
  }
}
