import React, { useEffect, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";
import { Eye, Hand, Sparkles } from "lucide-react";

export default function GradCamViewer({
  imageBase64,
  attentionRegions = [],
  heatmapBase64 = null,
  label = "Image",
  icon = "eye",
  engine = "mobilenetv2",
  variant = "default", // "default" | "attention-map"
}) {
  const canvasRef = useRef(null);
  const [opacity, setOpacity] = useState(variant === "attention-map" ? 70 : 60);

  const dataUrl = imageBase64
    ? (imageBase64.startsWith("data:") ? imageBase64 : `data:image/jpeg;base64,${imageBase64}`)
    : null;
  const heatmapUrl = heatmapBase64
    ? (heatmapBase64.startsWith("data:") ? heatmapBase64 : `data:image/png;base64,${heatmapBase64}`)
    : null;

  useEffect(() => {
    if (!dataUrl || heatmapUrl) return;
    const img = new window.Image();
    img.onload = () => {
      const W = 480;
      const ratio = img.height / img.width;
      const H = Math.round(W * ratio);
      drawHeatmap(W, H);
    };
    img.src = dataUrl;
  }, [dataUrl, heatmapUrl, attentionRegions, variant]);

  const drawHeatmap = (W, H) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);

    const regions = (attentionRegions && attentionRegions.length > 0)
      ? attentionRegions
      : [{ cx: 0.5, cy: 0.5, radius: 0.35, intensity: 0.6 }];

    regions.forEach(({ cx, cy, radius, intensity }) => {
      const x = cx * W;
      const y = cy * H;
      const r = Math.max(20, (radius || 0.25) * Math.min(W, H));
      const i = Math.max(0.2, Math.min(1, intensity || 0.6));
      const grad = ctx.createRadialGradient(x, y, 0, x, y, r);

      if (variant === "attention-map") {
        // Richer, more saturated JET-style colormap
        grad.addColorStop(0,    `rgba(255, 20,  20,  ${0.95 * i})`); // vivid red core
        grad.addColorStop(0.25, `rgba(255, 140, 0,   ${0.80 * i})`); // orange
        grad.addColorStop(0.5,  `rgba(255, 230, 0,   ${0.55 * i})`); // yellow
        grad.addColorStop(0.75, `rgba(30,  180, 80,  ${0.25 * i})`); // green fade
        grad.addColorStop(1,    `rgba(30,  180, 80,  0)`);
      } else {
        // Original softer palette
        grad.addColorStop(0,    `rgba(224, 60,  30,  ${0.85 * i})`);
        grad.addColorStop(0.45, `rgba(255, 170, 60,  ${0.55 * i})`);
        grad.addColorStop(0.8,  `rgba(60,  120, 255, ${0.18 * i})`);
        grad.addColorStop(1,    `rgba(60,  120, 255, 0)`);
      }

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  const isReal = !!heatmapUrl && engine === "mobilenetv2";
  const isAttentionMap = variant === "attention-map";

  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] overflow-hidden">

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          {icon === "nail"
            ? <Hand className="w-4 h-4 text-[var(--primary)]" />
            : <Eye  className="w-4 h-4 text-[var(--primary)]" />}
          <span className="font-heading font-medium">{label}</span>
        </div>

        {/* CHANGED: always show green pill in attention-map variant */}
        <span
          className={`inline-flex items-center gap-1.5 text-[10px] uppercase tracking-[0.2em] px-2.5 py-1 rounded-full font-medium
            ${isAttentionMap || isReal
              ? "bg-[var(--secondary)] text-[var(--primary)]"
              : "text-[var(--muted)]"
            }`}
        >
          <Sparkles className="w-3 h-3" />
          {isAttentionMap ? "Attention Map" : "Where the AI looked"}
        </span>
      </div>

      {/* Body */}
      <div className="p-4 grid sm:grid-cols-2 gap-4">

        {/* Original */}
        <div>
          <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--muted)] mb-2">Original</div>
          <div className="relative rounded-xl overflow-hidden border border-[var(--border)] bg-black/5">
            {dataUrl && <img src={dataUrl} alt="original" className="w-full block" />}
          </div>
        </div>

        {/* Heatmap */}
        <div>
          <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--muted)] mb-2">Attention heatmap</div>
          <div className="relative rounded-xl overflow-hidden border border-[var(--border)] bg-black/5">
            {dataUrl && <img src={dataUrl} alt="overlay base" className="w-full block" />}
            {heatmapUrl ? (
              <img
                src={heatmapUrl}
                alt="grad-cam overlay"
                className="absolute inset-0 w-full h-full heatmap-overlay"
                style={{
                  opacity: opacity / 100,
                  // CHANGED: boost contrast in attention-map variant
                  filter: isAttentionMap ? "saturate(1.4) contrast(1.1)" : "none",
                  mixBlendMode: isAttentionMap ? "hard-light" : "normal",
                }}
              />
            ) : (
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full heatmap-overlay"
                style={{ opacity: opacity / 100 }}
              />
            )}
          </div>

          {/* Opacity slider */}
          <div className="mt-3 flex items-center gap-3">
            <span className="text-xs text-[var(--muted)] min-w-[64px]">Overlay</span>
            <Slider
              value={[opacity]}
              min={0}
              max={100}
              step={1}
              onValueChange={(v) => setOpacity(v[0])}
              data-testid={`gradcam-opacity-${icon}`}
              className="flex-1"
            />
            <span className="text-xs font-mono w-10 text-right">{opacity}%</span>
          </div>
        </div>

      </div>
    </div>
  );
}