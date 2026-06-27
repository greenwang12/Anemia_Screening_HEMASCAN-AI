import React, { useEffect, useRef, useState } from "react";
import api from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Brain, Upload, CheckCircle2, XCircle, Trash2, RefreshCcw, Loader2, Eye, Hand, Download, Github, Rocket } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";

function ModelCard({ kind, label, icon, status, onUpload, onDelete }) {
  const inputRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const loaded = kind === "eye" ? status?.eye_loaded : status?.nail_loaded;
  const size = kind === "eye" ? status?.eye_size_mb : status?.nail_size_mb;

  const handleFile = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!/\.(h5|keras)$/i.test(f.name)) {
      toast.error("Upload a .h5 or .keras file");
      return;
    }
    setBusy(true);
    try {
      await onUpload(kind, f);
      toast.success(`${label} uploaded`);
    } catch (err) {
      toast.error(err.response?.data?.detail || "Upload failed");
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 flex flex-col" data-testid={`model-card-${kind}`}>
      <div className="flex items-center gap-3 mb-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--secondary)] grid place-items-center text-[var(--primary)]">{icon}</div>
        <div>
          <div className="font-heading text-lg">{label}</div>
          <div className="text-xs text-[var(--muted)] font-mono">
            {kind === "eye" ? "MobileNetV2 · binary (anemia / non-anemia)" : "MobileNetV2 · 6-class"}
          </div>
        </div>
        <span className={`ml-auto inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full ${loaded ? "risk-low" : "risk-high"}`}>
          {loaded ? <><CheckCircle2 className="w-3.5 h-3.5" /> Loaded</> : <><XCircle className="w-3.5 h-3.5" /> Missing</>}
        </span>
      </div>

      <div className="text-xs text-[var(--muted)] font-mono break-all">
        path: {kind === "eye" ? status?.eye_path : status?.nail_path}
      </div>
      {loaded && size && (
        <div className="text-xs text-[var(--muted)] mt-1">Size on disk: <b className="text-[var(--fg)]">{size} MB</b></div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="rounded-xl bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white"
          data-testid={`upload-${kind}-btn`}
        >
          {busy ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Upload className="w-4 h-4 mr-2" />}
          {loaded ? "Replace .h5" : "Upload .h5"}
        </Button>
        {loaded && (
          <Button
            type="button"
            variant="outline"
            onClick={() => onDelete(kind)}
            className="rounded-xl border-[var(--border)]"
            data-testid={`delete-${kind}-btn`}
          >
            <Trash2 className="w-4 h-4 mr-2" /> Remove
          </Button>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".h5,.keras"
          className="hidden"
          onChange={handleFile}
          data-testid={`upload-${kind}-input`}
        />
      </div>
    </div>
  );
}

export default function ModelsAdmin() {
  const { user } = useAuth();
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [zipBusy, setZipBusy] = useState(false);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const { data } = await api.get("/admin/models");
      setStatus(data);
    } catch (e) {
      toast.error(e.response?.data?.detail || "Could not load status");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchStatus(); }, []);

  const upload = async (kind, file) => {
    const fd = new FormData();
    fd.append("file", file);
    await api.post(`/admin/models/${kind}`, fd, { headers: { "Content-Type": "multipart/form-data" } });
    await fetchStatus();
  };

  const remove = async (kind) => {
    if (!window.confirm(`Remove the ${kind} model?`)) return;
    await api.delete(`/admin/models/${kind}`);
    await fetchStatus();
    toast.success(`${kind} model removed`);
  };

  const downloadCodebase = () => {
    setZipBusy(true);
    try {
      const token = localStorage.getItem("token");
      const backendUrl =
  process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";
      const href = `${backendUrl}/api/admin/export?token=${encodeURIComponent(token || "")}`;
      // Open in a new tab so Chrome's preview-iframe sandbox doesn't swallow the download.
      // The backend response sets Content-Disposition: attachment, so the new tab will
      // trigger a native download and close itself.
      const win = window.open(href, "_blank");
      if (!win) {
        // Popup blocked — fall back to assigning to a real anchor click.
        const a = document.createElement("a");
        a.href = href;
        a.target = "_blank";
        a.rel = "noopener";
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        setTimeout(() => { try { a.remove(); } catch {} }, 1000);
      }
      toast.success("Download starting in a new tab — check your Downloads (~10s).");
    } catch (e) {
      toast.error("Export failed");
    } finally {
      setTimeout(() => setZipBusy(false), 1200);
    }
  };

  if (user && user.role !== "admin") {
    return (
      <div className="max-w-3xl mx-auto px-5 py-12">
        <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-8 text-center">
          <h2 className="font-heading text-2xl">Admin only</h2>
          <p className="text-sm text-[var(--muted)] mt-2">Only the admin account can upload and replace model weights.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-5 py-10">
      <div className="flex items-end justify-between mb-6 fade-up">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-[var(--muted)] flex items-center gap-2">
            <Brain className="w-3.5 h-3.5" /> Admin · ML
          </div>
          <h1 className="font-heading text-4xl sm:text-5xl tracking-tight mt-1">Model weights</h1>
          <p className="text-[var(--muted)] mt-1 max-w-2xl">
            Upload your trained MobileNetV2 <code className="font-mono text-[var(--primary)]">.h5</code> files.
            Until both are present, the system falls back to a vision-LLM proxy.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={fetchStatus} className="rounded-xl" data-testid="refresh-models-btn">
            <RefreshCcw className="w-4 h-4 mr-2" /> Refresh
          </Button>
          <Button onClick={downloadCodebase} disabled={zipBusy} className="rounded-xl bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white" data-testid="download-codebase-btn">
            {zipBusy ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Download className="w-4 h-4 mr-2" />}
            {zipBusy ? "Zipping…" : "Download codebase (.zip)"}
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="text-[var(--muted)] text-sm flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" /> Loading…</div>
      ) : (
        <div className="grid md:grid-cols-2 gap-5 fade-up delay-1">
          <ModelCard kind="eye" label="Eye model" icon={<Eye className="w-5 h-5" />} status={status} onUpload={upload} onDelete={remove} />
          <ModelCard kind="nail" label="Nail model" icon={<Hand className="w-5 h-5" />} status={status} onUpload={upload} onDelete={remove} />
        </div>
      )}

      <div className="mt-8 rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 fade-up delay-2">
        <div className="font-heading text-lg">Inference pipeline summary</div>
        <ul className="mt-3 space-y-2 text-sm text-[var(--muted)]">
          <li>1. Quality check (blur variance + brightness) on the upload.</li>
          <li>2. TTA batch (orig + h-flip + ±10° rotation + center-crop) averaged through MobileNetV2.</li>
          <li>3. Real Grad-CAM from the last conv layer (<span className="font-mono">Conv_1</span>) via <span className="font-mono">tf.GradientTape</span>.</li>
          <li>4. Eye head returns P(anemia); nail head returns 6-class softmax — anemia probability = sum of <span className="font-mono">blue_finger + clubbing + pitting</span>.</li>
          <li>5. Fusion: Noisy-OR + confidence-weighted average + optional logistic-regression meta-learner.</li>
        </ul>
      </div>

      <div className="mt-6 grid md:grid-cols-2 gap-5 fade-up delay-3">
        <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6">
          <div className="flex items-center gap-2 text-[var(--primary)]">
            <Github className="w-4 h-4" /> <span className="text-[10px] uppercase tracking-[0.25em]">Export</span>
          </div>
          <h3 className="font-heading text-lg mt-1">Open in VS Code</h3>
          <p className="text-sm text-[var(--muted)] mt-2 leading-relaxed">
            Click <b>Download codebase (.zip)</b> above. Unzip it, then run <code className="font-mono text-[var(--primary)]">code hemascan</code> from your terminal — or right-click → <i>Open with VS Code</i>. The zip excludes <code>node_modules</code>, caches, and <code>.git</code> so it stays small.
          </p>
          <div className="mt-3 text-xs font-mono text-[var(--muted)] bg-[var(--surface-2)] rounded-lg p-3 leading-relaxed">
            unzip hemascan-*.zip<br/>
            cd hemascan<br/>
            code .
          </div>
        </div>
        <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6">
          <div className="flex items-center gap-2 text-[var(--primary)]">
            <Rocket className="w-4 h-4" /> <span className="text-[10px] uppercase tracking-[0.25em]">Deploy & GitHub</span>
          </div>
          <h3 className="font-heading text-lg mt-1">Ship it</h3>
          <p className="text-sm text-[var(--muted)] mt-2 leading-relaxed">
            Use the Emergent dashboard for one-click <b>GitHub push</b> and <b>Deploy</b> — both options live in the top-right menu of the chat sidebar. See the chat below for full instructions.
          </p>
        </div>
      </div>
    </div>
  );
}
