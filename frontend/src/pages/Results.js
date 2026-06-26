import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import api from "@/lib/api";
import RiskCard from "@/components/RiskCard";
import GradCamViewer from "@/components/GradCamViewer";
import { Button } from "@/components/ui/button";
import { Download, Eye, Hand, Layers, ArrowLeft, Loader2 } from "lucide-react";
import { toast } from "sonner";

const NAIL_SIGN_META = {
  pallor:      { label: "Paleness of the nail bed",   desc: "How washed-out the colour under the nail looks. Healthy nail beds are pink." },
  koilonychia: { label: "Spoon-shape (koilonychia)",  desc: "A curved, dipped nail with a dark centre and bright edges — a classic sign." },
  platonychia: { label: "Flattened nail plate",       desc: "The nail looks unusually flat instead of gently curved." },
  ridging:     { label: "Vertical ridges",            desc: "Thin lines running from cuticle to tip across the nail." },
  brittleness: { label: "Brittle / chipped edges",    desc: "Irregular, broken-looking nail tips." },
  yellowing:   { label: "Yellow tint",                desc: "A slight yellow shift in the nail (weaker sign on its own)." },
};

export default function Results() {
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await api.get(`/screenings/${id}`);
        setData(res.data);
      } catch (e) {
        toast.error("Could not load screening");
      } finally {
        setLoading(false);
      }
    })();
  }, [id]);

  const downloadReport = () => {
    // Open a printable view that uses window.print()
    const w = window.open("", "_blank");
    if (!w) return;
    const html = buildReportHTML(data);
    w.document.write(html);
    w.document.close();
    setTimeout(() => w.print(), 400);
  };

  if (loading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center text-[var(--muted)]">
        <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading screening…
      </div>
    );
  }
  if (!data) {
    return <div className="max-w-6xl mx-auto px-5 py-10 text-[var(--muted)]">Not found.</div>;
  }

  return (
    <div className="max-w-6xl mx-auto px-5 py-10" data-testid="results-page">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-6 fade-up">
        <div>
          <Link to="/history" className="text-xs text-[var(--muted)] inline-flex items-center gap-1 hover:text-[var(--primary)]">
            <ArrowLeft className="w-3 h-3" /> All screenings
          </Link>
          <h1 className="font-heading text-4xl sm:text-5xl tracking-tight mt-1">
            Screening for <span className="text-[var(--primary)]">{data.patient_name || "Anonymous"}</span>
          </h1>
          <div className="text-sm text-[var(--muted)] mt-1">
            {new Date(data.created_at).toLocaleString()}
            {data.patient_age ? ` · ${data.patient_age} y` : ""}
            {data.patient_sex ? ` · ${data.patient_sex}` : ""}
          </div>
        </div>
        <Button onClick={downloadReport} className="rounded-xl bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white" data-testid="download-report-btn">
          <Download className="w-4 h-4 mr-2" /> Download report
        </Button>
      </div>

      {/* Bento grid: three risk cards */}
      <section className="grid md:grid-cols-3 gap-5 fade-up delay-1">
        {data.eye_result ? (
          <RiskCard
            title="Eye check"
            subtitle="Paleness of the inner eyelid"
            riskPercent={data.eye_result.risk_percent}
            riskLabel={data.eye_result.risk_label}
            confidence={data.eye_result.confidence}
            testId="risk-eye"
          />
        ) : <EmptyRisk title="Eye check" subtitle="No photo provided" icon={<Eye className="w-4 h-4" />} />}

        {data.nail_result ? (
          <RiskCard
            title="Nail check"
            subtitle="Paleness & shape of the nail"
            riskPercent={data.nail_result.risk_percent}
            riskLabel={data.nail_result.risk_label}
            confidence={data.nail_result.confidence}
            testId="risk-nail"
          />
        ) : <EmptyRisk title="Nail check" subtitle="No photo provided" icon={<Hand className="w-4 h-4" />} />}

        <RiskCard
          title="Overall estimate"
          subtitle="Both signals combined"
          riskPercent={data.fusion_result.risk_percent}
          riskLabel={data.fusion_result.risk_label}
          confidence={data.fusion_result.confidence}
          accent
          testId="risk-fusion"
        />
      </section>

      {/* Findings */}
      <section className="grid md:grid-cols-2 gap-5 mt-6 fade-up delay-2">
        {data.eye_result && (
          <FindingsCard icon={<Eye className="w-4 h-4" />} title="Eye check — what we saw" result={data.eye_result} testId="findings-eye" />
        )}
        {data.nail_result && (
          <FindingsCard icon={<Hand className="w-4 h-4" />} title="Nail check — what we saw" result={data.nail_result} testId="findings-nail" />
        )}
      </section>

      {/* Nail clinical signs (multi-feature analyzer) */}
      {data.nail_result?.model_extras?.clinical_features && (
        <section className="mt-6 rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5 fade-up delay-2" data-testid="nail-clinical-signs">
          <div className="flex items-center gap-2 mb-1">
            <Hand className="w-4 h-4 text-[var(--primary)]" />
            <span className="font-heading font-medium">What we looked at on the nail</span>
            <span className="ml-auto text-[10px] uppercase tracking-[0.25em] text-[var(--muted)]">image analysis</span>
          </div>
          <p className="text-xs text-[var(--muted)] mb-3">
            We checked six visual signs that doctors look for when screening for anemia. The bar shows how strong each sign is in your photo.
          </p>
          <div className="grid sm:grid-cols-2 gap-x-6 gap-y-2">
            {Object.entries(data.nail_result.model_extras.clinical_features).map(([sign, v]) => {
              const meta = NAIL_SIGN_META[sign] || { label: sign, desc: "" };
              const weight = data.nail_result.model_extras.feature_weights?.[sign] || 0;
              const positive = v > 0.30;
              return (
                <div key={sign} className="py-1.5" data-testid={`nail-sign-${sign}`}>
                  <div className="flex items-center justify-between text-sm">
                    <div>
                      <span className={positive ? "font-medium text-[var(--accent-urgent)]" : "text-[var(--fg)]"}>
                        {meta.label}
                      </span>
                      <span className="ml-2 text-[10px] font-mono text-[var(--muted)]">
                        weight {Math.round(weight * 100)}%
                      </span>
                    </div>
                    <span className="font-mono text-[var(--muted)]">{Math.round(v * 100)}%</span>
                  </div>
                  <div className="h-1.5 bg-[var(--surface-2)] rounded-full overflow-hidden mt-1">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.max(2, v * 100)}%`,
                        background: positive ? "var(--accent-urgent)" : "var(--primary)",
                      }}
                    />
                  </div>
                  <div className="text-[11px] text-[var(--muted)] mt-1">{meta.desc}</div>
                </div>
              );
            })}
          </div>
          <div className="text-[11px] text-[var(--muted)] mt-4 pt-3 border-t border-[var(--border)]">
            <span className="text-[var(--fg)] font-medium">How the final nail score is built:</span>{" "}
            We blend the AI model with these visual checks.
            {" "}AI model says anemia is likely:{" "}
            <span className="text-[var(--fg)] font-mono">{Math.round((data.nail_result.model_extras.p_model_anemia || 0) * 100)}%</span>
            {data.nail_result.model_extras.blend_weights?.model != null && (
              <span> (weight {Math.round(data.nail_result.model_extras.blend_weights.model * 100)}%)</span>
            )}
            . Visual checks say:{" "}
            <span className="text-[var(--fg)] font-mono">{Math.round((data.nail_result.model_extras.p_features_anemia || 0) * 100)}%</span>
            {data.nail_result.model_extras.blend_weights?.features != null && (
              <span> (weight {Math.round(data.nail_result.model_extras.blend_weights.features * 100)}%)</span>
            )}
            .
          </div>
        </section>
      )}

      {/* Grad-CAM viewers */}
      <section className="grid lg:grid-cols-2 gap-5 mt-6 fade-up delay-3">
        {data.eye_image_base64 && (
          <GradCamViewer
            imageBase64={data.eye_image_base64}
            attentionRegions={data.eye_result?.attention_regions}
            heatmapBase64={data.eye_result?.gradcam_heatmap_base64}
            engine={data.eye_result?.engine}
            label="Eye · where the AI looked"
            icon="eye"
          />
        )}
        {data.nail_image_base64 && (
          <GradCamViewer
            imageBase64={data.nail_image_base64}
            attentionRegions={data.nail_result?.attention_regions}
            heatmapBase64={data.nail_result?.gradcam_heatmap_base64}
            engine={data.nail_result?.engine}
            label="Nail · where the AI looked"
            icon="nail"
          />
        )}
      </section>

      {/* Overall estimate explanation */}
      <section className="mt-6 rounded-2xl border border-[var(--primary)]/30 bg-[var(--surface)] p-6 fade-up delay-4">
        <div className="flex items-center gap-2 text-[var(--primary)]">
          <Layers className="w-4 h-4" />
          <span className="text-[10px] uppercase tracking-[0.25em]">How we got to one score</span>
        </div>
        <h3 className="font-heading text-2xl mt-1">Two signals, one easy-to-read result</h3>
        <p className="text-sm text-[var(--muted)] mt-2 leading-relaxed max-w-3xl">
          We look at your inner eyelid and your nail separately, then combine them.
          When both photos agree, the final score becomes more confident.
          When they disagree, we soften the score so we don&apos;t over-call it.
          Paleness gets the most weight because it&apos;s the clearest visual sign of low hemoglobin.
        </p>
        <div className="mt-3 text-xs text-[var(--muted)]">
          Photos used in this estimate: <span className="text-[var(--fg)]">{data.fusion_result.modalities_used?.join(" + ") || "—"}</span>
        </div>
      </section>
    </div>
  );
}

function FindingsCard({ icon, title, result, testId }) {
  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5" data-testid={testId}>
      <div className="flex items-center gap-2 mb-3">
        <div className="w-8 h-8 rounded-lg bg-[var(--secondary)] grid place-items-center text-[var(--primary)]">{icon}</div>
        <div className="font-heading font-medium">{title}</div>
      </div>
      <ul className="text-sm space-y-1.5 text-[var(--fg)]">
        {(result.key_findings || []).map((f, i) => (
          <li key={i} className="flex gap-2"><span className="text-[var(--primary)] mt-1">•</span>{f}</li>
        ))}
      </ul>
      <div className="text-sm text-[var(--muted)] mt-3 leading-relaxed">{result.reasoning}</div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-[var(--muted)]">
        <div>Paleness level: <span className="text-[var(--fg)]">{result.pallor_score}/10</span></div>
        <div>How sure we are: <span className="text-[var(--fg)]">{Math.round(result.confidence * 100)}%</span></div>
      </div>
    </div>
  );
}

function EmptyRisk({ title, subtitle, icon }) {
  return (
    <div className="rounded-2xl border border-dashed border-[var(--border)] bg-[var(--surface)] p-5 flex flex-col justify-center text-center">
      <div className="mx-auto w-9 h-9 rounded-lg bg-[var(--surface-2)] grid place-items-center text-[var(--muted)]">{icon}</div>
      <div className="text-[10px] uppercase tracking-[0.25em] text-[var(--muted)] mt-3">{title}</div>
      <div className="text-sm text-[var(--muted)] mt-1">{subtitle}</div>
    </div>
  );
}

function buildReportHTML(d) {
  const f = d.fusion_result || {};
  const safe = (v) => (v == null ? "—" : v);
  return `
<!doctype html><html><head><meta charset="utf-8"/><title>HemaScan report</title>
<style>
body { font-family: 'Manrope', system-ui, sans-serif; color: #1A1A1A; padding: 40px; max-width: 800px; margin: auto; }
h1 { font-family: 'Outfit', sans-serif; font-size: 28px; margin: 0 0 4px; }
h2 { font-family: 'Outfit', sans-serif; font-size: 18px; margin-top: 24px; }
.muted { color: #6B6B6B; font-size: 12px; }
.row { display: flex; gap: 16px; margin-top: 16px; }
.card { border: 1px solid #E5E5E5; border-radius: 14px; padding: 16px; flex: 1; }
.pct { font-family: 'Outfit', sans-serif; font-size: 28px; color: #2E5C4F; font-weight: 600; }
.label { font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #6B6B6B; }
.footer { margin-top: 32px; font-size: 11px; color: #6B6B6B; }
img { max-width: 240px; border-radius: 10px; }
</style></head><body>
<h1>HemaScan screening report</h1>
<div class="muted">${new Date(d.created_at).toLocaleString()}</div>
<h2>Patient</h2>
<div>${safe(d.patient_name)} · ${safe(d.patient_age)} y · ${safe(d.patient_sex)}</div>
<h2>Fusion result</h2>
<div class="row">
  <div class="card">
    <div class="label">Risk likelihood</div>
    <div class="pct">${f.risk_percent}%</div>
    <div>${f.risk_label} · confidence ${(Math.round((f.confidence||0)*100))}%</div>
  </div>
  <div class="card">
    <div class="label">Modalities</div>
    <div>${(f.modalities_used||[]).join(" + ")}</div>
  </div>
</div>
${d.eye_result ? `<h2>Eye baseline</h2>
  <div>Risk: <b>${d.eye_result.risk_percent}%</b> (${d.eye_result.risk_label})</div>
  <ul>${(d.eye_result.key_findings||[]).map(x=>`<li>${x}</li>`).join("")}</ul>
  <div class="muted">${d.eye_result.reasoning||""}</div>` : ""}
${d.nail_result ? `<h2>Nail baseline</h2>
  <div>Risk: <b>${d.nail_result.risk_percent}%</b> (${d.nail_result.risk_label})</div>
  <ul>${(d.nail_result.key_findings||[]).map(x=>`<li>${x}</li>`).join("")}</ul>
  <div class="muted">${d.nail_result.reasoning||""}</div>` : ""}
<div class="footer">This is an AI screening report and is not a medical diagnosis. Consult a qualified medical professional for treatment decisions.</div>
</body></html>`;
}
