import React from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowRight, Eye, Hand, Layers, Sparkles, ShieldCheck, Smartphone } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

export default function Landing() {
  const { user } = useAuth();
  const ctaTo = user ? "/screen" : "/register";
  return (
    <div className="relative overflow-hidden">
      {/* Hero */}
      <section className="relative dot-grid">
        <div className="max-w-6xl mx-auto px-5 pt-16 pb-24 md:pt-24 md:pb-32 grid md:grid-cols-12 gap-10 items-center">
          <div className="md:col-span-7 fade-up">
            <div className="inline-flex items-center gap-2 rounded-full bg-[var(--secondary)] text-[var(--primary)] px-3 py-1 text-xs font-medium mb-5">
              <Sparkles className="w-3.5 h-3.5" /> Research prototype · Explainable AI
            </div>
            <h1 className="font-heading text-4xl sm:text-5xl lg:text-6xl font-medium leading-[1.05] tracking-tight">
              Screen for anemia<br />
              <span className="text-[var(--primary)]">from a glance.</span>
            </h1>
            <p className="mt-5 text-base sm:text-lg text-[var(--muted)] max-w-xl leading-relaxed">
              A non-invasive screen-from-home tool. We check the paleness of your inner eyelid and your nail and
              <span className="text-[var(--fg)] font-medium"> combine both into one easy score</span> — with a visual heatmap that highlights exactly what the AI looked at.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Link to={ctaTo}>
                <Button size="lg" className="bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white rounded-xl px-6" data-testid="hero-cta-btn">
                  Start a screening <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </Link>
              <Link to="/learn">
                <Button size="lg" variant="outline" className="rounded-xl border-[var(--border)]" data-testid="hero-learn-btn">
                  How it works
                </Button>
              </Link>
            </div>
            <div className="mt-8 flex flex-wrap items-center gap-5 text-xs text-[var(--muted)]">
              <div className="flex items-center gap-2"><ShieldCheck className="w-4 h-4" /> No image leaves your account</div>
              <div className="flex items-center gap-2"><Smartphone className="w-4 h-4" /> Mobile-first capture</div>
            </div>
          </div>

          <div className="md:col-span-5 fade-up delay-2">
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <div className="text-[10px] uppercase tracking-[0.25em] text-[var(--muted)]">Live fusion model</div>
                <div className="font-mono text-xs text-[var(--primary)]">v0.1 · demo</div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <PreviewTile icon={<Eye className="w-4 h-4" />} label="Eye CNN" pct={42} bar="var(--accent-heat)" />
                <PreviewTile icon={<Hand className="w-4 h-4" />} label="Nail CNN" pct={51} bar="var(--accent-heat)" />
              </div>
              <div className="mt-3 rounded-xl bg-[var(--surface-2)] p-4">
                <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.25em] text-[var(--muted)]">
                  <Layers className="w-3.5 h-3.5" /> Fusion output
                </div>
                <div className="mt-1 flex items-baseline gap-2">
                  <div className="font-heading text-4xl font-semibold text-[var(--primary)]">47%</div>
                  <div className="text-xs text-[var(--muted)]">moderate risk</div>
                </div>
                <div className="h-1.5 w-full bg-white rounded-full overflow-hidden mt-3">
                  <div className="h-full" style={{ width: "47%", background: "var(--primary)" }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="border-t border-[var(--border)] bg-[var(--surface)]">
        <div className="max-w-6xl mx-auto px-5 py-16 grid md:grid-cols-3 gap-6">
          <Feature
            icon={<Eye className="w-5 h-5" />}
            title="Conjunctival pallor"
            desc="Baseline CNN evaluates the lower-eyelid lining — the most common clinical cue for anemia."
          />
          <Feature
            icon={<Hand className="w-5 h-5" />}
            title="Nail-bed pallor"
            desc="Second baseline CNN looks at fingernail bed colour, capturing a complementary signal."
          />
          <Feature
            icon={<Layers className="w-5 h-5" />}
            title="Late-fusion model"
            desc="Confidence-weighted combination of both baselines lifts overall accuracy versus either alone."
          />
        </div>
      </section>
    </div>
  );
}

function PreviewTile({ icon, label, pct, bar }) {
  return (
    <div className="rounded-xl border border-[var(--border)] p-3">
      <div className="flex items-center gap-2 text-xs text-[var(--muted)]">{icon} {label}</div>
      <div className="font-heading text-2xl font-semibold mt-1">{pct}%</div>
      <div className="h-1 w-full bg-[var(--surface-2)] rounded-full mt-2 overflow-hidden">
        <div className="h-full" style={{ width: `${pct}%`, background: bar }} />
      </div>
    </div>
  );
}

function Feature({ icon, title, desc }) {
  return (
    <div className="rounded-2xl border border-[var(--border)] p-6 bg-[var(--bg)]">
      <div className="w-9 h-9 rounded-xl bg-[var(--secondary)] grid place-items-center text-[var(--primary)] mb-3">{icon}</div>
      <h3 className="font-heading text-xl">{title}</h3>
      <p className="text-sm text-[var(--muted)] mt-2 leading-relaxed">{desc}</p>
    </div>
  );
}
