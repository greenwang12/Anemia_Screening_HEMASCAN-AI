import React from "react";
import { CheckCircle2, AlertTriangle, AlertOctagon } from "lucide-react";

const styles = {
  Low:      { cls: "risk-low",  icon: CheckCircle2,   label: "Low risk" },
  Moderate: { cls: "risk-mod",  icon: AlertTriangle,  label: "Moderate risk" },
  High:     { cls: "risk-high", icon: AlertOctagon,   label: "High risk" },
};

export default function RiskCard({ title, riskPercent, riskLabel, confidence, subtitle, accent = false, testId }) {
  const s = styles[riskLabel] || styles.Low;
  const Icon = s.icon;
  return (
    <div
      data-testid={testId}
      className={`rounded-2xl border bg-[var(--surface)] p-5 flex flex-col gap-3 ${accent ? "border-[var(--primary)] ring-1 ring-[var(--primary)]/30" : "border-[var(--border)]"}`}
    >
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.25em] text-[var(--muted)]">{title}</div>
          <div className="font-heading text-lg mt-0.5">{subtitle}</div>
        </div>
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${s.cls}`}>
          <Icon className="w-3.5 h-3.5" /> {s.label}
        </span>
      </div>

      <div className="flex items-baseline gap-2">
        <div className="font-heading text-4xl font-semibold text-[var(--primary)]" data-testid={`${testId}-percent`}>
          {riskPercent}%
        </div>
        <div className="text-xs text-[var(--muted)]">anemia likelihood</div>
      </div>

      <div className="h-1.5 w-full bg-[var(--surface-2)] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${Math.max(2, riskPercent)}%`,
            background: riskLabel === "High" ? "var(--accent-urgent)" : riskLabel === "Moderate" ? "var(--accent-heat)" : "var(--primary)",
          }}
        />
      </div>

      <div className="flex items-center justify-between text-xs text-[var(--muted)]">
        <span>Confidence</span>
        <span className="font-mono">{Math.round((confidence || 0) * 100)}%</span>
      </div>
    </div>
  );
}
