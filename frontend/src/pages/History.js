import React, { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import api from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Trash2, Eye, Hand, ArrowRight, Activity } from "lucide-react";
import { toast } from "sonner";

export default function History() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const location = useLocation();

  const load = async () => {
    setLoading(true);
    try {
      const { data } = await api.get("/screenings");
      setItems(data);
    } catch {
      toast.error("Could not load history");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [location.key]);

  const remove = async (id) => {
    if (!window.confirm("Delete this screening?")) return;
    try {
      await api.delete(`/screenings/${id}`);
      setItems((s) => s.filter((x) => x.id !== id));
      toast.success("Deleted");
    } catch {
      toast.error("Failed to delete");
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-5 py-10">
      <div className="flex items-end justify-between mb-6 fade-up">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-[var(--muted)]">History</div>
          <h1 className="font-heading text-4xl sm:text-5xl tracking-tight mt-1">Patient records</h1>
        </div>
        <Link to="/screen">
          <Button className="rounded-xl bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white" data-testid="history-new-btn">
            <Activity className="w-4 h-4 mr-2" /> New screening
          </Button>
        </Link>
      </div>

      {loading ? (
        <div className="text-[var(--muted)] text-sm">Loading…</div>
      ) : items.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-[var(--border)] bg-[var(--surface)] p-12 text-center" data-testid="history-empty">
          <h3 className="font-heading text-2xl">No screenings yet</h3>
          <p className="text-sm text-[var(--muted)] mt-2">Run your first screening to see it here.</p>
          <Link to="/screen" className="inline-block mt-4">
            <Button className="rounded-xl bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white">Start</Button>
          </Link>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5 fade-up delay-1" data-testid="history-list">
          {items.map((s) => {
            const risk = s.fusion_result?.risk_percent ?? 0;
            const label = s.fusion_result?.risk_label || "Low";
            const pill = label === "High" ? "risk-high" : label === "Moderate" ? "risk-mod" : "risk-low";
            return (
              <div key={s.id} className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5 flex flex-col" data-testid={`history-card-${s.id}`}>
                <div className="flex items-center justify-between">
                  <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${pill}`}>{label}</span>
                  <button onClick={() => remove(s.id)} className="text-[var(--muted)] hover:text-[var(--accent-urgent)]" data-testid={`history-delete-${s.id}`}>
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <div className="mt-3 flex items-baseline gap-2">
                  <div className="font-heading text-3xl font-semibold text-[var(--primary)]">{risk}%</div>
                  <div className="text-xs text-[var(--muted)]">fusion risk</div>
                </div>
                <div className="mt-1 font-heading text-lg">{s.patient_name || "Anonymous"}</div>
                <div className="text-xs text-[var(--muted)]">{new Date(s.created_at).toLocaleString()}</div>

                <div className="mt-3 flex items-center gap-3 text-xs text-[var(--muted)]">
                  {s.eye_result && <span className="inline-flex items-center gap-1"><Eye className="w-3.5 h-3.5" /> Eye {s.eye_result.risk_percent}%</span>}
                  {s.nail_result && <span className="inline-flex items-center gap-1"><Hand className="w-3.5 h-3.5" /> Nail {s.nail_result.risk_percent}%</span>}
                </div>

                <Link to={`/results/${s.id}`} className="mt-4 inline-flex items-center gap-1 text-sm text-[var(--primary)] hover:underline" data-testid={`history-view-${s.id}`}>
                  View report <ArrowRight className="w-4 h-4" />
                </Link>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
