import React, { useEffect, useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { CheckCircle2, Droplet } from "lucide-react";
import { toast } from "sonner";
import { formatApiError } from "@/lib/api";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const justRegistered = !!location.state?.justRegistered;
  const [email, setEmail] = useState(location.state?.email || "");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [registeredBanner, setRegisteredBanner] = useState(justRegistered);

  useEffect(() => {
    if (justRegistered) {
      toast.success("Account created successfully — please sign in");
      // Clear the navigation state so a refresh doesn't keep re-showing it
      window.history.replaceState({}, document.title);
    }
  }, [justRegistered]);

  const submit = async (e) => {
    e.preventDefault();
    setErr("");
    setLoading(true);
    try {
      await login(email.trim(), password);
      toast.success("Welcome back");
      navigate(location.state?.from || "/screen", { replace: true });
    } catch (e) {
      const msg = formatApiError(e.response?.data?.detail) || e.message;
      setErr(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[80vh] grid md:grid-cols-2">
      <div className="hidden md:flex relative items-center justify-center bg-[var(--primary)] text-white p-12 dot-grid">
        <div className="max-w-md">
          <div className="w-10 h-10 rounded-xl bg-white/15 grid place-items-center mb-6">
            <Droplet className="w-5 h-5" />
          </div>
          <h2 className="font-heading text-3xl lg:text-4xl leading-tight">
            Two signals. One score.<br />Explainable AI for anemia.
          </h2>
          <p className="mt-4 text-white/80 leading-relaxed">
            Sign in to run a screening, view your Grad-CAM heatmaps, and download a structured report.
          </p>
        </div>
      </div>

      <div className="flex items-center justify-center px-5 py-12">
        <form onSubmit={submit} className="w-full max-w-md fade-up" data-testid="login-form">
          <h1 className="font-heading text-3xl">Welcome back</h1>
          <p className="text-sm text-[var(--muted)] mt-1">Sign in to continue your screenings.</p>

          {registeredBanner && (
            <div
              className="mt-5 rounded-xl border border-[var(--primary)]/30 bg-[var(--secondary)] px-4 py-3 text-sm flex items-start gap-2"
              data-testid="register-success-banner"
            >
              <CheckCircle2 className="w-4 h-4 mt-0.5 text-[var(--primary)] flex-shrink-0" />
              <div>
                <div className="font-medium text-[var(--fg)]">Account created successfully</div>
                <div className="text-[var(--muted)] text-xs mt-0.5">Sign in below to start your first screening.</div>
              </div>
              <button
                type="button"
                className="ml-auto text-xs text-[var(--muted)] hover:text-[var(--fg)]"
                onClick={() => setRegisteredBanner(false)}
                aria-label="Dismiss"
              >×</button>
            </div>
          )}

          <div className="mt-6 space-y-4">
            <div>
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" required value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                data-testid="login-email-input" className="mt-1.5 rounded-xl" />
            </div>
            <div>
              <Label htmlFor="password">Password</Label>
              <Input id="password" type="password" required value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                data-testid="login-password-input" className="mt-1.5 rounded-xl" />
            </div>
            {err && <div className="text-sm text-[var(--accent-urgent)]" data-testid="login-error">{err}</div>}

            <Button type="submit" disabled={loading}
              className="w-full bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white rounded-xl"
              data-testid="login-submit-btn">
              {loading ? "Signing in…" : "Sign in"}
            </Button>

            <div className="text-sm text-[var(--muted)] text-center">
              Don&apos;t have an account?{" "}
              <Link to="/register" className="text-[var(--primary)] underline-offset-4 hover:underline" data-testid="login-register-link">
                Create one
              </Link>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
