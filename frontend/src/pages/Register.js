import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Droplet } from "lucide-react";
import { toast } from "sonner";
import { formatApiError } from "@/lib/api";

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setErr("");
    setLoading(true);
    try {
      await register(name.trim(), email.trim(), password);
      navigate("/login", {
        replace: true,
        state: { justRegistered: true, email: email.trim() },
      });
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
            Start screening<br /> in under a minute.
          </h2>
          <p className="mt-4 text-white/80 leading-relaxed">
            Create a free research account to upload eye and nail images, run the fusion model, and keep a history.
          </p>
        </div>
      </div>

      <div className="flex items-center justify-center px-5 py-12">
        <form onSubmit={submit} className="w-full max-w-md fade-up" data-testid="register-form">
          <h1 className="font-heading text-3xl">Create account</h1>
          <p className="text-sm text-[var(--muted)] mt-1">Free for research and demo use.</p>

          <div className="mt-6 space-y-4">
            <div>
              <Label htmlFor="name">Full name</Label>
              <Input id="name" required value={name} onChange={(e) => setName(e.target.value)}
                placeholder="Dr. Maya Patel"
                data-testid="register-name-input" className="mt-1.5 rounded-xl" />
            </div>
            <div>
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                data-testid="register-email-input" className="mt-1.5 rounded-xl" />
            </div>
            <div>
              <Label htmlFor="password">Password</Label>
              <Input id="password" type="password" required minLength={6} value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="At least 6 characters"
                data-testid="register-password-input" className="mt-1.5 rounded-xl" />
            </div>
            {err && <div className="text-sm text-[var(--accent-urgent)]" data-testid="register-error">{err}</div>}

            <Button type="submit" disabled={loading}
              className="w-full bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white rounded-xl"
              data-testid="register-submit-btn">
              {loading ? "Creating…" : "Create account"}
            </Button>

            <div className="text-sm text-[var(--muted)] text-center">
              Already registered?{" "}
              <Link to="/login" className="text-[var(--primary)] underline-offset-4 hover:underline" data-testid="register-login-link">
                Sign in
              </Link>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
