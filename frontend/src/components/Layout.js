import React from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Droplet, History, BookOpen, LogOut, Activity, Brain } from "lucide-react";

export default function Layout({ children }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const navItem = ({ isActive }) =>
    `px-3 py-2 rounded-xl text-sm font-medium transition ${
      isActive ? "bg-[var(--surface-2)] text-[var(--primary)]" : "text-[var(--fg)] hover:bg-[var(--surface-2)]"
    }`;

  return (
    <div className="min-h-screen flex flex-col" data-testid="app-shell">
      <header className="border-b border-[var(--border)] bg-[var(--surface)] sticky top-0 z-30">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-5 py-3">
          <Link to="/" className="flex items-center gap-2" data-testid="nav-home">
            <div className="w-9 h-9 rounded-xl bg-[var(--primary)] grid place-items-center text-white">
              <Droplet className="w-5 h-5" strokeWidth={2.2} />
            </div>
            <div className="leading-tight">
              <div className="font-heading text-base font-semibold tracking-tight">HemaScan</div>
              <div className="text-[10px] uppercase tracking-[0.25em] text-[var(--muted)]">Anemia AI</div>
            </div>
          </Link>
          {user ? (
            <nav className="hidden md:flex items-center gap-1">
              <NavLink to="/screen" className={navItem} data-testid="nav-screen">
                <span className="inline-flex items-center gap-2"><Activity className="w-4 h-4" /> New scan</span>
              </NavLink>
              <NavLink to="/history" className={navItem} data-testid="nav-history">
                <span className="inline-flex items-center gap-2"><History className="w-4 h-4" /> History</span>
              </NavLink>
              <NavLink to="/learn" className={navItem} data-testid="nav-learn">
                <span className="inline-flex items-center gap-2"><BookOpen className="w-4 h-4" /> Learn</span>
              </NavLink>
              {user.role === "admin" && (
                <NavLink to="/admin/models" className={navItem} data-testid="nav-models">
                  <span className="inline-flex items-center gap-2"><Brain className="w-4 h-4" /> Models</span>
                </NavLink>
              )}
            </nav>
          ) : null}
          <div className="flex items-center gap-2">
            {user ? (
              <>
                <div className="hidden sm:block text-right">
                  <div className="text-sm font-medium">{user.name || user.email}</div>
                  <div className="text-[11px] text-[var(--muted)]">{user.email}</div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={async () => { await logout(); navigate("/login"); }}
                  data-testid="logout-btn"
                  className="rounded-xl"
                >
                  <LogOut className="w-4 h-4 mr-1" /> Logout
                </Button>
              </>
            ) : (
              <>
                <Link to="/login"><Button variant="ghost" size="sm" data-testid="header-login-btn">Login</Button></Link>
                <Link to="/register">
                  <Button size="sm" className="bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white rounded-xl" data-testid="header-register-btn">
                    Get started
                  </Button>
                </Link>
              </>
            )}
          </div>
        </div>
      </header>
      <main className="flex-1 w-full">{children}</main>
      <footer className="border-t border-[var(--border)] bg-[var(--surface)] py-6">
        <div className="max-w-6xl mx-auto px-5 text-xs text-[var(--muted)] flex flex-col sm:flex-row justify-between gap-2">
          <div>© {new Date().getFullYear()} HemaScan • Research prototype, not a medical device.</div>
          <div className="font-mono">AI vision · Visual heatmaps · Combined score · Mobile-ready</div>
        </div>
      </footer>
    </div>
  );
}
