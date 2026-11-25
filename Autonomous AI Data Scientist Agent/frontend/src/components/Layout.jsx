import { useState } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { STEPS } from "../steps.js";
import { useDataset } from "../context/DatasetContext.jsx";
import { useAuth } from "../context/AuthContext.jsx";

const WORKFLOW_ICONS = {
  upload: <><path d="M12 15V4" /><path d="m7 9 5-5 5 5" /><path d="M5 14v5h14v-5" /></>,
  preprocess: <><path d="M4 6h16M4 12h16M4 18h16" /><circle cx="9" cy="6" r="2" /><circle cx="15" cy="12" r="2" /><circle cx="7" cy="18" r="2" /></>,
  eda: <><path d="M4 19V5M4 19h16" /><path d="M8 16v-4M12 16V8M16 16v-6" /></>,
  ml: <><path d="m7 7 5 5 5-5M12 12v5" /><circle cx="7" cy="6" r="2" /><circle cx="17" cy="6" r="2" /><circle cx="12" cy="18" r="2" /></>,
  insights: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" /></>,
  report: <><path d="M7 3h7l4 4v14H7z" /><path d="M14 3v5h5M10 12h5M10 16h5" /></>,
};

function WorkflowIcon({ step, complete }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      {complete ? <path d="m5 12 4 4L19 6" /> : WORKFLOW_ICONS[step]}
    </svg>
  );
}

function useDone() {
  const { id, status, results } = useDataset();
  return {
    upload: !!id,
    preprocess: !!status.cleaned,
    eda: !!results.eda,
    ml: !!status.trained,
    insights: !!results.insights,
    report: !!status.report,
  };
}

export default function Layout({ children }) {
  const { meta, clearDataset } = useDataset();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [logoutError, setLogoutError] = useState("");
  const done = useDone();
  const { pathname } = useLocation();
  const idx = STEPS.findIndex((s) => s.path === pathname);
  const prev = idx > 0 ? STEPS[idx - 1] : null;
  const next = idx >= 0 && idx < STEPS.length - 1 ? STEPS[idx + 1] : null;

  async function signOut() {
    setLogoutError("");
    try {
      await logout();
      navigate("/login", { replace: true });
    } catch (error) {
      setLogoutError(error.message);
    }
  }

  return (
    <div className="app-shell min-h-screen">
      <aside className="sidebar">
        <Link to="/" className="brand-lockup">
          <span className="brand-mark">ML</span>
          <span>
            <span className="block text-sm font-bold tracking-tight text-white">ML Insight Studio</span>
            <span className="mt-0.5 block text-xs text-slate-400">Your data, in perspective</span>
          </span>
        </Link>

        <div className="sidebar-label">YOUR WORKFLOW</div>
        <nav aria-label="Workspace navigation" className="space-y-1">
          {STEPS.map((s) => (
            <NavLink key={s.id} to={s.path} className={({ isActive }) => `side-link ${isActive ? "side-link-active" : ""}`}>
              <span className={`side-step ${done[s.key] ? "side-step-done" : ""}`}>
                {done[s.key] ? "✓" : s.icon}
              </span>
              <span className="min-w-0">
                <span className="block truncate text-sm font-medium">{s.title.replace(" (EDA)", "")}</span>
                <span className="mt-0.5 block truncate text-[11px] opacity-70">{s.desc}</span>
              </span>
            </NavLink>
          ))}
          {user && (
            <NavLink to="/profile" className={({ isActive }) => `side-link side-link-profile ${isActive ? "side-link-active" : ""}`}>
              <span className="side-step" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="8" r="3.5" />
                  <path d="M5 20c.7-3.5 3.2-5.3 7-5.3s6.3 1.8 7 5.3" />
                </svg>
              </span>
              <span className="block truncate text-sm font-medium">Profile</span>
            </NavLink>
          )}
        </nav>

        <div className="sidebar-foot">
          <span className="status-dot" />
          <span>{meta ? "Workspace ready" : "Ready for your first dataset"}</span>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <Link to="/" className="mobile-brand">
            <span className="brand-mark">ML</span>
            <span className="mobile-brand-name font-bold text-slate-900">ML Insight</span>
          </Link>
          <nav className="topbar-workflow" aria-label="Workflow stages">
            <div className="topbar-flow-copy">
              <span className="topbar-flow-kicker"><span className="flow-live-dot" /> YOUR ANALYSIS</span>
              <span className="topbar-flow-caption">{meta ? "Taking shape" : "Ready when you are"}</span>
            </div>
            <div className="topbar-flow-track">
              {STEPS.map((s) => (
                <NavLink key={s.id} to={s.path} title={s.title}
                  aria-label={`${s.title}${done[s.key] ? " — completed" : ""}`}
                  className={({ isActive }) => `topbar-flow-step ${isActive ? "topbar-flow-step-active" : ""} ${done[s.key] ? "topbar-flow-step-done" : ""}`}>
                  <WorkflowIcon step={s.key} complete={done[s.key]} />
                </NavLink>
              ))}
            </div>
          </nav>
          <div className="ml-auto flex items-center gap-3">
            {meta ? (
              <button onClick={clearDataset} className="btn-ghost !px-3 !py-1.5 text-xs">New dataset</button>
            ) : <span className="workspace-idle"><span className="status-dot" /><span className="idle-text">No dataset loaded</span></span>}
            {user ? (
              <button type="button" onClick={signOut} className="topbar-account-link">Sign out</button>
            ) : <Link to="/login" className="topbar-account-link"><span className="account-link-icon">◉</span><span className="account-link-label">Sign in</span></Link>}
          </div>
        </header>
        {logoutError && <p className="auth-notice mx-5 mt-3" role="alert">{logoutError}</p>}

        <div className="mobile-pipeline" aria-label="Pipeline navigation">
          {STEPS.map((s) => (
            <NavLink key={s.id} to={s.path} aria-label={s.title}
              className={({ isActive }) => `mobile-step ${isActive ? "mobile-step-active" : ""} ${done[s.key] ? "mobile-step-done" : ""}`}>
              {done[s.key] ? "✓" : s.icon}
            </NavLink>
          ))}
        </div>

        <main className="page-content">
          {children}
          {idx >= 0 && pathname !== "/insights" && (
            <div className="mt-8 flex items-center justify-between border-t border-slate-200 pt-5">
              {prev ? <Link to={prev.path} className="btn-ghost">← {prev.title.replace(" (EDA)", "")}</Link> : <span />}
              {next ? <Link to={next.path} className="btn-primary">{next.title.replace(" (EDA)", "")} →</Link> : <span />}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
