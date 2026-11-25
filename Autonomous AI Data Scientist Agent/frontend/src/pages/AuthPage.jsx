import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function AuthPage({ mode }) {
  const isCreateAccount = mode === "create-account";
  const { login, register } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    if (isCreateAccount && form.get("password") !== form.get("confirmPassword")) {
      setMessage("The passwords do not match.");
      return;
    }
    setMessage("");
    setBusy(true);
    try {
      if (isCreateAccount) {
        await register({
          name: form.get("name"),
          email: form.get("email"),
          password: form.get("password"),
        });
      } else {
        await login({ email: form.get("email"), password: form.get("password") });
      }
      navigate(location.state?.from?.pathname || "/", { replace: true });
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-story">
        <Link to="/" className="auth-brand">
          <span className="brand-mark">ML</span>
          <span>ML Insight Studio</span>
        </Link>
        <div className="auth-story-copy">
          <span className="auth-kicker"><span /> A CLEARER VIEW OF YOUR DATA</span>
          <h1>Good decisions start with a better question.</h1>
          <p>Bring your datasets into one thoughtful workspace. Explore patterns, build models, and see what matters.</p>
        </div>
        <div className="auth-visual" aria-hidden="true">
          <div className="auth-visual-glow" />
          <div className="auth-chart">
            <div className="auth-chart-head"><span /><span /><span /></div>
            <div className="auth-chart-content">
              <div className="auth-chart-line"><i /><i /><i /><i /><i /></div>
              <div className="auth-chart-bars"><i /><i /><i /><i /><i /><i /></div>
            </div>
          </div>
          <div className="auth-orbit"><span>✦</span></div>
          <div className="auth-floating-note"><span className="status-dot" /> Insights, in focus</div>
        </div>
        <div className="auth-story-foot">Thoughtful analysis. Clearer next steps.</div>
      </section>

      <section className="auth-panel">
        <div className="auth-panel-top">
          <Link to="/" className="auth-back">← <span>Back to workspace</span></Link>
          <span className="auth-secure-label"><span>✦</span> PRIVATE WORKSPACE</span>
        </div>

        <div className="auth-form-wrap">
          <div className="auth-mobile-mark brand-mark">ML</div>
          <span className="auth-form-kicker">{isCreateAccount ? "START YOUR WORKSPACE" : "WELCOME BACK"}</span>
          <h2>{isCreateAccount ? "Create your account" : "Sign in to your account"}</h2>
          <p className="auth-form-intro">{isCreateAccount ? "Set up your space for clearer data decisions." : "Pick up where your data story left off."}</p>

          <form onSubmit={submit} className="auth-form">
            {isCreateAccount && (
              <label className="auth-field">
                <span>Your name</span>
                <input className="field" name="name" type="text" autoComplete="name" placeholder="Alex Morgan" required />
              </label>
            )}
            <label className="auth-field">
              <span>Email address</span>
              <input className="field" name="email" type="email" autoComplete="email" placeholder="you@example.com" required />
            </label>
            <label className="auth-field">
              <span>Password</span>
              <span className="auth-password-wrap">
                <input className="field" name="password" type={showPassword ? "text" : "password"}
                  autoComplete={isCreateAccount ? "new-password" : "current-password"}
                  placeholder={isCreateAccount ? "At least 12 characters" : "Enter your password"}
                  minLength={isCreateAccount ? 12 : undefined} required />
                <button type="button" className="password-toggle" onClick={() => setShowPassword((value) => !value)}
                  aria-label={showPassword ? "Hide password" : "Show password"}>
                  {showPassword ? "Hide" : "Show"}
                </button>
              </span>
            </label>
            {isCreateAccount && (
              <label className="auth-field">
                <span>Confirm password</span>
                <input className="field" name="confirmPassword" type={showPassword ? "text" : "password"}
                  autoComplete="new-password" placeholder="Enter your password again" minLength={12} required />
              </label>
            )}
            {message && <p className="auth-notice" role="alert">{message}</p>}
            <button type="submit" className="auth-submit" disabled={busy}>
              {busy ? "Please wait…" : isCreateAccount ? "Create account" : "Sign in"} {!busy && <span>→</span>}
            </button>
          </form>

          <p className="auth-switch">
            {isCreateAccount ? "Already have an account?" : "New to ML Insight Studio?"}{" "}
            <Link to={isCreateAccount ? "/login" : "/create-account"}>
              {isCreateAccount ? "Sign in" : "Create an account"}
            </Link>
          </p>
          <p className="auth-demo-note">Passwords are stored as one-way hashes. Session tokens are kept in HTTP-only cookies.</p>
        </div>
      </section>
    </main>
  );
}
