import { Link } from "react-router-dom";
import { STEPS } from "../steps.js";
import { useDataset } from "../context/DatasetContext.jsx";

export default function Home() {
  const { id, meta, status, results, loading } = useDataset();
  const done = {
    upload: !!id,
    preprocess: !!status.cleaned,
    eda: !!results.eda,
    ml: !!status.trained,
    insights: !!results.insights,
    report: !!status.report,
  };
  const completed = Object.values(done).filter(Boolean).length;
  const nextStep = STEPS.find((step) => !done[step.key]) || STEPS[STEPS.length - 1];

  return (
    <div className="dashboard-page">
      <section className="welcome-banner">
        <div className="welcome-copy">
          <div className="eyebrow"><span className="eyebrow-line" /> YOUR DATA, MADE CLEAR</div>
          <h1>{meta ? "Your analysis, at a glance." : "Find the story in your data."}</h1>
          <p>{meta ? `Continue exploring ${meta.filename} or pick up where you left off.` : "Clean, explore, and understand your data—all in one place."}</p>
          <div className="banner-actions">
            <Link to={meta ? nextStep.path : "/upload"} className="banner-button">
              {meta ? `Continue to ${nextStep.title.replace(" (EDA)", "")}` : "Start with your dataset"} <span>→</span>
            </Link>
            {!meta && <Link to="/upload" state={{ loadSample: true }} className="banner-sample-link">Try sample data <span>↗</span></Link>}
          </div>
        </div>
        <div className="welcome-art" aria-hidden="true">
          <div className="art-orbit orbit-one" />
          <div className="art-orbit orbit-two" />
          <div className="art-center"><span>ML</span><i>✦</i></div>
          <span className="art-spark spark-one">✦</span>
          <span className="art-spark spark-two">✧</span>
          <span className="art-spark spark-three">·</span>
        </div>
        <div className="banner-caption"><span className="status-dot" /> From your first upload to clear decisions</div>
      </section>

      <section className="overview-row" aria-label="Dataset overview">
        <article className="overview-card">
          <div className="overview-icon icon-green">▤</div>
          <div><p>ACTIVE DATASET</p><strong className="truncate">{meta?.filename || "Nothing uploaded yet"}</strong><span>{meta ? "Ready to explore" : "Upload a file to begin"}</span></div>
        </article>
        <article className="overview-card">
          <div className="overview-icon icon-blue">⌗</div>
          <div><p>DATA POINTS</p><strong>{meta ? meta.rows.toLocaleString() : "—"}</strong><span>{meta ? `${meta.cols} features in your dataset` : "Rows will appear here"}</span></div>
        </article>
        <article className="overview-card">
          <div className="overview-icon icon-violet">◷</div>
          <div><p>PIPELINE PROGRESS</p><strong>{loading ? "…" : completed === STEPS.length ? "Complete" : completed ? "In motion" : "Ready to begin"}</strong><span>{completed ? "Keep building your analysis" : "Your journey starts here"}</span></div>
          <div className="progress-ring" role="img" aria-label="Pipeline completion progress" style={{ "--progress": `${(completed / STEPS.length) * 100}%` }}><span>✦</span></div>
        </article>
      </section>

      <section className="pipeline-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow section-eyebrow">THE WORKFLOW</div>
            <h2>Your path to insight</h2>
            <p>Move at your own pace. Every step brings your data into sharper focus.</p>
          </div>
          <span className="pipeline-count">{completed === STEPS.length ? "ALL CAUGHT UP" : "READY WHEN YOU ARE"}</span>
        </div>

        <div className="pipeline-grid">
          {STEPS.map((step, index) => (
            <Link key={step.id} to={step.path} className={`pipeline-card ${done[step.key] ? "pipeline-card-done" : ""}`}>
              <div className="pipeline-card-top">
                <span className={`pipeline-icon pipeline-icon-${index + 1}`}>{step.icon}</span>
                {done[step.key] && <span className="pipeline-status-mark" aria-label="Completed">✓</span>}
              </div>
              <h3>{step.title}</h3>
              <p>{step.desc}</p>
              <div className="pipeline-card-foot">
                <span className={done[step.key] ? "completion-label" : ""}>{done[step.key] ? "Completed" : "Explore step"}</span>
                <span className="card-arrow">{done[step.key] ? "✓" : "↗"}</span>
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
