import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { useDataset } from "../context/DatasetContext.jsx";
import { Card, ErrorBox, NeedDataset, PageHeader, Spinner } from "../components/UI.jsx";

export default function ReportPage() {
  const { id, status, results, setResult, refresh } = useDataset();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const rep = results.report;

  if (!id) return <NeedDataset />;

  async function generate() {
    setBusy(true); setError("");
    try {
      setResult("report", await api.generateReport(id));
      await refresh(id);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  const parts = [
    { label: "Dataset overview and column statistics", ok: true, link: "/upload" },
    { label: "Data preprocessing log (ML and DL cleaning)", ok: status.cleaned, link: "/preprocessing" },
    { label: "Exploratory data analysis charts", ok: true, link: "/eda" },
    { label: "Random Forest results and predictions", ok: status.trained, link: "/machine-learning" },
    { label: "AI insights and feature effects", ok: true, link: "/insights" },
  ];

  return (
    <div>
      <PageHeader step={6} tone="forest" title="Automated Report Generation" desc="Bundle everything into one downloadable PDF: cleaning log, charts, model metrics, explanations and next steps." />
      <ErrorBox error={error} />
      <Card tone="forest" title="What the report will contain">
        <ul className="space-y-2">
          {parts.map((p) => (
            <li key={p.label} className="flex items-center gap-3 text-sm">
              <span className={`grid h-5 w-5 place-items-center rounded-full text-xs font-bold ${p.ok ? "bg-emerald-500 text-black" : "border border-slate-600 text-slate-400"}`}>{p.ok ? "✓" : "–"}</span>
              <span className={p.ok ? "text-slate-100" : "text-mute"}>{p.label}</span>
              {!p.ok && <Link to={p.link} className="text-xs text-emerald-400 underline">complete this step</Link>}
            </li>
          ))}
        </ul>
        <div className="mt-5 flex flex-wrap items-center gap-4">
          <button className="btn-primary" onClick={generate} disabled={busy}>{rep || status.report ? "Regenerate PDF report" : "Generate PDF report"}</button>
          {busy && <Spinner label="Rendering charts and building the PDF..." />}
        </div>
      </Card>

      {(rep || status.report) && (
        <Card title="Your report is ready" className="mt-6">
          {rep && (
            <p className="mb-3 text-sm text-slate-300">
              {rep.filename} · {rep.size_kb} KB · generated {new Date(rep.generated_at).toLocaleString()}
              <br /><span className="text-xs text-mute">Sections: {rep.sections.join(", ")}</span>
            </p>
          )}
          <a className="btn-primary" href={api.url.report(id)}>Download PDF</a>
        </Card>
      )}
    </div>
  );
}
