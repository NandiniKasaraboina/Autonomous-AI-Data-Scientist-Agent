import { Link } from "react-router-dom";
import { useDataset } from "../context/DatasetContext.jsx";

export const COLORS = { green: "#0b916a", blue: "#4387d7", violet: "#8b68d0", amber: "#d39432", red: "#d86e7c", grid: "#e6ece8", axis: "#718096" };
export const tooltipStyle = { background: "#ffffff", border: "1px solid #e1e8e4", borderRadius: 10, color: "#26364b", fontSize: 12, boxShadow: "0 8px 24px #17243a18" };

export function fmt(v, d = 3) {
  if (v === null || v === undefined || v === "") return "–";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "–";
    if (Number.isInteger(v)) return v.toLocaleString();
    const a = Math.abs(v);
    return a >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 1 }) : Number(v.toPrecision(d)).toString();
  }
  return String(v);
}
export const pct = (v, d = 1) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`);

export function PageHeader({ title, desc, tone = "navy" }) {
  return (
    <div className={`${tone === "forest" ? "card-green" : "card"} mb-6 text-center`}>
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700">Your analysis workspace</p>
      <h1 className="mt-1 text-2xl font-bold text-white">{title}</h1>
      <p className="mx-auto mt-1 max-w-2xl text-sm text-mute">{desc}</p>
    </div>
  );
}

export function Card({ title, subtitle, children, tone = "navy", className = "", actions }) {
  return (
    <section className={`${tone === "forest" ? "card-green" : "card"} ${className}`}>
      {(title || actions) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-2">
          <div>
            {title && <h2 className="text-base font-semibold text-white">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-mute">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, accent }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-black/40 p-3">
      <p className="text-xs text-mute">{label}</p>
      <p className={`mt-1 text-xl font-bold ${accent || "text-white"}`}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-mute">{sub}</p>}
    </div>
  );
}

export function Spinner({ label }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-mute">
      <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
        <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
      {label}
    </span>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return (
    <div role="alert" className="mb-4 rounded-lg border border-red-900 bg-red-950/40 p-3 text-sm text-red-200">
      {error}
    </div>
  );
}

const BADGE = {
  "Rule-based": "border-slate-600 text-slate-300",
  "Machine learning": "border-blue-700 text-blue-300",
  "Deep learning": "border-violet-700 text-violet-300",
  "Machine learning + Deep learning": "border-violet-700 text-violet-300",
  good: "border-emerald-700 text-emerald-300",
  warning: "border-amber-700 text-amber-300",
  info: "border-blue-700 text-blue-300",
};
export function Badge({ kind, children }) {
  return <span className={`inline-block rounded-full border px-2 py-0.5 text-[11px] font-medium ${BADGE[kind] || BADGE["Rule-based"]}`}>{children}</span>;
}

/** Shown on pages that need an uploaded dataset first. */
export function NeedDataset() {
  const { loading } = useDataset();
  if (loading) return <Spinner label="Loading dataset..." />;
  return (
    <Card className="text-center">
      <p className="text-white">No dataset loaded yet.</p>
      <p className="mt-1 text-sm text-mute">Upload a CSV or Excel file to start the pipeline.</p>
      <Link to="/upload" className="btn-primary mt-4">Go to Dataset Upload</Link>
    </Card>
  );
}

export function DataTable({ rows, columns, maxHeight = "max-h-80" }) {
  if (!rows || rows.length === 0) return <p className="text-sm text-mute">No rows to show.</p>;
  const cols = columns || Object.keys(rows[0]);
  return (
    <div className={`${maxHeight} overflow-auto rounded-lg border border-slate-800`}>
      <table className="min-w-full text-left text-xs">
        <thead className="sticky top-0 bg-slate-950 text-mute">
          <tr>{cols.map((c) => <th key={c} className="whitespace-nowrap px-3 py-2 font-medium">{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-slate-800/80 hover:bg-white/5">
              {cols.map((c) => <td key={c} className="whitespace-nowrap px-3 py-1.5 text-slate-200">{r[c] === null || r[c] === undefined ? <span className="text-slate-600">null</span> : fmt(r[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-mute">{hint}</span>}
    </label>
  );
}

/** Radio cards used for choosing a method. */
export function OptionGroup({ value, onChange, options }) {
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {options.map((o) => (
        <button type="button" key={o.value} onClick={() => onChange(o.value)} aria-pressed={value === o.value}
          className={`rounded-lg border p-3 text-left transition ${value === o.value ? "border-emerald-500 bg-emerald-500/10" : "border-slate-700 hover:border-slate-500"}`}>
          <span className="block text-sm font-semibold text-white">{o.label}</span>
          <span className="mt-0.5 block text-xs text-mute">{o.desc}</span>
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label }) {
  return (
    <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)} className="flex items-center gap-3 text-sm text-slate-200">
      <span className={`relative h-5 w-9 rounded-full transition ${checked ? "bg-emerald-500" : "bg-slate-700"}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${checked ? "left-4" : "left-0.5"}`} />
      </span>
      {label}
    </button>
  );
}
