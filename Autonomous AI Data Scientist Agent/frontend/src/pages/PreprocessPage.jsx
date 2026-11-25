import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, Cell } from "recharts";
import { api } from "../api.js";
import { useDataset } from "../context/DatasetContext.jsx";
import { Badge, Card, COLORS, DataTable, ErrorBox, Field, NeedDataset, OptionGroup, PageHeader, Spinner, Stat, Toggle, fmt, tooltipStyle } from "../components/UI.jsx";

const IMPUTE = [
  { value: "iterative", label: "Iterative ML imputer", desc: "ExtraTrees predicts missing cells from other columns." },
  { value: "knn", label: "KNN imputer", desc: "Uses values from the 5 most similar rows." },
  { value: "mean", label: "Mean / most frequent", desc: "Mean for numeric columns; most frequent for categories." },
  { value: "median", label: "Median / most frequent", desc: "Median for numeric columns; most frequent for categories." },
  { value: "most_frequent", label: "Most frequent", desc: "Uses the mode for both numeric and categorical columns." },
  { value: "simple", label: "Simple baseline", desc: "Median for numeric columns; most frequent for categories." },
];
const ENCODINGS = [
  { value: "onehot", label: "One-hot encoding", desc: "Creates indicator features; recommended for nominal categories." },
  { value: "label", label: "Label encoding", desc: "Assigns integer codes by first appearance; codes do not represent rank." },
  { value: "ordinal", label: "Ordinal encoding", desc: "Assigns integer codes in sorted category order." },
];
const OUTLIER = [
  { value: "reconstruct", label: "Autoencoder repair (deep learning)", desc: "Replaces corrupted cells with the autoencoder's reconstruction." },
  { value: "cap", label: "Cap extreme values", desc: "Clips outlier cells to the IQR fences." },
  { value: "remove", label: "Remove outlier rows", desc: "Drops rows confirmed as anomalies." },
  { value: "none", label: "Detect only", desc: "Report outliers but leave the data untouched." },
];

export default function PreprocessPage() {
  const { id, info, meta, columns, status, results, setResult, invalidateFrom, refresh } = useDataset();
  const [opts, setOpts] = useState({
    target: "", imputation: "iterative", outlier_treatment: "reconstruct", drop_duplicates: true,
    missing_col_threshold: 0.6, contamination: 0.03, categorical_encoding: "onehot",
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("after");
  const prep = results.prep;

  // load the previous result when coming back to this page
  useEffect(() => {
    if (id && status.cleaned && !results.prep) api.preprocessResult(id).then((r) => setResult("prep", r)).catch(() => {});
  }, [id, status.cleaned]); // eslint-disable-line

  useEffect(() => { if (meta?.target) setOpts((o) => ({ ...o, target: o.target || meta.target })); }, [meta]);
  useEffect(() => {
    if (!prep?.options) return;
    const savedOptions = { ...prep.options };
    delete savedOptions.ordinal_orders;
    if (typeof savedOptions.categorical_encoding !== "string") savedOptions.categorical_encoding = "onehot";
    setOpts((o) => ({
      ...o,
      ...savedOptions,
    }));
  }, [prep]);

  if (!id) return <NeedDataset />;

  async function run() {
    setBusy(true); setError("");
    try {
      const r = await api.preprocess(id, { ...opts, target: opts.target || null });
      invalidateFrom(["eda", "ml", "insights", "report"]);
      setResult("prep", r);
      await refresh(id);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  const s = prep?.summary;
  const o = prep?.outliers;
  const missingCells = info?.missing_cells ?? columns.reduce((total, column) => total + column.missing, 0);
  const duplicateRows = info?.duplicates ?? 0;
  const missingColumns = columns
    .filter((column) => column.missing > 0)
    .map((column) => ({
      column: column.name,
      missing: column.missing,
      missing_percent: `${((column.missing / Math.max(meta?.rows ?? 0, 1)) * 100).toFixed(1)}%`,
    }));
  const outlierCandidateColumns = columns.filter(
    (column) => column.kind === "numeric" && column.unique > 5 && column.name !== opts.target,
  ).length;
  const meetsOutlierRequirements = (meta?.rows ?? 0) >= 50 && outlierCandidateColumns >= 2;
  return (
    <div>
      <PageHeader step={2} tone="forest" title="Data Preprocessing" desc="ML imputes missing values, while Isolation Forest and an autoencoder detect and treat numeric anomalies. Exact duplicates and basic format cleanup use transparent rules." />
      <ErrorBox error={error} />

      <Card title="Dataset quality scan" subtitle={`Checked from ${meta.filename} before cleaning`}>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
          <Stat label="Rows" value={meta.rows.toLocaleString()} />
          <Stat label="Missing cells" value={missingCells.toLocaleString()} accent={missingCells ? "text-amber-300" : "text-emerald-300"} />
          <Stat label="Exact duplicate rows" value={duplicateRows.toLocaleString()} accent={duplicateRows ? "text-amber-300" : "text-emerald-300"} />
        </div>
        <div className="mt-4">
          <h2 className="mb-2 text-sm font-semibold text-white">Columns with missing values</h2>
          {missingColumns.length > 0
            ? <DataTable rows={missingColumns} columns={["column", "missing", "missing_percent"]} maxHeight="max-h-56" />
            : <p className="text-sm text-mute">No missing values found in the uploaded dataset.</p>}
        </div>
      </Card>

      <Card tone="forest" title="Cleaning options">
        <div className="grid gap-5">
          <p className="text-sm text-mute">
            Choose one categorical encoding for all features during model training. Classification targets are label-encoded automatically; cleaned CSV/Excel exports keep category labels readable. Duplicate removal uses exact row matching.
          </p>
          <Field label="Target column (optional)" hint="The column you want to predict later. Rows without a target are removed and the column is never altered.">
            <select className="field" value={opts.target} onChange={(e) => setOpts({ ...opts, target: e.target.value })}>
              <option value="">No target yet</option>
              {columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </Field>
          {missingCells > 0 ? (
            <div>
              <span className="label">Missing value imputation (machine learning)</span>
              <OptionGroup value={opts.imputation} onChange={(v) => setOpts({ ...opts, imputation: v })} options={IMPUTE} />
            </div>
          ) : (
            <p className="text-sm text-mute">No missing values to impute; imputation options are not needed for this file.</p>
          )}
          <div>
            <span className="label">Categorical feature encoding</span>
            <p className="mb-3 text-xs text-mute">One choice applies to every categorical feature during model training. Cleaned CSV/Excel exports keep readable labels. Click Clean dataset to save these settings. Ordinal codes use sorted category values.</p>
            <OptionGroup value={opts.categorical_encoding} onChange={(v) => setOpts((o) => ({ ...o, categorical_encoding: v }))} options={ENCODINGS} />
          </div>
          <div><span className="label">Outlier treatment (Isolation Forest + autoencoder)</span><OptionGroup value={opts.outlier_treatment} onChange={(v) => setOpts({ ...opts, outlier_treatment: v })} options={OUTLIER} /></div>
          <p className={`text-xs ${meetsOutlierRequirements ? "text-mute" : "text-amber-700"}`}>
            {meetsOutlierRequirements
              ? `The raw profile meets the initial outlier checks: ${meta.rows.toLocaleString()} rows and ${outlierCandidateColumns} candidate numeric columns. Cleaning may change eligibility.`
              : `Outlier detection needs at least 50 rows and 2 suitable numeric columns. This file has ${meta.rows.toLocaleString()} rows and ${outlierCandidateColumns} candidates; it may be skipped.`}
          </p>
          {missingCells > 0 && (
            <Field label={`Drop columns with more than ${Math.round(opts.missing_col_threshold * 100)}% missing`}>
              <input type="range" min="0.2" max="1" step="0.05" value={opts.missing_col_threshold} onChange={(e) => setOpts({ ...opts, missing_col_threshold: +e.target.value })} className="w-full accent-emerald-500" />
            </Field>
          )}
          {duplicateRows > 0 ? (
            <Toggle checked={opts.drop_duplicates} onChange={(v) => setOpts({ ...opts, drop_duplicates: v })} label={`Remove ${duplicateRows.toLocaleString()} exact duplicate rows`} />
          ) : (
            <p className="text-sm text-mute">No exact duplicate rows to remove.</p>
          )}
          {meetsOutlierRequirements && (
            <Field label={`Expected outlier share (Isolation Forest): ${(opts.contamination * 100).toFixed(1)}%`}>
              <input type="range" min="0.005" max="0.15" step="0.005" value={opts.contamination} onChange={(e) => setOpts({ ...opts, contamination: +e.target.value })} className="w-full accent-emerald-500" />
            </Field>
          )}
          <div className="flex items-center gap-4">
            <button className="btn-primary" disabled={busy} onClick={run}>{prep ? "Clean again" : "Clean dataset"}</button>
            {busy && <Spinner label="Training imputer and autoencoder, this can take a few seconds..." />}
          </div>
        </div>
      </Card>

      {prep && (
        <>
          <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Rows" value={`${s.rows_before.toLocaleString()} → ${s.rows_after.toLocaleString()}`} />
            <Stat label="Columns" value={`${s.cols_before} → ${s.cols_after}`} />
            <Stat label="Missing cells" value={`${s.missing_before.toLocaleString()} → ${s.missing_after.toLocaleString()}`} accent="text-emerald-300" />
            <Stat label="Completeness" value={`${s.completeness_before}% → ${s.completeness_after}%`} accent="text-emerald-300" />
            <Stat label="Cells imputed by ML" value={s.cells_imputed.toLocaleString()} />
            <Stat label="Cells repaired (DL)" value={s.cells_repaired.toLocaleString()} />
            <Stat label="Outlier rows removed" value={s.rows_removed_outliers.toLocaleString()} />
            <Stat label="Duplicates removed" value={s.duplicates_removed.toLocaleString()} />
          </div>

          <Card title="What happened, step by step" className="mt-6">
            <ol className="space-y-3">
              {prep.steps.map((st, i) => (
                <li key={i} className={`rounded-lg border border-slate-800 p-3 ${st.status === "skipped" ? "opacity-60" : ""}`}>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold text-white">{st.title}</span>
                    <Badge kind={st.method}>{st.method}</Badge>
                    {st.status === "skipped" && <span className="text-xs text-mute">nothing to do</span>}
                  </div>
                  <p className="mt-1 text-sm text-slate-300">{st.detail}</p>
                  {st.items.length > 0 && <p className="mt-1 text-xs text-mute">{st.items.slice(0, 8).join(" · ")}</p>}
                </li>
              ))}
            </ol>
          </Card>

          {o?.ran && (
            <Card tone="forest" title="Anomaly detection" subtitle={`Autoencoder backend: ${o.autoencoder_backend}`} className="mt-6">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="Isolation Forest" value={o.isolation_forest_flagged} sub="rows flagged" />
                <Stat label="Autoencoder" value={o.autoencoder_flagged} sub="rows flagged" />
                <Stat label="Range check" value={o.range_check_flagged} sub="rows flagged" />
                <Stat label="Confirmed" value={o.consensus_flagged} sub="by voting" accent="text-amber-300" />
              </div>
              <div className="mt-5 grid gap-5 lg:grid-cols-2">
                <div>
                  <p className="mb-2 text-xs text-mute">Rows projected to 2-D (PCA). Amber points are confirmed outliers.</p>
                  <div className="h-64">
                    <ResponsiveContainer>
                      <ScatterChart margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
                        <CartesianGrid stroke={COLORS.grid} />
                        <XAxis type="number" dataKey="x" stroke={COLORS.axis} tick={{ fontSize: 10 }} />
                        <YAxis type="number" dataKey="y" stroke={COLORS.axis} tick={{ fontSize: 10 }} />
                        <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
                        <Scatter data={o.projection}>
                          {o.projection.map((p, i) => <Cell key={i} fill={p.outlier ? COLORS.amber : COLORS.blue} fillOpacity={p.outlier ? 1 : 0.45} />)}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div>
                  <p className="mb-2 text-xs text-mute">Autoencoder training loss (lower means it learned what normal rows look like).</p>
                  <div className="h-64">
                    <ResponsiveContainer>
                      <LineChart data={o.loss_curve.map((v, i) => ({ step: i + 1, loss: v }))} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
                        <CartesianGrid stroke={COLORS.grid} />
                        <XAxis dataKey="step" stroke={COLORS.axis} tick={{ fontSize: 10 }} />
                        <YAxis stroke={COLORS.axis} tick={{ fontSize: 10 }} />
                        <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
                        <Line type="monotone" dataKey="loss" stroke={COLORS.green} dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
              {o.samples.length > 0 && (
                <div className="mt-5">
                  <p className="mb-2 text-xs text-mute">Examples of repaired cells</p>
                  <DataTable rows={o.samples} columns={["row", "column", "before", "after"]} maxHeight="max-h-56" />
                </div>
              )}
            </Card>
          )}

          {prep.imputation?.columns?.length > 0 && (
            <Card title="Imputed columns" subtitle={prep.imputation.method} className="mt-6">
              <DataTable rows={prep.imputation.columns} columns={["column", "kind", "filled"]} maxHeight="max-h-64" />
            </Card>
          )}

          <Card title="Before and after" className="mt-6" actions={
            <div className="flex gap-2">
              {["before", "after"].map((t) => <button key={t} onClick={() => setTab(t)} className={`rounded-md border px-3 py-1 text-xs ${tab === t ? "border-emerald-500 text-white" : "border-slate-700 text-mute"}`}>{t === "before" ? "Raw" : "Cleaned"}</button>)}
            </div>}>
            <DataTable rows={tab === "before" ? prep.preview_before : prep.preview_after} />
            <div className="mt-4 flex flex-wrap gap-3">
              <a className="btn-ghost" href={api.url.cleaned(id, "csv")}>Download cleaned CSV</a>
              <a className="btn-ghost" href={api.url.cleaned(id, "xlsx")}>Download cleaned Excel</a>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
