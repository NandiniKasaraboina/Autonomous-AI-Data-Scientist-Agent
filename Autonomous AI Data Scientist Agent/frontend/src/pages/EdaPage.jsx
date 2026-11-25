import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api.js";
import { useDataset } from "../context/DatasetContext.jsx";
import { Badge, Card, COLORS, DataTable, ErrorBox, Field, NeedDataset, PageHeader, Spinner, Stat, fmt, tooltipStyle } from "../components/UI.jsx";

const axis = { stroke: COLORS.axis, tick: { fontSize: 10 } };

function MiniBars({ data, xKey, yKey = "count", color = COLORS.blue, height = 150, horizontal = false }) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <BarChart data={data} layout={horizontal ? "vertical" : "horizontal"} margin={{ top: 4, right: 8, bottom: 0, left: horizontal ? 30 : -18 }}>
          <CartesianGrid stroke={COLORS.grid} vertical={false} />
          {horizontal ? (
            <>
              <XAxis type="number" {...axis} />
              <YAxis type="category" dataKey={xKey} width={110} {...axis} />
            </>
          ) : (
            <>
              <XAxis dataKey={xKey} {...axis} interval="preserveStartEnd" />
              <YAxis {...axis} />
            </>
          )}
          <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.05)" }} formatter={(v) => fmt(v)} labelFormatter={(l, p) => p?.[0]?.payload?.label || l} />
          <Bar dataKey={yKey} fill={color} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Heatmap({ columns, matrix }) {
  const cell = (r) => (r >= 0 ? `rgba(16,185,129,${Math.abs(r)})` : `rgba(239,68,68,${Math.abs(r)})`);
  return (
    <div className="overflow-auto">
      <table className="border-separate border-spacing-0.5 text-[10px]">
        <thead>
          <tr><th />{columns.map((c) => <th key={c} className="max-w-[64px] truncate px-1 pb-1 text-mute" title={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={columns[i]}>
              <th className="max-w-[90px] truncate pr-2 text-right font-normal text-mute" title={columns[i]}>{columns[i]}</th>
              {row.map((v, j) => (
                <td key={j} title={`${columns[i]} vs ${columns[j]}: ${v}`} style={{ background: cell(v ?? 0), color: "#fff" }} className="h-8 w-12 rounded-sm text-center">
                  {v === null ? "" : v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function EdaPage() {
  const { id, meta, columns, status, results, setResult } = useDataset();
  const [source, setSource] = useState("auto");
  const [target, setTarget] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const eda = results.eda;

  useEffect(() => { if (meta?.target) setTarget((t) => t || meta.target); }, [meta]);

  async function load(src = source, tgt = target) {
    setBusy(true); setError("");
    try { setResult("eda", await api.eda(id, { source: src, target: tgt })); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  useEffect(() => { if (id) load(); }, [id, status.cleaned]); // eslint-disable-line

  if (!id) return <NeedDataset />;
  const ov = eda?.overview;

  return (
    <div>
      <PageHeader step={3} title="Exploratory Data Analysis (EDA)" desc="Statistical summaries, distributions, correlations and the relationship between each column and your target." />
      <ErrorBox error={error} />
      <Card>
        <div className="grid items-end gap-4 sm:grid-cols-3">
          <Field label="Data source">
            <select className="field" value={source} onChange={(e) => { setSource(e.target.value); load(e.target.value, target); }}>
              <option value="auto">Cleaned if available, otherwise raw</option>
              <option value="raw">Raw upload</option>
              <option value="cleaned" disabled={!status.cleaned}>Cleaned data</option>
            </select>
          </Field>
          <Field label="Target column (optional)">
            <select className="field" value={target} onChange={(e) => { setTarget(e.target.value); load(source, e.target.value); }}>
              <option value="">None</option>
              {columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </Field>
          <div>{busy ? <Spinner label="Analysing..." /> : <button className="btn-ghost" onClick={() => load()}>Refresh analysis</button>}</div>
        </div>
      </Card>

      {eda && (
        <>
          <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
            <Stat label="Rows" value={ov.rows.toLocaleString()} />
            <Stat label="Columns" value={ov.columns} />
            <Stat label="Numeric" value={ov.numeric} />
            <Stat label="Categorical" value={ov.categorical} />
            <Stat label="Missing cells" value={ov.missing_cells.toLocaleString()} />
            <Stat label="Duplicates" value={ov.duplicates.toLocaleString()} sub={`${eda.source} data`} />
          </div>

          <Card tone="forest" title="Automatic observations" className="mt-6">
            <ul className="space-y-2">
              {eda.insights.map((i, k) => (
                <li key={k} className="flex items-start gap-2 text-sm text-slate-200"><Badge kind={i.level}>{i.level}</Badge><span>{i.text}</span></li>
              ))}
            </ul>
          </Card>

          <Card title="Column statistics" className="mt-6">
            <div className="max-h-96 overflow-auto rounded-lg border border-slate-800">
              <table className="min-w-full text-left text-xs">
                <thead className="sticky top-0 bg-slate-950 text-mute">
                  <tr>{["Column", "Type", "Missing %", "Unique", "Mean", "Std", "Min", "Median", "Max", "Skew", "Outliers"].map((h) => <th key={h} className="whitespace-nowrap px-3 py-2">{h}</th>)}</tr>
                </thead>
                <tbody>
                  {eda.columns.map((c) => (
                    <tr key={c.name} className="border-t border-slate-800/80">
                      <td className="px-3 py-1.5 font-medium text-white">{c.name}</td>
                      <td className="px-3 py-1.5 text-mute">{c.kind}</td>
                      <td className={`px-3 py-1.5 ${c.missing_pct > 0 ? "text-amber-300" : "text-slate-400"}`}>{c.missing_pct}</td>
                      <td className="px-3 py-1.5">{c.unique}</td>
                      {c.kind === "numeric" ? (
                        <>
                          <td className="px-3 py-1.5">{fmt(c.mean)}</td><td className="px-3 py-1.5">{fmt(c.std)}</td><td className="px-3 py-1.5">{fmt(c.min)}</td>
                          <td className="px-3 py-1.5">{fmt(c.median)}</td><td className="px-3 py-1.5">{fmt(c.max)}</td><td className="px-3 py-1.5">{fmt(c.skew, 2)}</td><td className="px-3 py-1.5">{c.outliers_iqr}</td>
                        </>
                      ) : (
                        <td colSpan={7} className="px-3 py-1.5 text-mute">{c.top_values ? c.top_values.slice(0, 3).map((t) => `${t.value} (${t.count})`).join(", ") : `${c.min} → ${c.max}`}</td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {eda.missing.length > 0 && (
            <Card title="Missing values by column" className="mt-6">
              <MiniBars data={eda.missing} xKey="column" yKey="missing" color={COLORS.amber} height={Math.max(120, eda.missing.length * 28)} horizontal />
            </Card>
          )}

          {Object.keys(eda.histograms).length > 0 && (
            <Card tone="forest" title="Distributions of numeric columns" className="mt-6">
              <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
                {Object.entries(eda.histograms).map(([name, h]) => (
                  <div key={name}><p className="mb-1 text-xs font-medium text-slate-200">{name}</p><MiniBars data={h} xKey="range" /></div>
                ))}
              </div>
            </Card>
          )}

          {Object.keys(eda.categorical).length > 0 && (
            <Card title="Categorical columns" className="mt-6">
              <div className="grid gap-5 md:grid-cols-2">
                {Object.entries(eda.categorical).map(([name, rows]) => (
                  <div key={name}><p className="mb-1 text-xs font-medium text-slate-200">{name}</p><MiniBars data={rows} xKey="value" color={COLORS.violet} height={Math.max(110, rows.length * 24)} horizontal /></div>
                ))}
              </div>
            </Card>
          )}

          {eda.correlation.matrix.length > 0 && (
            <Card tone="forest" title="Correlation matrix" subtitle="Green = positive, red = negative (Pearson)" className="mt-6">
              <Heatmap columns={eda.correlation.columns} matrix={eda.correlation.matrix} />
              {eda.top_correlations.length > 0 && (
                <div className="mt-4"><DataTable rows={eda.top_correlations} columns={["a", "b", "r"]} maxHeight="max-h-48" /></div>
              )}
            </Card>
          )}

          {eda.scatter.length > 0 && (
            <Card title="Most correlated pairs" className="mt-6">
              <div className="grid gap-5 md:grid-cols-3">
                {eda.scatter.map((s) => (
                  <div key={s.x + s.y}>
                    <p className="mb-1 text-xs text-slate-200">{s.x} vs {s.y} (r = {s.r})</p>
                    <div className="h-44">
                      <ResponsiveContainer>
                        <ScatterChart margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
                          <CartesianGrid stroke={COLORS.grid} />
                          <XAxis type="number" dataKey="x" name={s.x} {...axis} /><YAxis type="number" dataKey="y" name={s.y} {...axis} />
                          <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: "3 3" }} formatter={(v) => fmt(v)} />
                          <Scatter data={s.points} fill={COLORS.green} fillOpacity={0.5} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {eda.target && (
            <Card tone="forest" title={`Target analysis: ${eda.target.name}`} subtitle={`${eda.target.task} problem`} className="mt-6">
              <div className="grid gap-6 lg:grid-cols-2">
                <div>
                  <p className="mb-1 text-xs text-mute">Target distribution</p>
                  <MiniBars data={eda.target.distribution} xKey={eda.target.task === "classification" ? "value" : "range"} color={COLORS.green} height={200} />
                </div>
                <div>
                  <p className="mb-1 text-xs text-mute">Mutual information with the target (higher = more informative column)</p>
                  <MiniBars data={eda.target.mutual_information} xKey="feature" yKey="score" color={COLORS.blue} height={Math.max(200, eda.target.mutual_information.length * 24)} horizontal />
                </div>
              </div>
            </Card>
          )}

          <Card title="Data preview" subtitle="First 10 rows" className="mt-6"><DataTable rows={eda.preview} /></Card>
        </>
      )}
    </div>
  );
}
