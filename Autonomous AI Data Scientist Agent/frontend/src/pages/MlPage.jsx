import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ReferenceLine } from "recharts";
import { api } from "../api.js";
import { useDataset } from "../context/DatasetContext.jsx";
import { Card, COLORS, DataTable, ErrorBox, Field, NeedDataset, PageHeader, Spinner, Stat, Toggle, fmt, pct, tooltipStyle } from "../components/UI.jsx";

const axis = { stroke: COLORS.axis, tick: { fontSize: 10 } };

function ImportanceChart({ data, color }) {
  return (
    <div style={{ height: Math.max(180, data.length * 26) }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, bottom: 0, left: 40 }}>
          <CartesianGrid stroke={COLORS.grid} horizontal={false} />
          <XAxis type="number" {...axis} /><YAxis type="category" dataKey="feature" width={120} {...axis} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v, 3)} cursor={{ fill: "rgba(255,255,255,0.05)" }} />
          <Bar dataKey="importance" fill={color} radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ConfusionMatrix({ labels, matrix }) {
  const max = Math.max(...matrix.flat(), 1);
  return (
    <div className="overflow-auto">
      <table className="border-separate border-spacing-1 text-xs">
        <thead><tr><th /><th colSpan={labels.length} className="pb-1 font-medium text-mute">Predicted</th></tr>
          <tr><th />{labels.map((l) => <th key={l} className="px-2 text-slate-300">{l}</th>)}</tr></thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="pr-2 text-right font-normal text-slate-300">{labels[i]}</th>
              {row.map((v, j) => (
                <td key={j} style={{ background: `rgba(${i === j ? "16,185,129" : "239,68,68"},${0.12 + 0.8 * (v / max)})` }}
                  className="h-14 w-20 rounded-md text-center text-base font-bold text-white">{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-[11px] text-mute">Rows are the actual class. Green diagonal = correct predictions.</p>
    </div>
  );
}

export default function MlPage() {
  const { id, meta, columns, modelColumns, status, results, setResult, invalidateFrom, refresh } = useDataset();
  const [cfg, setCfg] = useState({ target: "", test_size: 0.2, n_estimators: 300, tune: false, model: "random_forest" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({});
  const [pred, setPred] = useState(null);
  const [predBusy, setPredBusy] = useState(false);
  const ml = results.ml;

  useEffect(() => {
    if (id && status.trained && !results.ml) api.trainResult(id).then((r) => setResult("ml", r)).catch(() => {});
  }, [id, status.trained]); // eslint-disable-line
  useEffect(() => {
    const availableTargets = modelColumns.length ? modelColumns : columns;
    const existingTarget = availableTargets.find((column) => column.name === cfg.target);
    const savedTarget = availableTargets.find((column) => column.name === meta?.target);
    const t = existingTarget?.name || savedTarget?.name || (availableTargets.length ? availableTargets[availableTargets.length - 1].name : "");
    setCfg((c) => {
      if (c.target === t) return c;
      const selected = availableTargets.find((column) => column.name === t);
      const model = selected?.prediction_task === "regression" ? "linear_regression" : "random_forest";
      return { ...c, target: t, model };
    });
  }, [meta, columns, modelColumns, cfg.target]);
  useEffect(() => {
    if (ml) setForm(Object.fromEntries(ml.schema.map((f) => [f.name, f.default])));
  }, [ml]);

  if (!id) return <NeedDataset />;

  async function train() {
    setBusy(true); setError(""); setPred(null);
    try {
      const r = await api.train(id, cfg);
      invalidateFrom(["insights", "report"]);
      setResult("ml", r);
      await refresh(id);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  async function predict() {
    setPredBusy(true); setError("");
    try { setPred(await api.predict(id, form)); } catch (e) { setError(e.message); } finally { setPredBusy(false); }
  }

  const m = ml?.metrics;
  const trainingColumns = modelColumns.length ? modelColumns : columns;
  const selectedTarget = trainingColumns.find((column) => column.name === cfg.target);
  const task = selectedTarget?.prediction_task;
  const classificationModels = [
    ["random_forest", "Random Forest"], ["logistic_regression", "Logistic Regression"],
    ["svm", "Support Vector Machine"], ["knn", "K-Nearest Neighbors"],
    ["gradient_boosting", "Gradient Boosting"], ["decision_tree", "Decision Tree"],
  ];
  const regressionModels = [
    ["random_forest", "Random Forest"], ["linear_regression", "Linear Regression"],
    ["ridge", "Ridge"], ["lasso", "Lasso"], ["svm", "Support Vector Machine"],
    ["knn", "K-Nearest Neighbors"], ["gradient_boosting", "Gradient Boosting"],
    ["decision_tree", "Decision Tree"],
  ];
  const modelOptions = task === "regression" ? regressionModels : task === "classification" ? classificationModels : [];
  return (
    <div>
      <PageHeader step={4} tone="forest" title="Machine Learning" desc="Choose a suitable model, train it on cleaned data, check it on unseen rows and try live predictions." />
      <ErrorBox error={error} />
      {!status.cleaned && <div className="mb-4 rounded-lg border border-amber-900 bg-amber-950/30 p-3 text-sm text-amber-200">The dataset has not been preprocessed. Training works on the raw file, but cleaning first usually gives better models.</div>}

      <Card tone="forest" title="Model settings">
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Target column to predict">
            <select className="field" value={cfg.target} onChange={(e) => {
              const target = e.target.value;
              const selected = trainingColumns.find((column) => column.name === target);
              setCfg({
                ...cfg,
                target,
                model: selected?.prediction_task === "regression" ? "linear_regression" : "random_forest",
              });
            }}>
              {trainingColumns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </Field>
          <Field label={`Test set size: ${Math.round(cfg.test_size * 100)}%`}>
            <input type="range" min="0.1" max="0.4" step="0.05" value={cfg.test_size} onChange={(e) => setCfg({ ...cfg, test_size: +e.target.value })} className="w-full accent-emerald-500" />
          </Field>
          {task && (
            <Field label="Model algorithm">
              <select className="field" value={cfg.model} onChange={(e) => setCfg({ ...cfg, model: e.target.value })}>
                {modelOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </Field>
          )}
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-5">
          <Toggle checked={cfg.tune} onChange={(v) => setCfg({ ...cfg, tune: v })} label="Tune hyper-parameters (slower, randomized search)" />
          <button className="btn-primary" disabled={busy || !cfg.target || !task} onClick={train}>{ml ? "Train again" : "Train model"}</button>
          {busy && <Spinner label={cfg.tune ? "Searching the best settings..." : "Training..."} />}
        </div>
        <div className="mt-4 rounded-lg border border-slate-800 bg-black/30 p-3 text-sm">
          {selectedTarget?.prediction_task ? (
            <>
              <p className="text-slate-200">
                Detected task for <strong>{cfg.target}</strong>: <strong className="capitalize">{selectedTarget.prediction_task}</strong>
              </p>
              {!selectedTarget.prediction_supported && (
                <p className="mt-1 text-xs text-amber-300">
                  {selectedTarget.prediction_reason}
                </p>
              )}
            </>
          ) : selectedTarget ? (
            <p className="text-amber-300">{selectedTarget.prediction_reason || "Choose a numeric or categorical target column."}</p>
          ) : (
            <p className="text-slate-300">Choose a target column to detect whether this file needs classification or regression.</p>
          )}
          <p className="mt-1 text-xs text-mute">Task type is inferred from the uploaded target values and applied automatically when you train.</p>
        </div>
      </Card>

      {ml && (
        <>
          <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Task" value={ml.task} sub={`target: ${ml.target}`} />
            <Stat label="Algorithm" value={ml.model_name || ml.model || "Model"} />
            {ml.task === "classification" ? (
              <>
                <Stat label="Accuracy" value={pct(m.accuracy)} sub={`baseline ${pct(m.baseline_accuracy)}`} accent="text-emerald-300" />
                <Stat label="F1 (weighted)" value={fmt(m.f1, 3)} />
                <Stat label="ROC-AUC" value={m.roc_auc !== undefined ? fmt(m.roc_auc, 3) : "–"} />
                <Stat label="Precision" value={fmt(m.precision, 3)} />
                <Stat label="Recall" value={fmt(m.recall, 3)} />
              </>
            ) : (
              <>
                <Stat label="R²" value={fmt(m.r2, 3)} sub={`baseline ${fmt(m.baseline_r2, 2)}`} accent="text-emerald-300" />
                <Stat label="MAE" value={fmt(m.mae, 4)} />
                <Stat label="RMSE" value={fmt(m.rmse, 4)} />
              </>
            )}
            <Stat label="Rows train / test" value={`${ml.n_train.toLocaleString()} / ${ml.n_test.toLocaleString()}`} />
            {ml.cv && <Stat label={`${ml.cv.folds}-fold CV`} value={fmt(ml.cv.mean, 3)} sub={`± ${fmt(ml.cv.std, 3)}`} />}
            {ml.best_params && <Stat label="Best params" value="tuned" sub={Object.entries(ml.best_params).map(([k, v]) => `${k}=${v}`).join(", ")} />}
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <Card title="Feature importance" subtitle="How much each column contributes to the selected model">
              <ImportanceChart data={ml.feature_importance.slice(0, 12)} color={COLORS.green} />
            </Card>
            <Card title="Permutation importance" subtitle="Drop in score on unseen rows when a column is shuffled (more trustworthy)">
              <ImportanceChart data={ml.permutation_importance.slice(0, 12)} color={COLORS.blue} />
            </Card>
          </div>

          {ml.task === "classification" ? (
            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              <Card tone="forest" title="Confusion matrix"><ConfusionMatrix labels={ml.confusion_matrix.labels} matrix={ml.confusion_matrix.matrix} /></Card>
              {ml.roc_curve ? (
                <Card tone="forest" title="ROC curve" subtitle={`AUC ${fmt(m.roc_auc, 3)}`}>
                  <div className="h-64">
                    <ResponsiveContainer>
                      <LineChart data={ml.roc_curve} margin={{ top: 5, right: 10, bottom: 15, left: -10 }}>
                        <CartesianGrid stroke={COLORS.grid} />
                        <XAxis type="number" dataKey="x" domain={[0, 1]} {...axis} label={{ value: "False positive rate", position: "insideBottom", offset: -8, fill: COLORS.axis, fontSize: 10 }} />
                        <YAxis type="number" domain={[0, 1]} {...axis} />
                        <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v, 3)} />
                        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="4 4" />
                        <Line type="monotone" dataKey="y" stroke={COLORS.green} dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              ) : (
                <Card tone="forest" title="Per-class results"><DataTable rows={ml.per_class} /></Card>
              )}
              {ml.roc_curve && <Card className="lg:col-span-2" title="Per-class results"><DataTable rows={ml.per_class} /></Card>}
            </div>
          ) : (
            <Card tone="forest" title="Actual vs predicted" subtitle="Points close to the diagonal are accurate predictions" className="mt-6">
              <div className="h-72">
                <ResponsiveContainer>
                  <ScatterChart margin={{ top: 5, right: 10, bottom: 15, left: -5 }}>
                    <CartesianGrid stroke={COLORS.grid} />
                    <XAxis type="number" dataKey="x" name="actual" {...axis} label={{ value: "Actual", position: "insideBottom", offset: -8, fill: COLORS.axis, fontSize: 10 }} />
                    <YAxis type="number" dataKey="y" name="predicted" {...axis} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
                    <Scatter data={ml.actual_vs_predicted} fill={COLORS.blue} fillOpacity={0.6} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          <Card title="Predictions on unseen test rows" subtitle="First 25 test rows. Row numbers refer to the cleaned dataset." className="mt-6"
            actions={<a className="btn-ghost" href={api.url.predictions(id)}>Download all test predictions (CSV)</a>}>
            <DataTable rows={ml.predictions_sample} />
          </Card>

          <Card tone="forest" title="Try a prediction" subtitle="Change the values and let the trained model predict" className="mt-6">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {ml.schema.map((f) => (
                <Field key={f.name} label={f.name} hint={f.type === "number" ? `range ${fmt(f.min)} – ${fmt(f.max)}` : undefined}>
                  {f.type === "number" ? (
                    <input type="number" className="field" value={form[f.name] ?? ""} onChange={(e) => setForm({ ...form, [f.name]: e.target.value })} />
                  ) : (
                    <select className="field" value={form[f.name] ?? ""} onChange={(e) => setForm({ ...form, [f.name]: e.target.value })}>
                      {f.options.map((o) => <option key={o}>{o}</option>)}
                    </select>
                  )}
                </Field>
              ))}
            </div>
            <div className="mt-4 flex items-center gap-4">
              <button className="btn-primary" onClick={predict} disabled={predBusy}>Predict</button>
              {predBusy && <Spinner label="Predicting..." />}
            </div>
            {pred && (
              <div className="mt-4 rounded-lg border border-emerald-800 bg-emerald-500/10 p-4">
                <p className="text-xs text-mute">Predicted {ml.target}</p>
                <p className="text-2xl font-bold text-white">{pred.task === "regression" ? fmt(pred.prediction, 5) : pred.prediction}</p>
                {pred.probabilities && (
                  <div className="mt-3 space-y-1.5">
                    {pred.probabilities.map((p) => (
                      <div key={p.class} className="flex items-center gap-3 text-xs">
                        <span className="w-24 truncate text-slate-300">{p.class}</span>
                        <div className="h-2 flex-1 rounded bg-slate-800"><div className="h-2 rounded bg-emerald-500" style={{ width: `${p.probability * 100}%` }} /></div>
                        <span className="w-12 text-right text-slate-300">{pct(p.probability)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
