import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { useDataset } from "../context/DatasetContext.jsx";
import { Badge, Card, DataTable, ErrorBox, PageHeader, Spinner, Stat } from "../components/UI.jsx";

export default function UploadPage() {
  const { info, setDataset } = useDataset();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const input = useRef(null);
  const sampleStarted = useRef(false);
  const location = useLocation();
  const navigate = useNavigate();

  async function send(file) {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      setDataset(await api.upload(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function useSample() {
    try {
      const res = await fetch("/sample_churn.csv");
      if (!res.ok) throw new Error(`Sample dataset request failed (${res.status}).`);
      const blob = await res.blob();
      await send(new File([blob], "sample_churn.csv", { type: "text/csv" }));
    } catch {
      setError("Could not load the sample file.");
    }
  }

  useEffect(() => {
    if (!location.state?.loadSample || sampleStarted.current) return;
    sampleStarted.current = true;
    navigate(location.pathname, { replace: true, state: null });
    useSample();
  }, [location.pathname, location.state, navigate]);

  return (
    <div>
      <PageHeader step={1} title="Dataset Upload" desc="Upload a CSV or Excel dataset. The file is read by the Python backend and profiled instantly." />
      <ErrorBox error={error} />

      <Card>
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); send(e.dataTransfer.files?.[0]); }}
          className={`rounded-xl border-2 border-dashed p-10 text-center transition ${drag ? "border-emerald-500 bg-emerald-500/5" : "border-slate-700"}`}>
          <p className="text-white">Drag and drop your file here</p>
          <p className="mt-1 text-xs text-mute">.csv, .tsv, .xlsx or .xls · up to 50 MB (first sheet is used for Excel)</p>
          <div className="mt-4 flex flex-wrap justify-center gap-3">
            <button className="btn-primary" disabled={busy} onClick={() => input.current?.click()}>Choose file</button>
            <button className="btn-ghost" disabled={busy} onClick={useSample}>Try the sample dataset</button>
          </div>
          <input ref={input} type="file" accept=".csv,.tsv,.txt,.xlsx,.xlsm,.xls" className="hidden"
            onChange={(e) => { send(e.target.files?.[0]); e.target.value = ""; }} />
          {busy && <div className="mt-4"><Spinner label="Uploading and profiling..." /></div>}
        </div>
      </Card>

      {info && (
        <>
          <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Rows" value={info.meta.rows.toLocaleString()} />
            <Stat label="Columns" value={info.meta.cols} />
            <Stat label="Missing cells" value={info.missing_cells.toLocaleString()} accent={info.missing_cells ? "text-amber-300" : "text-emerald-300"} />
            <Stat label="Duplicate rows" value={info.duplicates.toLocaleString()} accent={info.duplicates ? "text-amber-300" : "text-emerald-300"} />
          </div>

          <Card tone="forest" title="Columns" subtitle={`Detected in ${info.meta.filename}`} className="mt-6">
            <div className="max-h-96 overflow-auto rounded-lg border border-slate-800">
              <table className="min-w-full text-left text-xs">
                <thead className="sticky top-0 bg-slate-950 text-mute">
                  <tr><th className="px-3 py-2">Column</th><th className="px-3 py-2">Type</th><th className="px-3 py-2">Missing</th><th className="px-3 py-2">Unique</th><th className="px-3 py-2">Examples</th></tr>
                </thead>
                <tbody>
                  {info.columns.map((c) => (
                    <tr key={c.name} className="border-t border-slate-800/80">
                      <td className="px-3 py-1.5 font-medium text-white">{c.name}</td>
                      <td className="px-3 py-1.5"><Badge kind={c.kind === "numeric" ? "info" : "Rule-based"}>{c.kind}</Badge></td>
                      <td className={`px-3 py-1.5 ${c.missing ? "text-amber-300" : "text-slate-400"}`}>{c.missing}</td>
                      <td className="px-3 py-1.5 text-slate-300">{c.unique}</td>
                      <td className="px-3 py-1.5 text-mute">{c.sample.join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Preview" subtitle="First 12 rows of the raw file" className="mt-6">
            <DataTable rows={info.preview} />
          </Card>

          <div className="mt-6 text-center">
            <Link to="/preprocessing" className="btn-primary">Continue to Data Preprocessing →</Link>
          </div>
        </>
      )}
    </div>
  );
}
