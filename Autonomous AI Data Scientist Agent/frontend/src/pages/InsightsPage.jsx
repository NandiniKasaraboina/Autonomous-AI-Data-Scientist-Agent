import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useDataset } from "../context/DatasetContext.jsx";
import { NeedDataset, Spinner } from "../components/UI.jsx";

const SUGGESTED_QUESTIONS = [
  "What are the most important features?",
  "How accurate is the model?",
  "Is the model overfitting?",
  "What was cleaned in my data?",
  "Which columns are correlated?",
  "How can I improve the model?",
  "Give me a summary",
];

export default function InsightsPage() {
  const { id } = useDataset();
  const [chat, setChat] = useState([]);
  const [q, setQ] = useState("");
  const [asking, setAsking] = useState(false);
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, [chat]);

  if (!id) return <NeedDataset />;

  async function ask(text) {
    const question = (text ?? q).trim();
    if (!question) return;
    setChat((c) => [...c, { role: "user", text: question }]);
    setQ(""); setAsking(true);
    try {
      const r = await api.ask(id, question);
      setChat((c) => [...c, { role: "ai", text: r.answer }]);
    } catch (e) {
      setChat((c) => [...c, { role: "ai", text: `Sorry, that failed: ${e.message}` }]);
    } finally { setAsking(false); }
  }

  return (
    <section className="mx-auto max-w-5xl rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
      <header className="mb-4">
        <h1 className="text-lg font-semibold text-slate-900">Ask about your data</h1>
        <p className="mt-1 text-xs text-slate-500">Answers are computed from your real results, no data leaves your machine.</p>
      </header>
      {chat.length > 0 && (
        <div className="max-h-80 min-h-20 space-y-3 overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-3 sm:p-4" role="log" aria-label="Chat conversation" aria-live="polite">
          {chat.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <p className={`chat-message ${m.role === "user" ? "chat-message-user" : "chat-message-assistant"}`}>{m.text}</p>
            </div>
          ))}
          {asking && <div className="py-1"><Spinner label="Thinking..." /></div>}
          <div ref={endRef} />
        </div>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        {SUGGESTED_QUESTIONS.map((s) => (
          <button key={s} type="button" onClick={() => ask(s)} disabled={asking}
            className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-600 transition hover:border-emerald-600 hover:text-emerald-800 disabled:opacity-50">
            {s}
          </button>
        ))}
      </div>
      <form className="mt-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); ask(); }}>
        <input className="field" aria-label="Ask about your dataset" placeholder="e.g. Which features matter most for churn?" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="btn-primary chat-ask-button shrink-0" type="submit" disabled={asking || !q.trim()}>Ask</button>
      </form>
    </section>
  );
}
