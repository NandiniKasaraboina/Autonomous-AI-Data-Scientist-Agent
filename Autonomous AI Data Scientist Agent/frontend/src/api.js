const BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(BASE + path, { ...options, credentials: options.credentials || "include" });
  } catch {
    throw new Error("Cannot reach the backend. Make sure it is running on http://localhost:8000.");
  }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      const d = j.detail ?? j;
      msg = Array.isArray(d) ? d.map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ") : typeof d === "string" ? d : JSON.stringify(d);
    } catch {
      /* keep statusText */
    }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return res;
}

const json = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const get = async (path) => (await request(path)).json();
const post = async (path, body) => (await request(path, json(body))).json();

export const api = {
  auth: {
    me: () => get("/api/auth/me"),
    register: (body) => post("/api/auth/register", body),
    login: (body) => post("/api/auth/login", body),
    logout: () => post("/api/auth/logout", {}),
    changePassword: (body) => post("/api/auth/change-password", body),
  },
  health: () => get("/api/health"),
  upload: async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return (await request("/api/upload", { method: "POST", body: fd })).json();
  },
  dataset: (id) => get(`/api/datasets/${id}`),
  preprocess: (id, body) => post(`/api/datasets/${id}/preprocess`, body),
  preprocessResult: (id) => get(`/api/datasets/${id}/preprocess`),
  eda: (id, { source = "auto", target } = {}) =>
    get(`/api/datasets/${id}/eda?source=${source}${target ? `&target=${encodeURIComponent(target)}` : ""}`),
  train: (id, body) => post(`/api/datasets/${id}/train`, body),
  trainResult: (id) => get(`/api/datasets/${id}/train`),
  predict: (id, values) => post(`/api/datasets/${id}/predict`, { values }),
  insights: (id) => get(`/api/datasets/${id}/insights`),
  ask: (id, question) => post(`/api/datasets/${id}/ask`, { question }),
  explainRow: (id, row) => post(`/api/datasets/${id}/explain-row`, { row }),
  generateReport: (id) => post(`/api/datasets/${id}/report/generate`, {}),
  url: {
    cleaned: (id, format = "csv") => `${BASE}/api/datasets/${id}/download/cleaned?format=${format}`,
    predictions: (id) => `${BASE}/api/datasets/${id}/download/predictions`,
    report: (id) => `${BASE}/api/datasets/${id}/report/download`,
  },
};
