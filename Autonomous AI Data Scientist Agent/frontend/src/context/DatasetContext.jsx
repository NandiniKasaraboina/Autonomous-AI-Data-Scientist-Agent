import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api } from "../api.js";
import { useAuth } from "./AuthContext.jsx";

const Ctx = createContext(null);
const KEY = "mlis.datasetId";

/**
 * Holds the uploaded dataset and the results of every stage so pages can be visited in any order
 * and still show what was already computed.
 */
export function DatasetProvider({ children }) {
  const { user, loading: authLoading } = useAuth();
  const [info, setInfo] = useState(null); // {meta, status, columns, preview, ...}
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState({}); // {prep, eda, ml, insights, report}

  const refresh = useCallback(async (id) => {
    const data = await api.dataset(id);
    setInfo(data);
    return data;
  }, []);

  // restore the last dataset after a page reload
  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      setInfo(null);
      setResults({});
      setLoading(false);
      return;
    }
    setLoading(true);
    const id = localStorage.getItem(`${KEY}:${user.id}`);
    if (!id) {
      setInfo(null);
      setResults({});
      setLoading(false);
      return;
    }
    refresh(id)
      .catch(() => localStorage.removeItem(`${KEY}:${user.id}`))
      .finally(() => setLoading(false));
  }, [authLoading, refresh, user]);

  const setDataset = useCallback((data) => {
    if (!user) throw new Error("Sign in before saving a dataset.");
    localStorage.setItem(`${KEY}:${user.id}`, data.meta.id);
    setInfo(data);
    setResults({});
  }, [user]);

  const clearDataset = useCallback(() => {
    if (user) localStorage.removeItem(`${KEY}:${user.id}`);
    setInfo(null);
    setResults({});
  }, [user]);

  const setResult = useCallback((key, value) => setResults((r) => ({ ...r, [key]: value })), []);
  const invalidateFrom = useCallback((keys) => setResults((r) => {
    const n = { ...r };
    keys.forEach((k) => delete n[k]);
    return n;
  }), []);

  const value = {
    info, loading, id: info?.meta?.id ?? null, meta: info?.meta ?? null, status: info?.status ?? {},
    columns: info?.columns ?? [], modelColumns: info?.model_columns ?? info?.columns ?? [],
    results, setResult, invalidateFrom, setDataset, clearDataset, refresh,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useDataset = () => useContext(Ctx);
