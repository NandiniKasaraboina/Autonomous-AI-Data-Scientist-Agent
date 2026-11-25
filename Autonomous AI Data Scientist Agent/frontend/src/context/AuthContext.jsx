import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api } from "../api.js";

const AuthCtx = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const result = await api.auth.me();
      setUser(result.user);
      setError("");
      return result.user;
    } catch (err) {
      if (err.status === 401) {
        setUser(null);
        setError("");
        return null;
      }
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const login = useCallback(async (credentials) => {
    const result = await api.auth.login(credentials);
    setUser(result.user);
    setError("");
    return result.user;
  }, []);

  const register = useCallback(async (details) => {
    const legacyDatasetId = localStorage.getItem("mlis.datasetId");
    const result = await api.auth.register({ ...details, legacy_dataset_id: legacyDatasetId });
    if (result.adopted_dataset_id) {
      localStorage.setItem(`mlis.datasetId:${result.user.id}`, result.adopted_dataset_id);
    }
    localStorage.removeItem("mlis.datasetId");
    setUser(result.user);
    setError("");
    return result.user;
  }, []);

  const logout = useCallback(async () => {
    await api.auth.logout();
    setUser(null);
  }, []);

  const changePassword = useCallback((details) => api.auth.changePassword(details), []);

  return (
    <AuthCtx.Provider value={{ user, loading, error, login, register, logout, changePassword, refresh }}>
      {children}
    </AuthCtx.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthCtx);
  if (!context) throw new Error("useAuth must be used within AuthProvider.");
  return context;
}
