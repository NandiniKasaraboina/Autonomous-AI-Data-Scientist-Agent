import { Route, Routes, Navigate, useLocation } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Home from "./pages/Home.jsx";
import UploadPage from "./pages/UploadPage.jsx";
import PreprocessPage from "./pages/PreprocessPage.jsx";
import EdaPage from "./pages/EdaPage.jsx";
import MlPage from "./pages/MlPage.jsx";
import InsightsPage from "./pages/InsightsPage.jsx";
import ReportPage from "./pages/ReportPage.jsx";
import ProfilePage from "./pages/ProfilePage.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import { useAuth } from "./context/AuthContext.jsx";

function SessionLoading() {
  return <main className="auth-loading"><span className="brand-mark">ML</span><p>Checking your secure session…</p></main>;
}

function PublicAuthRoute({ mode }) {
  const { user, loading } = useAuth();
  if (loading) return <SessionLoading />;
  if (user) return <Navigate to="/" replace />;
  return <AuthPage mode={mode} />;
}

function Workspace() {
  const location = useLocation();
  const { user, loading, error, refresh } = useAuth();
  if (loading) return <SessionLoading />;
  if (error) {
    return (
      <main className="auth-loading">
        <p role="alert">{error}</p>
        <button className="btn-primary" onClick={refresh}>Retry connection</button>
      </main>
    );
  }
  if (!user) return <Navigate to="/login" state={{ from: location }} replace />;

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/preprocessing" element={<PreprocessPage />} />
        <Route path="/eda" element={<EdaPage />} />
        <Route path="/machine-learning" element={<MlPage />} />
        <Route path="/insights" element={<InsightsPage />} />
        <Route path="/report" element={<ReportPage />} />
        <Route path="/profile" element={<ProfilePage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<PublicAuthRoute mode="login" />} />
      <Route path="/create-account" element={<PublicAuthRoute mode="create-account" />} />
      <Route path="*" element={<Workspace />} />
    </Routes>
  );
}
