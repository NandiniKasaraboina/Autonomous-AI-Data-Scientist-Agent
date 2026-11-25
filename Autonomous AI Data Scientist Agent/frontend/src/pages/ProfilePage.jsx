import { useState } from "react";
import { useAuth } from "../context/AuthContext.jsx";
import { Card, ErrorBox, PageHeader } from "../components/UI.jsx";

export default function ProfilePage() {
  const { user, changePassword } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  async function submitPasswordChange(event) {
    event.preventDefault();
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const currentPassword = String(form.get("currentPassword"));
    const newPassword = String(form.get("newPassword"));
    if (newPassword !== String(form.get("confirmPassword"))) {
      setError("The new passwords do not match.");
      setSuccess("");
      return;
    }

    setBusy(true);
    setError("");
    setSuccess("");
    try {
      const result = await changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      });
      formElement.reset();
      setSuccess(result.message);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="profile-page">
      <PageHeader title="Profile" desc="Manage your account details and password." />
      <Card title="Account details" subtitle="Your signed-in account">
        <dl className="profile-details">
          <div>
            <dt>Name</dt>
            <dd>{user.name}</dd>
          </div>
          <div>
            <dt>Email</dt>
            <dd>{user.email}</dd>
          </div>
        </dl>
      </Card>

      <Card title="Change password" subtitle="Verify your current password before choosing a new one." className="mt-6">
        <form className="profile-password-form" onSubmit={submitPasswordChange}>
          <label className="profile-password-field">
            Current password
            <input className="field" type="password" name="currentPassword" autoComplete="current-password" required />
          </label>
          <label className="profile-password-field">
            New password
            <input className="field" type="password" name="newPassword" autoComplete="new-password" minLength={12} maxLength={128} required />
            <span className="profile-password-hint">Use at least 12 characters.</span>
          </label>
          <label className="profile-password-field">
            Confirm new password
            <input className="field" type="password" name="confirmPassword" autoComplete="new-password" minLength={12} maxLength={128} required />
          </label>
          {error && <ErrorBox error={error} />}
          {success && <p className="profile-success" role="status">{success}</p>}
          <button type="submit" className="btn-primary" disabled={busy}>
            {busy ? "Updating password..." : "Update password"}
          </button>
        </form>
      </Card>
    </div>
  );
}
