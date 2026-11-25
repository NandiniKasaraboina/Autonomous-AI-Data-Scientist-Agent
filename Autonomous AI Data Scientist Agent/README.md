# ML Insight Studio

Upload a CSV/Excel file and walk it through six stages: **Upload → Preprocessing → EDA → Machine Learning → AI Insights → PDF Report**.

- `frontend/` React 18 + Vite + Tailwind CSS + Recharts (6 pages, dark UI from your reference)
- `backend/`  Python FastAPI + pandas + scikit-learn + PyTorch + ReportLab
- `sample_data/sample_churn.csv` messy demo file (missing values, outliers, duplicates, typos)

## Run it (two terminals)

Requirements: Python 3.10+ and Node.js 18+.

**Windows:** double-click `run_backend.bat`, then `run_frontend.bat`
**Mac/Linux:** `./run_backend.sh` and `./run_frontend.sh`

Or manually:
```
cd backend  && python -m venv .venv && .venv\Scripts\activate   # (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

cd frontend && npm install && npm run dev
```
Open **http://localhost:5173** (API docs: http://localhost:8000/docs). On the Upload page, click "Try the sample dataset" and use target `churn`.

Torch is large; CPU build: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
If torch is not installed the autoencoder automatically falls back to a scikit-learn neural network (the app still runs).

## Accounts and security

Create an account on first use; the workspace and dataset APIs require a signed-in session. Passwords must be at least 12 characters and are stored as PBKDF2-SHA256 hashes. Change a password from the Profile page; the current password is required and other active sessions are revoked. Random session tokens are stored server-side only as hashes and sent to the browser in HTTP-only, SameSite=Lax cookies. Sessions expire after 14 days. Failed sign-ins and password-change attempts are rate-limited, and dataset access is restricted to its owner.

Account and session records are stored in `backend/storage/auth.sqlite3`. For deployment, serve the app over HTTPS, set `MLAPP_ENV=production` (which enables Secure cookies by default), and set `MLAPP_ALLOWED_ORIGINS` to the exact trusted origins, comma-separated. `MLAPP_AUTH_DB` and `MLAPP_COOKIE_SECURE` can override the database path and cookie setting.

Email verification and password reset are not configured. Existing pre-account datasets can be adopted by the first account in that browser; other unowned datasets must be uploaded again.

## Where ML and deep learning are used
| Stage | Technique |
|---|---|
| Missing values | ExtraTrees **IterativeImputer**, **KNN**, **mean**, **median**, or **most-frequent** imputation |
| Outlier detection | **Isolation Forest** + **denoising autoencoder** (PyTorch, trained on trimmed data) + IQR range check, combined by voting |
| Outlier repair | Autoencoder reconstruction, limited to a plausible range (or cap / remove / detect only) |
| Categorical features | Choose one global **one-hot**, **label**, or **ordinal** encoding in preprocessing for all features; cleaned CSV retains readable category labels |
| Classification target | **LabelEncoder** before Random Forest training |
| Rule-based | whitespace, hidden NA tokens, exact duplicate rows, type inference, dates, inconsistent spelling, ID/constant columns |
| Modelling | Random Forest (classification or regression auto-detected), CV, optional RandomizedSearch tuning |
| Explainability | impurity + permutation importance, partial dependence, per-row occlusion, Q&A over your results |

The upload and preprocessing screens report missing cells (including per-column counts) and exact duplicate rows before cleaning. Missing-value methods appear when needed; one categorical encoding is selected for all features during training, and the cleaned export keeps readable labels. The model page previews the inferred classification or regression task for the selected target, using the same target-value detection as training.

## Structure
```
backend/app/main.py            API routes
backend/app/services/          loader, preprocessing, dl (autoencoder), eda, ml, insights, report
backend/storage/               uploaded data, models, reports (created at runtime)
frontend/src/pages/            Home, Upload, Preprocess, Eda, Ml, Insights, Report
```
Production: `npm run build` in `frontend/`; the backend then serves the built app at http://localhost:8000.

## Limits
Excel: first sheet only. Max 50 MB / 200k rows. Data is kept on disk locally in `backend/storage`.
