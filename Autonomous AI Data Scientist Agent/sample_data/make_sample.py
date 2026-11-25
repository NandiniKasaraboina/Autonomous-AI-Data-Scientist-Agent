"""Creates a messy demo dataset (customer churn) with missing values, outliers, duplicates and typos."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
n = 1500
age = rng.integers(18, 80, n)
tenure = rng.integers(1, 72, n)
monthly = np.round(20 + tenure * 0.3 + rng.normal(0, 10, n) + (age < 30) * 15, 2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])
gender = rng.choice(["Male", "Female"], n)
support_calls = rng.poisson(1.5, n)
score = 0.04 * (monthly - 50) - 0.05 * tenure + 0.35 * support_calls + (contract == "Month-to-month") * 1.2 + rng.normal(0, 1, n)
churn = np.where(score > np.percentile(score, 72), "Yes", "No")
df = pd.DataFrame({"customer_id": [f"C{100000 + i}" for i in range(n)], "age": age, "gender": gender,
                   "tenure_months": tenure, "monthly_charges": monthly, "contract": contract,
                   "support_calls": support_calls, "signup_date": pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 1800, n), unit="D"),
                   "churn": churn})
df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")
for c, frac in [("age", .06), ("monthly_charges", .08), ("contract", .05), ("tenure_months", .04)]:
    df.loc[rng.choice(n, int(n * frac), replace=False), c] = np.nan
df.loc[rng.choice(n, 25, replace=False), "monthly_charges"] = rng.choice([950.0, 1200.0, 3000.0], 25)
df.loc[rng.choice(n, 15, replace=False), "age"] = rng.choice([180, 250, 999], 15)
df.loc[rng.choice(n, 40, replace=False), "gender"] = rng.choice(["male", "FEMALE", " Male "], 40)
df["support_calls"] = df["support_calls"].astype(object)
df.loc[rng.choice(n, 20, replace=False), "support_calls"] = "N/A"
df = pd.concat([df, df.sample(20, random_state=1)], ignore_index=True)
df.to_csv("sample_churn.csv", index=False)
print(df.shape)
