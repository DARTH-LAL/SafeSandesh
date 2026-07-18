# Deployment Guide

This project should be deployed as two connected views:

- Consumer app online: Home, Detector, and Consumer Dashboard.
- Technical app private: Technical Dashboard and AI Studio.

Both apps can read the same Supabase database, so online scans are still visible in the local technical app.

## 1. Create Supabase Tables

1. Create a free Supabase project.
2. Open the Supabase SQL editor.
3. Paste and run:

```sql
-- scripts/supabase_schema.sql
```

Use the contents of `scripts/supabase_schema.sql`.

The schema stores full scan messages, timestamps, verdicts, language, scam type, model outputs, and feedback.

If the Supabase project was created before readable scan date/time columns were added, also run:

```sql
-- scripts/supabase_add_scan_datetime_columns.sql
```

## 2. Test Supabase Locally

From the project folder:

```bash
cd "/Users/ajneya/Desktop/ FYP Main/scam-webapp"
export SCAN_DB_BACKEND="supabase"
export SUPABASE_URL="https://YOUR-PROJECT.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="YOUR-SERVICE-ROLE-KEY"
./.venv/bin/streamlit run apps/consumer_app.py --server.port 8502
```

Open:

```text
http://localhost:8502
```

Run a scan. If it appears in the dashboard, Supabase storage is working.

## 3. Migrate Local Scan History

First count local rows:

```bash
cd "/Users/ajneya/Desktop/ FYP Main/scam-webapp"
./.venv/bin/python scripts/migrate_sqlite_to_supabase.py --dry-run
```

Then migrate:

```bash
cd "/Users/ajneya/Desktop/ FYP Main/scam-webapp"
export SUPABASE_URL="https://YOUR-PROJECT.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="YOUR-SERVICE-ROLE-KEY"
./.venv/bin/python scripts/migrate_sqlite_to_supabase.py
```

This copies local SQLite `data/app.db` scans and feedback into Supabase.

## 4. Deploy Consumer App For Other People

On Streamlit Community Cloud:

- Repository: this project repository.
- Main file path: `apps/consumer_app.py`.
- Python version: use the default supported version unless Streamlit asks.
- Secrets:

```toml
SCAN_DB_BACKEND = "supabase"
SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "YOUR-SERVICE-ROLE-KEY"
```

After deployment, Streamlit gives a public URL. Other people can open that URL from their own laptops.

## 5. Deploy Technical App With Password Protection

Create a second Streamlit app from the same GitHub repository:

- Repository: this project repository.
- Main file path: `apps/technical_app.py`.
- App URL: use a separate name, for example `safesandesh-technical`.
- Sharing: keep it private if your Streamlit plan allows it. If not, the app still has its own password-only gate.

Add these secrets to the technical Streamlit app:

```toml
SCAN_DB_BACKEND = "supabase"
SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "YOUR-SERVICE-ROLE-KEY"
TECHNICAL_APP_PASSWORD = "CHOOSE-A-STRONG-PASSWORD"
```

For a cleaner submission, you can store a SHA-256 hash instead of the raw password:

```bash
python - <<'PY'
import hashlib
password = input("Technical app password: ")
print(hashlib.sha256(password.encode("utf-8")).hexdigest())
PY
```

Then use this secret instead:

```toml
TECHNICAL_APP_PASSWORD_HASH = "PASTE-THE-SHA256-HASH-HERE"
```

This is not a user login system. It is only one shared technical access password. The password is never committed to GitHub. It must be added only in Streamlit secrets or local environment variables.

## 6. Run Technical App Locally

Run this locally when you want the private analyst view connected to the same Supabase scans:

```bash
cd "/Users/ajneya/Desktop/ FYP Main/scam-webapp"
export SCAN_DB_BACKEND="supabase"
export SUPABASE_URL="https://YOUR-PROJECT.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="YOUR-SERVICE-ROLE-KEY"
export TECHNICAL_APP_PASSWORD="CHOOSE-A-STRONG-PASSWORD"
./.venv/bin/streamlit run apps/technical_app.py --server.port 8503
```

Open:

```text
http://localhost:8503
```

## 7. FYP Submission Backup

For submission, export the Supabase tables as CSV files from Supabase Table Editor or SQL editor. Keep these exports with the project:

- `scans.csv`
- `feedback.csv`

That gives you a portable backup of the deployed database.

## Privacy Note

Because full messages are stored, the consumer app should tell users that submitted text may be saved for scan history and project evaluation. Do not expose Supabase anon read policies for these tables.
