# Running HemaScan Locally (VS Code)

This guide walks you through running the anemia-screening web app on your own machine. The app uses **SQLite** (a single file, zero setup) so you do **not** need to install or run a database server.

## 1. Prerequisites

| Tool | Why | Install |
|---|---|---|
| Python 3.11 | Backend | https://www.python.org/downloads/ |
| Node.js 18+ | Frontend | https://nodejs.org/ |
| Yarn | Frontend package manager (NOT npm) | `npm install -g yarn` |
| VS Code | IDE | https://code.visualstudio.com/ |

> ✅ No database server to install. SQLite is bundled with Python.

## 2. Open the project

```bash
cd hemascan
code .
```

## 3. Environment files

Create `backend/.env`:
```
JWT_SECRET="change-me-to-any-64-char-hex-string"
ADMIN_EMAIL="admin@anemiacheck.app"
ADMIN_PASSWORD="Admin@123"
EMERGENT_LLM_KEY="sk-emergent-1494aC21c743eC4E47"
```

Optional — override where the SQLite file lives (defaults to `backend/data/hemascan.db`):
```
SQLITE_PATH="./data/hemascan.db"
```

Create `frontend/.env`:
```
REACT_APP_BACKEND_URL=http://localhost:8001
WDS_SOCKET_PORT=0
```

## 4. Backend (first terminal)

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-local.txt        # TensorFlow for the .h5 models
pip install emergentintegrations --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/

uvicorn server:app --reload --host 0.0.0.0 --port 8001
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete.
... [INFO] Admin user seeded
```

On first run, a file `backend/data/hemascan.db` is created automatically — this is your entire database.

## 5. Frontend (second terminal)

```bash
cd frontend
yarn install
yarn start
```

Opens automatically at http://localhost:3000.

## 6. First login

- Email: `admin@anemiacheck.app`
- Password: `Admin@123`

Visit `/admin/models` to confirm both `.h5` files are detected (they ship in `backend/ml/models/`).

## 7. Useful VS Code extensions

- Python (Microsoft)
- Pylance
- ES7+ React/Redux snippets
- Tailwind CSS IntelliSense
- Thunder Client (API testing)
- SQLite Viewer (alexcvzz.vscode-sqlite) — browse `backend/data/hemascan.db` right inside VS Code

## 8. Common issues

| Error | Fix |
|---|---|
| `ModuleNotFoundError: aiosqlite` | Re-run `pip install -r requirements.txt` |
| `ModuleNotFoundError: emergentintegrations` | Re-run the special-index pip install |
| `CORS error` | Verify `frontend/.env` `REACT_APP_BACKEND_URL` is `http://localhost:8001` (no trailing slash) |
| `tensorflow` install fails on Apple Silicon | `pip install tensorflow-macos` instead |
| Port 8001 in use | `uvicorn server:app --port 8002` and update `frontend/.env` |
| Need a fresh DB | Stop backend, delete `backend/data/hemascan.db`, restart |

## 9. Where is the data?

A single SQLite file at `backend/data/hemascan.db` with two tables:
- `users` (bcrypt-hashed passwords)
- `screenings` (patient records + base64 images + model outputs + Grad-CAM)

Browse it inside VS Code with the **SQLite Viewer** extension, or from the terminal:
```bash
# install once
pip install sqlite-utils
# inspect
sqlite-utils tables backend/data/hemascan.db --counts
sqlite-utils rows  backend/data/hemascan.db users
```

Or with the built-in CLI:
```bash
python -m sqlite3 backend/data/hemascan.db ".tables"
python -m sqlite3 backend/data/hemascan.db "SELECT id, email FROM users;"
```

## 10. Production deploy

Use the Emergent dashboard: **Save to GitHub** to version it, then **Deploy** for a public URL. Costs ~50 credits/month for managed hosting.

## 11. Smoke test (optional but nice for viva)

With both servers running, paste this in a third terminal:
```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@anemiacheck.app","password":"Admin@123"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['token'])")

# Confirm /me works
curl -s http://localhost:8001/api/auth/me -H "Authorization: Bearer $TOKEN"

# Confirm models are loaded
curl -s http://localhost:8001/api/admin/models -H "Authorization: Bearer $TOKEN"
```
You should see your admin profile + both `eye_loaded` and `nail_loaded` reported as `true`.
