# Predictive Maintenance System (PDM)

Minimal runnable app with Backend (Node + Express) and Frontend (HTML/CSS/JS). No DB.

## Structure

- backend
- frontend

Backend serves the frontend statically at http://localhost:4000/.

## Setup & Run

1. Open a terminal
2. Run:

```
cd backend
npm install
node server.js
```

3. Open http://localhost:4000/ in your browser.

## Environment

Create `.env` in `backend/` by copying `.env.example` and fill in values.

Required:

- PORT (default 4000)
- JWT_SECRET
- MODEL_URL (the hosted model endpoint)
- MODEL_API_KEY (if required by your model)

## Seeded Admin User

- Email: admin@example.com
- Password: pass1234

## API Endpoints (summary)

- POST /api/auth/login
- POST /api/auth/signup
- GET /api/machines
- GET /api/machines/:id/history
- POST /api/predict
- POST /api/upload (multipart CSV: file)
- GET /api/upload/:id (training job status)
- GET /api/machines/:id/predictions.csv
- GET /health

## CSV Upload Format

Headers required: `machine_id,timestamp,temperature,vibration,pressure,humidity`

Upload triggers a training job and produces next-30-days predictions per machine (stored in-memory).

## Sample curl

- Login

```
curl -s -X POST http://localhost:4000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"pass1234"}'
```

- Upload CSV (replace TOKEN and path)

```
curl -s -X POST http://localhost:4000/api/upload \
  -H "Authorization: Bearer TOKEN" \
  -F file=@data.csv
```

- Predict (new request shape, replace TOKEN)

```
curl -s -X POST http://localhost:4000/api/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{
    "machine_id":"M1",
    "timestamp":"2025-11-17T10:00:00Z",
    "features": {"temperature":82, "vibration":6.8, "pressure":95, "humidity":30}
  }'
```

## Notes

- Backend logs model request/response summaries.
- Model calls use 3s timeout with up to 2 retries. If the model fails, a deterministic fallback is returned with explanation `fallback: local heuristic`.
- All data is kept in memory.
