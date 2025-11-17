import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import jwt from 'jsonwebtoken';
import fetch from 'node-fetch';
import multer from 'multer';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 4000;
const JWT_SECRET = process.env.JWT_SECRET || 'dev_secret_change_me';
const MODEL_URL = process.env.MODEL_URL || '';
const MODEL_API_KEY = process.env.MODEL_API_KEY || '';

app.use(express.json({ limit: '1mb' }));
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 5 * 1024 * 1024 } });

// Serve frontend statically from ../frontend
const frontendDir = path.join(__dirname, '../frontend');
app.use(express.static(frontendDir));

// In-memory data stores
const users = [
  {
    id: 'U1',
    email: 'admin@example.com',
    password: 'pass1234', // do not use in production; hashing omitted per spec
    role: 'admin',
    name: 'Admin'
  }
];

const machines = [
  {
    id: 'M1',
    name: 'Compressor A',
    last_health: 0.86,
    failure_probability: 0.22,
    most_critical: 'vibration',
    last_checked_at: new Date(Date.now() - 1000 * 60 * 12).toISOString()
  },
  {
    id: 'M2',
    name: 'Pump B',
    last_health: 0.58,
    failure_probability: 0.41,
    most_critical: 'temperature',
    last_checked_at: new Date(Date.now() - 1000 * 60 * 30).toISOString()
  },
  {
    id: 'M3',
    name: 'Turbine C',
    last_health: 0.44,
    failure_probability: 0.72,
    most_critical: 'pressure',
    last_checked_at: new Date(Date.now() - 1000 * 60 * 60).toISOString()
  }
];

// recent predictions by machine id
const recentPredictions = {
  M1: [],
  M2: [],
  M3: []
};

// future predictions storage (next 30 days per machine)
const futurePredictions = {
  M1: [],
  M2: [],
  M3: []
};

// parameter history captured from CSV uploads
const paramHistory = {
  M1: [],
  M2: [],
  M3: []
};

// training jobs in memory
const trainingJobs = new Map(); // id -> { id, status, progress, startedAt, completedAt, error }
let trainingCounter = 1;

function authMiddleware(req, res, next) {
  const auth = req.headers.authorization || '';
  const token = auth.startsWith('Bearer ') ? auth.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Unauthorized' });
  try {
    const payload = jwt.verify(token, JWT_SECRET);
    req.user = payload;
    next();
  } catch (e) {
    return res.status(401).json({ error: 'Invalid token' });
  }
}

function yyyymmdd(d) {
  return d.toISOString().slice(0, 10);
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

async function callHostedModel({ machine_id, params }) {
  // Model mapping comments:
  // - Input: { machine_id: string, params: { vibration?, temperature?, pressure?, ... } }
  // - Expected Model Response: map it to the required API shape below.
  // - This implementation forwards to MODEL_URL with API key and enforces a 3s timeout with up to 2 retries.
  if (!MODEL_URL) throw new Error('MODEL_URL not configured');

  let attempt = 0;
  const maxRetries = 2; // up to 2 retries
  const timeoutMs = 3000;
  let lastErr = null;

  while (attempt <= maxRetries) {
    attempt++;
    const controller = new AbortController();
    const to = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(MODEL_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': MODEL_API_KEY ? `Bearer ${MODEL_API_KEY}` : undefined
        },
        body: JSON.stringify({ machine_id, params }),
        signal: controller.signal
      });
      clearTimeout(to);
      if (!res.ok) throw new Error(`Model HTTP ${res.status}`);
      const data = await res.json();

      // Mapping: Adapt external model response fields to our required schema
      // Below is a conservative mapper assuming plausible keys; customize as needed.
      const failure_probability = clamp01(
        data.failure_probability ?? data.failureProb ?? data.score ?? 0.5
      );
      const health_scores = data.health_scores ?? data.scores ?? { overall: 1 - failure_probability };
      const critical_parameter = data.critical_parameter ?? data.top_feature ?? 'vibration';

      const start = yyyymmdd(new Date(Date.now() + 1000 * 60 * 60 * 24 * 3));
      const end = yyyymmdd(new Date(Date.now() + 1000 * 60 * 60 * 24 * 10));
      const response = {
        machine_id,
        failure_probability,
        health_scores,
        critical_parameter,
        recommended_maintenance_window: { start, end },
        explanation: data.explanation ?? 'Model prediction'
      };

      console.log('[MODEL OK]', {
        machine_id,
        failure_probability: response.failure_probability,
        critical_parameter: response.critical_parameter
      });

      return response;
    } catch (err) {
      clearTimeout(to);
      lastErr = err;
      console.warn(`[MODEL TRY ${attempt} FAILED]`, err.message);
      if (attempt > maxRetries) break;
    }
  }
  throw lastErr || new Error('Model unreachable');
}

function deterministicFallback({ machine_id, params }) {
  const vib = Number(params?.vibration ?? 0.5);
  const temp = Number(params?.temperature ?? 0.5);
  const pres = Number(params?.pressure ?? 0.5);
  const fp = clamp01(0.5 * vib + 0.3 * temp + 0.2 * pres);
  const criticals = [
    { key: 'vibration', val: vib },
    { key: 'temperature', val: temp },
    { key: 'pressure', val: pres }
  ].sort((a, b) => b.val - a.val);

  const start = yyyymmdd(new Date(Date.now() + 1000 * 60 * 60 * 24 * 2));
  const end = yyyymmdd(new Date(Date.now() + 1000 * 60 * 60 * 24 * 9));

  return {
    machine_id,
    failure_probability: fp,
    health_scores: { overall: 1 - fp, vibration: 1 - vib, temperature: 1 - temp, pressure: 1 - pres },
    critical_parameter: criticals[0].key,
    recommended_maintenance_window: { start, end },
    explanation: 'Deterministic fallback used due to model failure'
  };
}

app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body || {};
  const user = users.find(u => u.email === email && u.password === password);
  if (!user) return res.status(401).json({ error: 'Invalid credentials' });
  const token = jwt.sign({ sub: user.id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '8h' });
  res.json({ token, user: { id: user.id, email: user.email, role: user.role, name: user.name } });
});

app.post('/api/auth/signup', (req, res) => {
  const { email, password, name } = req.body || {};
  if (!email || !password) return res.status(400).json({ error: 'email and password required' });
  if (users.find(u => u.email === email)) return res.status(400).json({ error: 'email already exists' });
  const user = { id: 'U' + (users.length + 1), email, password, role: 'user', name: name || email.split('@')[0] };
  users.push(user);
  const token = jwt.sign({ sub: user.id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '8h' });
  res.json({ token, user: { id: user.id, email: user.email, role: user.role, name: user.name } });
});

app.get('/api/machines', authMiddleware, (req, res) => {
  res.json(machines);
});

app.get('/api/machines/:id/history', authMiddleware, (req, res) => {
  const { id } = req.params;
  const m = machines.find(x => x.id === id);
  if (!m) return res.status(404).json({ error: 'Not found' });
  const now = new Date();
  const months = [];
  for (let i = 11; i >= 0; i--) {
    const dt = new Date(now.getFullYear(), now.getMonth() - i, 1);
    const baseline = m.failure_probability;
    const seasonal = 0.1 * Math.sin((i / 12) * Math.PI * 2);
    const noise = 0.05 * Math.sin((i / 3) * Math.PI);
    const p = clamp01(baseline + seasonal + noise);
    months.push({ month: dt.toISOString().slice(0, 7), failure_probability: p });
  }
  res.json({
    machine_id: id,
    monthly: months,
    recent_predictions: recentPredictions[id] || [],
    next_30_days: futurePredictions[id] || [],
    param_history: (paramHistory[id] || []).slice(-50).reverse()
  });
});

app.post('/api/predict', authMiddleware, async (req, res) => {
  // New request shape: { machine_id, timestamp, features: {...} }
  const { machine_id, timestamp, features, params } = req.body || {};
  const feat = features || params || {};
  if (!machine_id) return res.status(400).json({ error: 'machine_id required' });

  let result;
  try {
    result = await callHostedModel({ machine_id, params: feat });
  } catch (e) {
    console.warn('[MODEL FALLBACK]', e.message);
    result = deterministicFallback({ machine_id, params: feat });
    result.explanation = 'fallback: local heuristic';
  }

  // Update machine snapshot
  const idx = machines.findIndex(m => m.id === machine_id);
  if (idx >= 0) {
    machines[idx].failure_probability = result.failure_probability;
    machines[idx].last_health = clamp01(1 - result.failure_probability);
    machines[idx].most_critical = result.critical_parameter;
    machines[idx].last_checked_at = new Date().toISOString();
  }

  // Log summary only
  console.log('[PREDICT]', {
    machine_id,
    failure_probability: result.failure_probability,
    critical_parameter: result.critical_parameter
  });

  // Save recent predictions (cap 10)
  if (!recentPredictions[machine_id]) recentPredictions[machine_id] = [];
  recentPredictions[machine_id].unshift({
    at: timestamp || new Date().toISOString(),
    failure_probability: result.failure_probability,
    critical_parameter: result.critical_parameter
  });
  recentPredictions[machine_id] = recentPredictions[machine_id].slice(0, 10);

  res.json(result);
});

// Upload CSV -> validate -> trigger training job -> simulate training & produce next-30-days predictions
// Expected CSV headers: machine_id,timestamp,temperature,vibration,pressure,humidity
app.post('/api/upload', authMiddleware, upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'file required' });
  const content = req.file.buffer.toString('utf-8');
  const lines = content.split(/\r?\n/).filter(l => l && l.trim().length > 0);
  if (lines.length < 2) return res.status(400).json({ error: 'CSV has no data' });
  // Normalize header: strip BOM, trim, lowercase
  let rawHeader = lines[0].replace(/^\uFEFF/, '').trim();
  const header = rawHeader.split(',').map(s => s.trim().toLowerCase());
  const provided = header.join(',');
  // Accept common aliases
  const aliases = {
    machine_id: ['machine_id','machine','machineid','id'],
    timestamp: ['timestamp','time','date','record_date','datetime'],
    temperature: ['temperature','temp','tmp'],
    vibration: ['vibration','vibe','vibr'],
    pressure: ['pressure','press','psi'],
    humidity: ['humidity','humid','rh']
  };
  const idx = {};
  for (const [key, list] of Object.entries(aliases)) {
    const i = header.findIndex(h => list.includes(h));
    if (i >= 0) idx[key] = i;
  }
  const required = Object.keys(aliases);
  const missing = required.filter(k => !(k in idx));
  if (missing.length) {
    return res.status(400).json({
      error: `missing columns: ${missing.join(', ')}`,
      provided_headers: provided,
      note: 'Headers are case-insensitive. Acceptable aliases: ' + Object.entries(aliases).map(([k,v])=>`${k}=>${v.join('/')}`).join('; ')
    });
  }

  const jobId = `T${trainingCounter++}`;
  const job = { id: jobId, status: 'queued', progress: 0, startedAt: new Date().toISOString(), completedAt: null };
  trainingJobs.set(jobId, job);

  // simulate async training
  setTimeout(() => {
    job.status = 'running';
    let step = 0;
    const timer = setInterval(() => {
      step += 1;
      job.progress = Math.min(95, step * 15);
      if (step >= 6) {
        clearInterval(timer);
        try {
          // compute simple aggregates from CSV and generate future predictions per machine
          const rows = lines.slice(1).map(l => l.split(',').map(x=>x.trim()));
          const byMachine = new Map();
          for (const r of rows) {
            const mid = r[idx.machine_id];
            if (!byMachine.has(mid)) byMachine.set(mid, []);
            byMachine.get(mid).push({
              timestamp: r[idx.timestamp],
              temperature: Number(r[idx.temperature] || 0),
              vibration: Number(r[idx.vibration] || 0),
              pressure: Number(r[idx.pressure] || 0),
              humidity: Number(r[idx.humidity] || 0)
            });
          }
          const now = new Date();
          byMachine.forEach((arr, mid) => {
            const avg = arr.reduce((a,b)=>({
              temperature: a.temperature + b.temperature,
              vibration: a.vibration + b.vibration,
              pressure: a.pressure + b.pressure,
              humidity: a.humidity + b.humidity
            }), {temperature:0,vibration:0,pressure:0,humidity:0});
            const n = arr.length || 1;
            const mean = { temperature: avg.temperature/n, vibration: avg.vibration/n, pressure: avg.pressure/n, humidity: avg.humidity/n };
            // heuristic failure probability based on normalized means
            const vib = clamp01(mean.vibration / 10);
            const temp = clamp01(mean.temperature / 100);
            const pres = clamp01(mean.pressure / 200);
            const hum = clamp01(mean.humidity / 100);
            const baseFp = clamp01(0.45*vib + 0.25*temp + 0.2*pres + 0.1*hum);
            const future = [];
            for (let d=1; d<=30; d++) {
              const t = new Date(now.getTime() + d*24*3600*1000);
              const seasonal = 0.05 * Math.sin((d/30)*Math.PI*2);
              const fp = clamp01(baseFp + seasonal);
              future.push({ date: yyyymmdd(t), failure_probability: fp });
            }
            futurePredictions[mid] = future;
            // save param history (most recent 100)
            if (!paramHistory[mid]) paramHistory[mid] = [];
            const ph = arr.slice(-100).map(row => ({
              timestamp: row.timestamp,
              temperature: row.temperature,
              vibration: row.vibration,
              pressure: row.pressure,
              humidity: row.humidity
            }));
            paramHistory[mid].push(...ph);
          });
          job.progress = 100;
          job.status = 'completed';
          job.completedAt = new Date().toISOString();
        } catch (e) {
          job.status = 'failed';
          job.error = e.message;
          job.completedAt = new Date().toISOString();
        }
      }
    }, 400);
  }, 300);

  res.json({ training_job_id: jobId, status: job.status, progress: job.progress });
});

app.get('/api/upload/:id', authMiddleware, (req, res) => {
  const job = trainingJobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'not found' });
  res.json(job);
});

// CSV download for predictions
app.get('/api/machines/:id/predictions.csv', authMiddleware, (req, res) => {
  const id = req.params.id;
  const future = futurePredictions[id] || [];
  const recent = (recentPredictions[id] || []).map(p => ({ date: p.at.slice(0,10), failure_probability: p.failure_probability }));
  const rows = [['date','failure_probability']].concat(
    [...recent, ...future].map(r => [r.date, r.failure_probability])
  );
  const csv = rows.map(r=>r.join(',')).join('\n');
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename="${id}_predictions.csv"`);
  res.send(csv);
});

app.get('/health', async (req, res) => {
  let model_reachable = false;
  if (MODEL_URL) {
    try {
      const controller = new AbortController();
      const to = setTimeout(() => controller.abort(), 2000);
      const r = await fetch(MODEL_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ping: true }),
        signal: controller.signal
      });
      clearTimeout(to);
      model_reachable = r.ok;
    } catch (e) {
      model_reachable = false;
    }
  }
  res.json({ status: 'ok', model_reachable });
});

// Fallback: serve index.html at root
app.get('/', (req, res) => {
  res.sendFile(path.join(frontendDir, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`PDM backend running on http://localhost:${PORT}`);
});
