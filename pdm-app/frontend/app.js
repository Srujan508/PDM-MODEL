function getToken() {
  return localStorage.getItem('jwt');
}

function ensureAuthOrRedirect() {
  if (!getToken()) {
    window.location.href = '/login.html';
  }
}

function setupLogout() {
  const link = document.getElementById('logoutLink');
  if (link) {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      localStorage.removeItem('jwt');
      window.location.href = '/login.html';
    });
  }
}

async function apiGet(url) {
  const res = await fetch(url, {
    headers: { 'Authorization': `Bearer ${getToken()}` }
  });
  if (!res.ok) throw new Error('Request failed');
  return res.json();
}

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${getToken()}`
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error('Request failed');
  return res.json();
}

function healthColor(health01) {
  if (health01 > 0.8) return 'green';
  if (health01 >= 0.5) return 'yellow';
  return 'red';
}

function setAlertVisibility(show) {
  const el = document.getElementById('alert');
  if (!el) return;
  if (show) {
    el.classList.remove('hidden');
    el.textContent = 'Alert: One or more machines need attention (health < 50% or failure probability > 70%).';
  } else {
    el.classList.add('hidden');
  }
}

// UI helpers
function progressHtml(pct, label) {
  const v = Math.max(0, Math.min(100, Math.round(pct)));
  return `
    <div class="progress">
      <div class="bar" style="width:${v}%"></div>
      <span class="progress-label">${label || ''}</span>
    </div>
  `;
}

function fmtDate(d) {
  try { return new Date(d).toLocaleString(); } catch { return d; }
}

function download(url) {
  const a = document.createElement('a');
  a.href = url;
  a.click();
}
