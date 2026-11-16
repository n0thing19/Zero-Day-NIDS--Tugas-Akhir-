const APP_STATE = {
  backendUrl: 'http://127.0.0.1:8080',
  isBackendAlive: false,
  isRunning: true,
  speedMs: 1000,
  timerId: null,
  percentile: 97.0,
  tau: 0.9832,
  activeAttacks: [],
  totalFlows: 0,
  anomalyFlows: 0,
  flowHistory: [],
  chartHistory: [],
  experiments: [
    { id: 'EXP-01', model: 'Max-Rank ECDF (Optimal)', pct: '97.0%', tau: 0.9832, fpr: '2.93%', recall: '90.50%', f1: 0.9358 },
    { id: 'EXP-02', model: 'Max-Rank ECDF (EER / F1-Max)', pct: '94.0%', tau: 0.9648, fpr: '5.90%', recall: '94.51%', f1: 0.9432 },
    { id: 'EXP-03', model: 'Max-Rank ECDF (Standard)', pct: '95.0%', tau: 0.9708, fpr: '4.93%', recall: '93.24%', f1: 0.9410 }
  ]
};

const MENU_TITLES = {
  'menu-1': 'Live Stream NetFlow Monitoring',
  'menu-2': 'Model Comparison & Benchmark (SAE vs IForest vs Max-Rank)',
  'menu-3': 'Threshold & Sensitivity Laboratory',
  'menu-4': 'Anomaly Explorer & Forensic Explanation',
  'menu-5': 'CSV Dataset Batch Analysis',
  'menu-6': 'Experiment History & Artifact Mapping'
};

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  sidebar.classList.toggle('collapsed');
}

function switchMenu(menuId) {
  document.querySelectorAll('.menu-section').forEach(sec => sec.classList.add('hidden'));
  document.getElementById(menuId).classList.remove('hidden');

  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.className = 'nav-item w-full flex items-center px-3 py-2 rounded text-xs font-medium text-zinc-400 hover:text-zinc-200 hover:bg-surface-800/50 transition';
  });

  const activeBtn = document.getElementById('nav-' + menuId);
  if (activeBtn) {
    activeBtn.className = 'nav-item w-full flex items-center px-3 py-2 rounded text-xs font-medium transition bg-surface-800 text-white border border-surface-700';
  }

  document.getElementById('pageHeaderTitle').textContent = MENU_TITLES[menuId];

  if (menuId === 'menu-2') {
    initComparisonCharts();
  }
}

async function checkBackendConnection() {
  const badge = document.getElementById('engineStatusBadge');
  try {
    const res = await fetch(`${APP_STATE.backendUrl}/api/status`, { method: 'GET' });
    if (res.ok) {
      APP_STATE.isBackendAlive = true;
      badge.textContent = 'API Live';
      badge.className = 'font-mono text-[11px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30';
      document.getElementById('latestFlowSourceBadge').textContent = 'Source: PyTorch SAE + Joblib IForest (Parquet Live)';
      return;
    }
  } catch (_) {}
  
  APP_STATE.isBackendAlive = false;
  badge.textContent = 'Offline';
  badge.className = 'font-mono text-[11px] px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-400 border border-rose-500/30';
  document.getElementById('latestFlowSourceBadge').textContent = 'Backend Offline — Jalankan python web/app.py';
}

async function fetchNextFlow() {
  if (!APP_STATE.isBackendAlive) {
    await checkBackendConnection();
    if (!APP_STATE.isBackendAlive) return null;
  }

  const hasAttack = APP_STATE.activeAttacks.length > 0;
  const chosenAttack = hasAttack
    ? APP_STATE.activeAttacks[Math.floor(Math.random() * APP_STATE.activeAttacks.length)]
    : 'Benign';

  try {
    const response = await fetch(`${APP_STATE.backendUrl}/api/sample-flow?category=${encodeURIComponent(chosenAttack)}&tau=${APP_STATE.tau}`);
    if (response.ok) {
      return await response.json();
    }
  } catch (_) {
    APP_STATE.isBackendAlive = false;
    checkBackendConnection();
  }
  return null;
}

async function stepSimulation() {
  const flow = await fetchNextFlow();
  if (!flow) return;

  APP_STATE.totalFlows++;
  if (flow.verdict === 'ANOMALY') APP_STATE.anomalyFlows++;

  APP_STATE.flowHistory.unshift(flow);
  if (APP_STATE.flowHistory.length > 200) APP_STATE.flowHistory.pop();

  APP_STATE.chartHistory.push(flow);
  if (APP_STATE.chartHistory.length > 25) APP_STATE.chartHistory.shift();

  updateDashboardUI(flow);
  updateStreamChart();
  renderLogTable();
}

function updateDashboardUI(flow) {
  document.getElementById('kpiTotal').textContent = APP_STATE.totalFlows.toLocaleString();
  document.getElementById('kpiAnomalies').textContent = APP_STATE.anomalyFlows.toLocaleString();
  const rate = ((APP_STATE.anomalyFlows / APP_STATE.totalFlows) * 100).toFixed(1);
  document.getElementById('kpiAnomalyRate').textContent = `${rate}% of traffic`;

  document.getElementById('lastFlowId').textContent = `${flow.id} (:${flow.port})`;
  const truthEl = document.getElementById('lastFlowTruth');
  truthEl.textContent = flow.groundTruth;
  truthEl.className = `font-medium mt-0.5 ${flow.isAttack ? 'text-rose-400' : 'text-emerald-400'}`;

  document.getElementById('lastFlowSub').textContent = `${flow.saeCdf.toFixed(3)} / ${flow.ifCdf.toFixed(3)}`;

  const verdictEl = document.getElementById('lastFlowVerdict');
  verdictEl.textContent = `${flow.verdict} (${flow.maxRank.toFixed(4)})`;
  verdictEl.className = `font-semibold mt-0.5 ${flow.verdict === 'ANOMALY' ? 'text-rose-400' : 'text-emerald-400'}`;

  const latencyEl = document.getElementById('lastFlowLatency');
  if (latencyEl) {
    latencyEl.textContent = `${flow.latencyMs} ms`;
  }
}

function toggleSimulation() {
  const btn = document.getElementById('btnPlayPause');
  const chip = document.getElementById('streamStatusChip');

  if (APP_STATE.isRunning) {
    clearInterval(APP_STATE.timerId);
    APP_STATE.isRunning = false;
    btn.textContent = 'Resume Stream';
    chip.textContent = 'PAUSED';
    chip.className = 'font-mono text-[11px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20';
  } else {
    APP_STATE.timerId = setInterval(stepSimulation, APP_STATE.speedMs);
    APP_STATE.isRunning = true;
    btn.textContent = 'Pause Stream';
    chip.textContent = 'RUNNING';
    chip.className = 'font-mono text-[11px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
  }
}

function resetSimulation() {
  APP_STATE.totalFlows = 0;
  APP_STATE.anomalyFlows = 0;
  APP_STATE.flowHistory = [];
  APP_STATE.chartHistory = [];
  document.getElementById('kpiTotal').textContent = '0';
  document.getElementById('kpiAnomalies').textContent = '0';
  updateStreamChart();
  renderLogTable();
}

function changeSpeed(val) {
  APP_STATE.speedMs = parseInt(val);
  document.getElementById('speedLabel').textContent = `${val} ms`;
  document.getElementById('kpiIntervalText').textContent = `${val}ms`;
  if (APP_STATE.isRunning) {
    clearInterval(APP_STATE.timerId);
    APP_STATE.timerId = setInterval(stepSimulation, APP_STATE.speedMs);
  }
}

function toggleAttack(attackName, isChecked) {
  if (isChecked) {
    if (!APP_STATE.activeAttacks.includes(attackName)) APP_STATE.activeAttacks.push(attackName);
  } else {
    APP_STATE.activeAttacks = APP_STATE.activeAttacks.filter(a => a !== attackName);
  }
}

let streamChart = null;
let rocChart = null;
let classChart = null;

function initStreamChart() {
  const ctx = document.getElementById('liveTrajectoryChart').getContext('2d');
  streamChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Max-Rank ECDF',
          data: [],
          borderColor: '#38bdf8',
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 2
        },
        {
          label: 'SAE CDF',
          data: [],
          borderColor: '#fbbf24',
          borderDash: [4, 4],
          borderWidth: 1.2,
          pointRadius: 0
        },
        {
          label: 'IF(Z) CDF',
          data: [],
          borderColor: '#818cf8',
          borderDash: [2, 2],
          borderWidth: 1.2,
          pointRadius: 0
        },
        {
          label: 'Cutoff Tau',
          data: [],
          borderColor: '#f43f5e',
          borderWidth: 1.2,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 1.05, grid: { color: '#1e1e24' }, ticks: { color: '#71717a' } },
        x: { grid: { display: false }, ticks: { color: '#71717a' } }
      }
    }
  });
}

function updateStreamChart() {
  if (!streamChart) return;
  streamChart.data.labels = APP_STATE.chartHistory.map(f => f.time);
  streamChart.data.datasets[0].data = APP_STATE.chartHistory.map(f => f.maxRank);
  streamChart.data.datasets[1].data = APP_STATE.chartHistory.map(f => f.saeCdf);
  streamChart.data.datasets[2].data = APP_STATE.chartHistory.map(f => f.ifCdf);
  streamChart.data.datasets[3].data = APP_STATE.chartHistory.map(() => APP_STATE.tau);
  streamChart.update();
}

function initComparisonCharts() {
  if (!rocChart) {
    const ctxRoc = document.getElementById('rocComparisonChart').getContext('2d');
    rocChart = new Chart(ctxRoc, {
      type: 'line',
      data: {
        labels: ['0.00', '0.03 (P97)', '0.05 (P95)', '0.10', '0.25', '0.50', '1.00'],
        datasets: [
          {
            label: 'Proposed Max-Rank (AUC = 0.9834)',
            data: [0, 0.9050, 0.9324, 0.968, 0.989, 0.996, 1.0],
            borderColor: '#38bdf8',
            borderWidth: 2
          },
          {
            label: 'SAE Alone (AUC = 0.8813)',
            data: [0, 0.610, 0.6673, 0.750, 0.840, 0.920, 1.0],
            borderColor: '#fbbf24',
            borderDash: [4, 4],
            borderWidth: 1.5
          },
          {
            label: 'IForest Alone (AUC = 0.8672)',
            data: [0, 0.620, 0.6940, 0.740, 0.820, 0.900, 1.0],
            borderColor: '#818cf8',
            borderDash: [2, 2],
            borderWidth: 1.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#a1a1aa' } } }
      }
    });
  }

  if (!classChart) {
    const ctxClass = document.getElementById('classRecallChart').getContext('2d');
    classChart = new Chart(ctxClass, {
      type: 'bar',
      data: {
        labels: ['Bot (100%)', 'Recon (89.3%)', 'DDoS (87.96%)', 'DoS (87.64%)', 'Brute Force (87.62%)'],
        datasets: [{
          label: 'Recall (%) pada P97',
          data: [100.0, 89.30, 87.96, 87.64, 87.62],
          backgroundColor: '#0284c7',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { min: 70, max: 100 } }
      }
    });
  }
}

const EXACT_SENSITIVITY_DATA = {
  "80.0": { tau: 0.8728, fpr: 19.12, fnr: 1.80, recall: 98.20, prec: 83.70, f1: 0.9037 },
  "80.5": { tau: 0.8751, fpr: 18.82, fnr: 1.80, recall: 98.20, prec: 83.91, f1: 0.9050 },
  "81.0": { tau: 0.8793, fpr: 18.16, fnr: 1.82, recall: 98.18, prec: 84.39, f1: 0.9076 },
  "81.5": { tau: 0.8816, fpr: 17.86, fnr: 1.83, recall: 98.17, prec: 84.61, f1: 0.9089 },
  "82.0": { tau: 0.8850, fpr: 17.32, fnr: 1.86, recall: 98.14, prec: 85.00, f1: 0.9110 },
  "82.5": { tau: 0.8882, fpr: 16.96, fnr: 1.88, recall: 98.12, prec: 85.26, f1: 0.9124 },
  "83.0": { tau: 0.8911, fpr: 16.42, fnr: 1.88, recall: 98.12, prec: 85.66, f1: 0.9147 },
  "83.5": { tau: 0.8943, fpr: 15.92, fnr: 1.89, recall: 98.11, prec: 86.04, f1: 0.9168 },
  "84.0": { tau: 0.8989, fpr: 15.46, fnr: 1.90, recall: 98.10, prec: 86.38, f1: 0.9187 },
  "84.5": { tau: 0.9021, fpr: 15.05, fnr: 1.90, recall: 98.10, prec: 86.70, f1: 0.9205 },
  "85.0": { tau: 0.9066, fpr: 14.51, fnr: 1.92, recall: 98.08, prec: 87.11, f1: 0.9227 },
  "85.5": { tau: 0.9096, fpr: 13.96, fnr: 1.94, recall: 98.06, prec: 87.54, f1: 0.9250 },
  "86.0": { tau: 0.9131, fpr: 13.55, fnr: 1.97, recall: 98.03, prec: 87.85, f1: 0.9266 },
  "86.5": { tau: 0.9165, fpr: 13.14, fnr: 2.03, recall: 97.97, prec: 88.17, f1: 0.9282 },
  "87.0": { tau: 0.9201, fpr: 12.47, fnr: 3.32, recall: 96.68, prec: 88.57, f1: 0.9245 },
  "87.5": { tau: 0.9230, fpr: 12.05, fnr: 3.38, recall: 96.62, prec: 88.91, f1: 0.9261 },
  "88.0": { tau: 0.9273, fpr: 11.54, fnr: 3.47, recall: 96.53, prec: 89.32, f1: 0.9279 },
  "88.5": { tau: 0.9313, fpr: 10.90, fnr: 3.50, recall: 96.50, prec: 89.85, f1: 0.9306 },
  "89.0": { tau: 0.9346, fpr: 10.60, fnr: 3.59, recall: 96.41, prec: 90.09, f1: 0.9314 },
  "89.5": { tau: 0.9379, fpr: 10.18, fnr: 3.68, recall: 96.32, prec: 90.44, f1: 0.9329 },
  "90.0": { tau: 0.9409, fpr: 9.66, fnr: 3.91, recall: 96.09, prec: 90.87, f1: 0.9341 },
  "90.5": { tau: 0.9437, fpr: 9.12, fnr: 4.04, recall: 95.96, prec: 91.32, f1: 0.9359 },
  "91.0": { tau: 0.9474, fpr: 8.69, fnr: 4.24, recall: 95.76, prec: 91.68, f1: 0.9367 },
  "91.5": { tau: 0.9498, fpr: 8.17, fnr: 4.32, recall: 95.68, prec: 92.13, f1: 0.9388 },
  "92.0": { tau: 0.9527, fpr: 7.68, fnr: 4.43, recall: 95.57, prec: 92.57, f1: 0.9404 },
  "92.5": { tau: 0.9560, fpr: 7.23, fnr: 5.12, recall: 94.88, prec: 92.92, f1: 0.9389 },
  "93.0": { tau: 0.9588, fpr: 6.85, fnr: 5.18, recall: 94.82, prec: 93.26, f1: 0.9404 },
  "93.5": { tau: 0.9619, fpr: 6.34, fnr: 5.32, recall: 94.68, prec: 93.72, f1: 0.9420 },
  "94.0": { tau: 0.9648, fpr: 5.90, fnr: 5.49, recall: 94.51, prec: 94.12, f1: 0.9432 },
  "94.5": { tau: 0.9678, fpr: 5.42, fnr: 5.96, recall: 94.04, prec: 94.55, f1: 0.9429 },
  "95.0": { tau: 0.9708, fpr: 4.93, fnr: 6.76, recall: 93.24, prec: 94.98, f1: 0.9410 },
  "95.5": { tau: 0.9737, fpr: 4.49, fnr: 7.25, recall: 92.75, prec: 95.38, f1: 0.9405 },
  "96.0": { tau: 0.9769, fpr: 3.94, fnr: 8.14, recall: 91.86, prec: 95.89, f1: 0.9383 },
  "96.5": { tau: 0.9802, fpr: 3.34, fnr: 8.72, recall: 91.28, prec: 96.47, f1: 0.9380 },
  "97.0": { tau: 0.9832, fpr: 2.93, fnr: 9.50, recall: 90.50, prec: 96.87, f1: 0.9358 },
  "97.5": { tau: 0.9864, fpr: 2.31, fnr: 10.24, recall: 89.76, prec: 97.49, f1: 0.9346 },
  "98.0": { tau: 0.9891, fpr: 1.73, fnr: 10.88, recall: 89.12, prec: 98.10, f1: 0.9339 },
  "98.5": { tau: 0.9923, fpr: 1.21, fnr: 12.69, recall: 87.31, prec: 98.63, f1: 0.9262 },
  "99.0": { tau: 0.9948, fpr: 0.84, fnr: 32.78, recall: 67.22, prec: 98.77, f1: 0.7999 }
};

function setLabPercentile(pct) {
  document.getElementById('labSlider').value = pct;
  updateLabMetrics(pct);
}

function updateLabMetrics(pctVal) {
  const p = parseFloat(pctVal).toFixed(1);
  const data = EXACT_SENSITIVITY_DATA[p] || EXACT_SENSITIVITY_DATA["97.0"];

  APP_STATE.percentile = parseFloat(p);
  APP_STATE.tau = data.tau;

  document.getElementById('labPercentileVal').textContent = `${p}%`;
  document.getElementById('labTauVal').textContent = data.tau.toFixed(4);
  document.getElementById('headerThresholdText').textContent = `P${parseFloat(p).toFixed(0)} (\u03C4 = ${data.tau.toFixed(4)})`;

  const fp = Math.round((data.fpr / 100) * 25000);
  const tn = 25000 - fp;
  const tp = Math.round((data.recall / 100) * 25000);
  const fn = 25000 - tp;

  document.getElementById('labTp').textContent = tp.toLocaleString();
  document.getElementById('labFp').textContent = fp.toLocaleString();
  document.getElementById('labFn').textContent = fn.toLocaleString();
  document.getElementById('labTn').textContent = tn.toLocaleString();

  document.getElementById('labFpr').textContent = `${data.fpr.toFixed(2)}%`;
  document.getElementById('labRecall').textContent = `${data.recall.toFixed(2)}%`;
  document.getElementById('labPrecision').textContent = `${data.prec.toFixed(2)}%`;
  document.getElementById('labF1').textContent = data.f1.toFixed(4);

  const noteEl = document.getElementById('labTradeoffNote');
  if (p === "97.0") {
    noteEl.innerHTML = '<span class="text-emerald-400 font-semibold">✓ Usulan Skripsi (P97):</span> FPR ditekan di bawah 3.00% (2.93%, FP: 732) dengan Recall tetap tinggi di atas 90.50% (TP: 22.626).';
  } else if (p === "94.0") {
    noteEl.innerHTML = '<span class="text-sky-400 font-semibold">★ Titik EER & F1 Maksimum (P94):</span> Titik temu FPR dan FNR seimbang (~5.69%) dengan F1-Score tertinggi (0.9432).';
  } else if (p === "95.0") {
    noteEl.innerHTML = '<span class="text-amber-400 font-semibold">● Standar Industri (P95):</span> Batas psikologis FPR di bawah 5.00% (4.93%) dengan Recall mencapai 93.24%.';
  } else if (parseFloat(p) < 94.0) {
    noteEl.innerHTML = '<span class="text-rose-400 font-semibold">⚠ Alert Fatigue High:</span> FPR terlalu longgar (> 6.00%), memicu alarm palsu pada lalu lintas normal.';
  } else {
    noteEl.innerHTML = '<span class="text-indigo-400 font-semibold">ℹ High Specificity:</span> Alarm palsu sangat minim, namun sebagian serangan bertipe stealthy mulai terlewat.';
  }
}

function renderLogTable() {
  const tbody = document.getElementById('logTableBody');
  const filter = document.getElementById('filterVerdict').value;
  const filtered = APP_STATE.flowHistory.filter(f => filter === 'all' || f.verdict === filter);

  tbody.innerHTML = filtered.slice(0, 20).map(f => `
    <tr class="hover:bg-surface-800/60">
      <td class="p-3 text-zinc-400">${f.time}</td>
      <td class="p-3 font-medium text-white">${f.id}</td>
      <td class="p-3 ${f.isAttack ? 'text-rose-400' : 'text-emerald-400'}">${f.groundTruth}</td>
      <td class="p-3">${f.saeCdf.toFixed(4)}</td>
      <td class="p-3">${f.ifCdf.toFixed(4)}</td>
      <td class="p-3 font-semibold text-sky-400">${f.maxRank.toFixed(4)}</td>
      <td class="p-3">
        <span class="px-1.5 py-0.5 rounded text-[11px] ${f.verdict === 'ANOMALY' ? 'bg-rose-500/15 text-rose-400' : 'bg-emerald-500/15 text-emerald-400'}">
          ${f.verdict}
        </span>
      </td>
      <td class="p-3 font-semibold">${f.evalStatus}</td>
      <td class="p-3 text-right">
        <button onclick="openForensicModal('${f.id}')" class="text-sky-400 hover:underline">
          Detail
        </button>
      </td>
    </tr>
  `).join('');
}

function openForensicModal(flowId) {
  const f = APP_STATE.flowHistory.find(item => item.id === flowId);
  if (!f) return;

  document.getElementById('modalFlowId').textContent = `${f.id} (Port ${f.port})`;
  document.getElementById('modalTruth').textContent = f.groundTruth;
  document.getElementById('modalTruth').className = `font-semibold mt-0.5 ${f.isAttack ? 'text-rose-400' : 'text-emerald-400'}`;
  document.getElementById('modalMaxRank').textContent = f.maxRank.toFixed(4);

  const diff = f.maxRank - APP_STATE.tau;
  document.getElementById('modalMargin').textContent = (diff >= 0 ? '+' : '') + diff.toFixed(4);
  document.getElementById('modalMargin').className = `font-semibold mt-0.5 ${diff >= 0 ? 'text-rose-400' : 'text-emerald-400'}`;

  document.getElementById('modalSaeRaw').textContent = f.saeRaw.toFixed(6);
  document.getElementById('modalSaeCdf').textContent = f.saeCdf.toFixed(4);
  document.getElementById('modalIfRaw').textContent = f.ifRaw.toFixed(4);
  document.getElementById('modalIfCdf').textContent = f.ifCdf.toFixed(4);

  const topListEl = document.getElementById('modalTopFeaturesList');
  if (f.topFeatures && f.topFeatures.length > 0) {
    topListEl.innerHTML = f.topFeatures.map((tf, idx) => `
      <div class="flex items-center justify-between p-2 rounded bg-surface-900 border border-surface-800">
        <span class="text-zinc-300 font-medium">${idx + 1}. ${tf.feature || tf.name}</span>
        <span class="text-amber-400 font-bold">MSE: ${(tf.error || tf.mse).toFixed(6)}</span>
      </div>
    `).join('');
  } else {
    topListEl.innerHTML = `<div class="p-2 text-zinc-500">Data fitur tidak tersedia.</div>`;
  }

  const trigger = f.saeCdf >= f.ifCdf ? 'SAE Reconstruction Error (Level 1)' : 'Isolation Forest Latent Z (Level 2)';
  document.getElementById('modalReasonText').innerHTML = f.verdict === 'ANOMALY'
    ? `Flagged as <b>ANOMALY</b>: Max-Rank (<b>${f.maxRank.toFixed(4)}</b>) melampaui ambang batas operasional &tau; (<b>${APP_STATE.tau.toFixed(4)}</b>). Pemicu dominan berasal dari <b>${trigger}</b>.`
    : `Classified as <b>NORMAL</b>: Baik peringkat ECDF SAE (${f.saeCdf.toFixed(4)}) maupun IForest (${f.ifCdf.toFixed(4)}) berada di bawah ambang batas &tau; (${APP_STATE.tau.toFixed(4)}).`;

  document.getElementById('forensicModal').classList.remove('hidden');
}

function closeForensicModal() {
  document.getElementById('forensicModal').classList.add('hidden');
}

function exportLogsCSV() {
  if (APP_STATE.flowHistory.length === 0) return;
  let csv = 'Time,Flow_ID,Ground_Truth,SAE_CDF,IF_CDF,Max_Rank,Verdict,Evaluation\n';
  APP_STATE.flowHistory.forEach(f => {
    csv += `${f.time},${f.id},${f.groundTruth},${f.saeCdf.toFixed(4)},${f.ifCdf.toFixed(4)},${f.maxRank.toFixed(4)},${f.verdict},${f.evalStatus}\n`;
  });
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'nids_live_records.csv';
  a.click();
}

function handleCSVFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const lines = e.target.result.trim().split('\n');
    const rowCount = Math.max(0, lines.length - 1);
    document.getElementById('csvResultBox').classList.remove('hidden');
    document.getElementById('csvName').textContent = file.name;
    document.getElementById('csvRows').textContent = rowCount;
    document.getElementById('csvStatus').textContent = 'Loaded into Buffer';
  };
  reader.readAsText(file);
}

function renderExperiments() {
  const tbody = document.getElementById('experimentTableBody');
  tbody.innerHTML = APP_STATE.experiments.map(exp => `
    <tr>
      <td class="p-3 font-semibold text-white">${exp.id}</td>
      <td class="p-3 font-sans text-sky-400">${exp.model}</td>
      <td class="p-3">${exp.pct}</td>
      <td class="p-3 text-amber-400">${Number(exp.tau).toFixed(4)}</td>
      <td class="p-3 text-emerald-400 font-semibold">${exp.fpr}</td>
      <td class="p-3 text-sky-400 font-semibold">${exp.recall}</td>
      <td class="p-3 text-white font-semibold">${exp.f1}</td>
    </tr>
  `).join('');
}

function saveExperimentBookmark() {
  const id = 'EXP-0' + (APP_STATE.experiments.length + 1);
  APP_STATE.experiments.unshift({
    id: id,
    model: `Custom Max-Rank (P${APP_STATE.percentile.toFixed(1)})`,
    pct: `${APP_STATE.percentile.toFixed(1)}%`,
    tau: APP_STATE.tau,
    fpr: document.getElementById('labFpr').textContent,
    recall: document.getElementById('labRecall').textContent,
    f1: document.getElementById('labF1').textContent
  });
  renderExperiments();
  switchMenu('menu-6');
}

setInterval(() => {
  document.getElementById('systemClock').textContent = new Date().toLocaleTimeString('id-ID', { hour12: false }) + ' WIB';
}, 1000);

window.addEventListener('DOMContentLoaded', () => {
  initStreamChart();
  checkBackendConnection();
  renderExperiments();
  APP_STATE.timerId = setInterval(stepSimulation, APP_STATE.speedMs);
});