/* ═══════════════════════════════════════════════════════════════
   ML Experiment Dashboard — JavaScript
   ═══════════════════════════════════════════════════════════════ */

// ─── State ───────────────────────────────────────────────────────
let currentConfig = null;
let selectedRunId = null;

// ─── Init ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    setupModal();
    loadConfig();
    loadExperiments();
});

// ─── Navigation ──────────────────────────────────────────────────
function setupNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const page = link.dataset.page;
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById(`page-${page}`).classList.add('active');
            if (page === 'experiments') loadExperiments();
        });
    });
}

// ─── Toast ───────────────────────────────────────────────────────
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
}

// ─── Config Editor ───────────────────────────────────────────────
async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        currentConfig = await res.json();
        renderConfigForm(currentConfig);
    } catch (e) {
        showToast('Failed to load config', 'error');
    }
}

function renderConfigForm(config) {
    const container = document.getElementById('config-form');
    container.innerHTML = '';

    const sections = [
        { key: 'dataset', title: '📂 Dataset', icon: '📂' },
        { key: 'forecasting', title: '🎯 Forecasting', icon: '🎯' },
        { key: 'feature_engineering', title: '🔧 Feature Engineering', icon: '🔧' },
        { key: 'validation', title: '✅ Validation', icon: '✅' },
        { key: 'model_params', title: '🧠 Model Parameters', icon: '🧠' },
    ];

    sections.forEach(sec => {
        if (!config[sec.key]) return;
        const card = document.createElement('div');
        card.className = 'glass-card';

        if (sec.key === 'model_params') {
            card.innerHTML = `<h3>${sec.title}</h3>`;
            // LSTM
            card.innerHTML += '<h4 style="color:var(--accent-2);font-size:13px;margin:12px 0 8px;font-weight:600;">LSTM</h4>';
            card.innerHTML += buildFields(config.model_params.lstm || {}, 'model_params.lstm');
            // XGBoost
            card.innerHTML += '<h4 style="color:var(--accent-2);font-size:13px;margin:16px 0 8px;font-weight:600;">XGBoost</h4>';
            card.innerHTML += buildFields(config.model_params.xgboost || {}, 'model_params.xgboost');
        } else if (sec.key === 'feature_engineering') {
            card.innerHTML = `<h3>${sec.title}</h3>`;
            // Toggle for add_time_features
            card.innerHTML += `
                <div class="form-group">
                    <label>Add Time Features</label>
                    <div class="toggle-wrap">
                        <input type="checkbox" class="toggle" data-path="feature_engineering.add_time_features"
                            ${config.feature_engineering.add_time_features ? 'checked' : ''}>
                        <span style="font-size:13px;color:var(--text-secondary)">
                            ${config.feature_engineering.add_time_features ? 'Enabled' : 'Disabled'}
                        </span>
                    </div>
                </div>`;
            // Lag columns
            const lagCols = (config.feature_engineering.lag_features?.columns || []).join(', ');
            card.innerHTML += `
                <div class="form-group">
                    <label>Lag Columns</label>
                    <textarea data-path="feature_engineering.lag_features.columns" data-type="string-array">${lagCols}</textarea>
                </div>`;
            // Lag values
            const lagVals = (config.feature_engineering.lag_features?.lags || []).join(', ');
            card.innerHTML += `
                <div class="form-group">
                    <label>Lag Values</label>
                    <input type="text" data-path="feature_engineering.lag_features.lags" data-type="number-array" value="${lagVals}">
                </div>`;
        } else {
            card.innerHTML = `<h3>${sec.title}</h3>` + buildFields(config[sec.key], sec.key);
        }

        container.appendChild(card);
    });

    // Toggle change handler
    container.querySelectorAll('.toggle').forEach(t => {
        t.addEventListener('change', () => {
            t.nextElementSibling.textContent = t.checked ? 'Enabled' : 'Disabled';
        });
    });

    // Buttons
    document.getElementById('btn-save-config').onclick = saveConfig;
    document.getElementById('btn-reset-config').onclick = loadConfig;
}

function buildFields(obj, prefix) {
    let html = '';
    for (const [key, value] of Object.entries(obj)) {
        const path = `${prefix}.${key}`;
        const label = key.replace(/_/g, ' ');

        if (Array.isArray(value)) {
            const isNumbers = value.every(v => typeof v === 'number');
            html += `<div class="form-group">
                <label>${label}</label>
                <input type="text" data-path="${path}" data-type="${isNumbers ? 'number-array' : 'string-array'}"
                    value="${value.join(', ')}">
            </div>`;
        } else if (typeof value === 'boolean') {
            html += `<div class="form-group">
                <label>${label}</label>
                <div class="toggle-wrap">
                    <input type="checkbox" class="toggle" data-path="${path}" ${value ? 'checked' : ''}>
                    <span style="font-size:13px;color:var(--text-secondary)">${value ? 'Enabled' : 'Disabled'}</span>
                </div>
            </div>`;
        } else if (typeof value === 'number') {
            html += `<div class="form-group">
                <label>${label}</label>
                <input type="number" data-path="${path}" value="${value}" step="any">
            </div>`;
        } else if (typeof value === 'object' && value !== null) {
            // Skip nested objects here (handled in special sections)
        } else {
            html += `<div class="form-group">
                <label>${label}</label>
                <input type="text" data-path="${path}" value="${value || ''}">
            </div>`;
        }
    }
    return html;
}

async function saveConfig() {
    const config = JSON.parse(JSON.stringify(currentConfig));

    document.querySelectorAll('[data-path]').forEach(el => {
        const path = el.dataset.path;
        const keys = path.split('.');
        let obj = config;

        for (let i = 0; i < keys.length - 1; i++) {
            if (!obj[keys[i]]) obj[keys[i]] = {};
            obj = obj[keys[i]];
        }

        const lastKey = keys[keys.length - 1];
        const dtype = el.dataset.type;

        if (el.type === 'checkbox') {
            obj[lastKey] = el.checked;
        } else if (dtype === 'number-array') {
            obj[lastKey] = el.value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
        } else if (dtype === 'string-array') {
            obj[lastKey] = el.value.split(',').map(s => s.trim()).filter(s => s);
        } else if (el.type === 'number') {
            obj[lastKey] = parseFloat(el.value);
        } else {
            obj[lastKey] = el.value;
        }
    });

    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (data.success) {
            currentConfig = config;
            showToast('Config saved successfully!', 'success');
        } else {
            showToast(data.error || 'Failed to save', 'error');
        }
    } catch (e) {
        showToast('Failed to save config', 'error');
    }
}

// ─── Experiments ─────────────────────────────────────────────────
async function loadExperiments() {
    try {
        const res = await fetch('/api/experiments');
        const experiments = await res.json();
        renderRunSelector(experiments);
    } catch (e) {
        showToast('Failed to load experiments', 'error');
    }
}

function renderRunSelector(experiments) {
    const container = document.getElementById('run-selector');

    if (!experiments.length) {
        container.innerHTML = `<div class="empty-state" style="grid-column:1/-1">
            <div class="empty-icon">📭</div>
            <p>No experiments found. Run one to get started!</p>
        </div>`;
        return;
    }

    container.innerHTML = experiments.map(exp => {
        const ts = formatTimestamp(exp.timestamp);
        return `<div class="run-card ${selectedRunId === exp.run_id ? 'selected' : ''}"
                    data-run="${exp.run_id}" onclick="selectExperiment('${exp.run_id}')">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div class="run-id">🧪 ${exp.run_id}</div>
                <span class="status-badge ${exp.status}">${exp.status}</span>
            </div>
            <div class="run-time">${ts}</div>
        </div>`;
    }).join('');
}

async function selectExperiment(runId) {
    selectedRunId = runId;

    // Update selection UI
    document.querySelectorAll('.run-card').forEach(c => c.classList.remove('selected'));
    document.querySelector(`.run-card[data-run="${runId}"]`)?.classList.add('selected');

    try {
        const res = await fetch(`/api/experiments/${runId}`);
        const data = await res.json();
        renderExperimentDetail(data);
    } catch (e) {
        showToast('Failed to load experiment details', 'error');
    }
}

function renderExperimentDetail(data) {
    const detail = document.getElementById('experiment-detail');
    detail.style.display = 'block';

    // ── Overview Section ──
    const overviewEl = document.getElementById('section-overview');
    let overviewHTML = `<h3>📋 Overview — ${data.run_id}</h3>`;

    // Config used (collapsible)
    if (data.config_used) {
        overviewHTML += `<div class="config-preview">
            <div class="config-preview-header" onclick="this.nextElementSibling.classList.toggle('open');this.querySelector('.arrow').textContent=this.nextElementSibling.classList.contains('open')?'▼':'▶'">
                <span>⚙️ Config Used</span><span class="arrow">▶</span>
            </div>
            <div class="config-preview-body"><code>${JSON.stringify(data.config_used, null, 2)}</code></div>
        </div>`;
    }

    // Benchmark summary table
    if (data.benchmark_summary && data.benchmark_summary.length) {
        overviewHTML += `<div class="glass-card" style="margin-bottom:20px">
            <h3>🏆 Benchmark Summary</h3>
            <table class="data-table">
                <thead><tr><th>Model</th><th>Average MSE</th><th>Average MAE</th></tr></thead>
                <tbody>`;
        data.benchmark_summary.forEach(row => {
            overviewHTML += `<tr>
                <td style="font-weight:600">${row.Model || ''}</td>
                <td class="num">${parseFloat(row['Average MSE'] || 0).toFixed(6)}</td>
                <td class="num">${parseFloat(row['Average MAE'] || 0).toFixed(6)}</td>
            </tr>`;
        });
        overviewHTML += `</tbody></table></div>`;
    }

    // Statistical tests table
    if (data.statistical_tests && data.statistical_tests.length) {
        overviewHTML += `<div class="glass-card" style="margin-bottom:20px">
            <h3>📐 Statistical Significance Tests</h3>
            <table class="data-table">
                <thead><tr><th>Model A</th><th>Model B</th><th>P-Value</th><th>Significant?</th><th>Winner</th></tr></thead>
                <tbody>`;
        data.statistical_tests.forEach(row => {
            const sig = row['Significant Diff? (p<0.05)'] === 'Yes';
            overviewHTML += `<tr>
                <td>${row['Model A'] || ''}</td>
                <td>${row['Model B'] || ''}</td>
                <td class="num">${parseFloat(row['Two-Tail P-Value'] || 0).toFixed(6)}</td>
                <td><span class="status-badge ${sig ? 'completed' : 'failed'}">${sig ? 'Yes' : 'No'}</span></td>
                <td style="font-weight:600">${row['Winner (One-Tail)'] || '-'}</td>
            </tr>`;
        });
        overviewHTML += `</tbody></table></div>`;
    }

    overviewEl.innerHTML = overviewHTML;

    // ── Model Tabs ──
    const modelNames = Object.keys(data.models || {});
    const tabsHeader = document.getElementById('tabs-header');
    const tabsContent = document.getElementById('tabs-content');

    if (modelNames.length) {
        tabsHeader.innerHTML = modelNames.map((name, i) =>
            `<button class="tab-btn ${i === 0 ? 'active' : ''}" data-tab="${name}" onclick="switchTab('${name}')">${capitalize(name)}</button>`
        ).join('');

        tabsContent.innerHTML = modelNames.map((name, i) => {
            const model = data.models[name];
            let paneHTML = `<div class="tab-pane ${i === 0 ? 'active' : ''}" id="tab-${name}">`;

            // Compute averages
            if (model.metrics) {
                const folds = Object.entries(model.metrics);
                const avgMSE = folds.reduce((s, [, v]) => s + v.MSE, 0) / folds.length;
                const avgMAE = folds.reduce((s, [, v]) => s + v.MAE, 0) / folds.length;

                paneHTML += `<div class="metric-cards">
                    <div class="metric-card">
                        <div class="metric-value">${avgMSE.toFixed(4)}</div>
                        <div class="metric-label">Avg MSE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${avgMAE.toFixed(4)}</div>
                        <div class="metric-label">Avg MAE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${folds.length}</div>
                        <div class="metric-label">Folds</div>
                    </div>
                </div>`;

                // Fold table
                paneHTML += `<div class="glass-card" style="margin-bottom:20px">
                    <h3>📊 Fold-by-Fold Metrics</h3>
                    <table class="data-table">
                        <thead><tr><th>Fold</th><th>MSE</th><th>MAE</th></tr></thead>
                        <tbody>`;
                // Sort folds naturally
                const sortedFolds = folds.sort((a, b) => {
                    const numA = parseInt(a[0].replace('fold_', ''));
                    const numB = parseInt(b[0].replace('fold_', ''));
                    return numA - numB;
                });
                sortedFolds.forEach(([fold, metrics]) => {
                    paneHTML += `<tr>
                        <td style="font-weight:500">${fold}</td>
                        <td class="num">${metrics.MSE.toFixed(6)}</td>
                        <td class="num">${metrics.MAE.toFixed(6)}</td>
                    </tr>`;
                });
                paneHTML += `</tbody></table></div>`;
            }

            // Model images (learning curves etc)
            if (model.images && model.images.length) {
                paneHTML += `<div class="glass-card"><h3>📈 Visualizations</h3><div class="image-gallery">`;
                model.images.forEach(img => {
                    const url = `/api/experiments/${data.run_id}/images/${name}/${img}`;
                    const label = img.replace('.png', '').replace(/_/g, ' ');
                    paneHTML += `<div class="image-card">
                        <img src="${url}" alt="${label}" loading="lazy" onclick="openLightbox('${url}')">
                        <div class="image-label">${label}</div>
                    </div>`;
                });
                paneHTML += `</div></div>`;
            }

            paneHTML += '</div>';
            return paneHTML;
        }).join('');
    } else {
        tabsHeader.innerHTML = '';
        tabsContent.innerHTML = '<div class="empty-state"><p>No model data found.</p></div>';
    }

    // ── Top-level Images ──
    const imagesEl = document.getElementById('section-images');
    if (data.images && data.images.length) {
        let imgHTML = `<h3>🖼️ Benchmark Visualizations</h3><div class="image-gallery">`;
        data.images.forEach(img => {
            const url = `/api/experiments/${data.run_id}/images/${img}`;
            const label = img.replace('.png', '').replace(/_/g, ' ');
            imgHTML += `<div class="image-card">
                <img src="${url}" alt="${label}" loading="lazy" onclick="openLightbox('${url}')">
                <div class="image-label">${label}</div>
            </div>`;
        });
        imgHTML += '</div>';
        imagesEl.innerHTML = imgHTML;
    } else {
        imagesEl.innerHTML = '';
    }

    // ── ETL Images ──
    const etlEl = document.getElementById('section-etl');
    if (data.etl_images && data.etl_images.length) {
        let etlHTML = `<h3>🔬 ETL & Feature Analysis</h3><div class="image-gallery">`;
        data.etl_images.forEach(img => {
            const url = `/api/experiments/${data.run_id}/images/data/2_etl/${img}`;
            const label = img.replace('.png', '').replace(/_/g, ' ');
            etlHTML += `<div class="image-card">
                <img src="${url}" alt="${label}" loading="lazy" onclick="openLightbox('${url}')">
                <div class="image-label">${label}</div>
            </div>`;
        });
        etlHTML += '</div>';
        etlEl.innerHTML = etlHTML;
    } else {
        etlEl.innerHTML = '';
    }
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.tab-btn[data-tab="${tabName}"]`)?.classList.add('active');
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    document.getElementById(`tab-${tabName}`)?.classList.add('active');
}

// ─── Lightbox ────────────────────────────────────────────────────
function openLightbox(url) {
    const lb = document.createElement('div');
    lb.className = 'lightbox';
    lb.innerHTML = `<img src="${url}" alt="Preview">`;
    lb.onclick = () => lb.remove();
    document.body.appendChild(lb);
}

// ─── Run Experiment Modal ────────────────────────────────────────
function setupModal() {
    const overlay = document.getElementById('modal-overlay');
    const btnRun = document.getElementById('btn-run-experiment');
    const btnClose = document.getElementById('btn-close-modal');
    const btnCloseX = document.getElementById('modal-close');
    const btnStart = document.getElementById('btn-start-run');

    btnRun.onclick = () => { overlay.style.display = 'flex'; resetModal(); };
    btnClose.onclick = () => { overlay.style.display = 'none'; };
    btnCloseX.onclick = () => { overlay.style.display = 'none'; };
    overlay.onclick = e => { if (e.target === overlay) overlay.style.display = 'none'; };

    btnStart.onclick = startExperiment;
}

function resetModal() {
    document.getElementById('run-status-area').innerHTML = '<p>Click "Start" to begin a new experiment with the current config.</p>';
    document.getElementById('run-output').style.display = 'none';
    document.getElementById('run-output').textContent = '';
    document.getElementById('btn-start-run').disabled = false;
    document.getElementById('btn-start-run').textContent = '▶ Start';
}

async function startExperiment() {
    const btnStart = document.getElementById('btn-start-run');
    const statusArea = document.getElementById('run-status-area');
    const outputEl = document.getElementById('run-output');

    btnStart.disabled = true;
    btnStart.textContent = '⏳ Running...';
    statusArea.innerHTML = '<p style="color:var(--warning)">⏳ Experiment is running...</p>';
    outputEl.style.display = 'block';
    outputEl.textContent = 'Starting experiment...\n';

    try {
        const res = await fetch('/api/run', { method: 'POST' });
        const data = await res.json();

        if (data.error) {
            statusArea.innerHTML = `<p style="color:var(--danger)">❌ ${data.error}</p>`;
            btnStart.disabled = false;
            btnStart.textContent = '▶ Start';
            return;
        }

        const runName = data.run_name;
        // Poll for status
        const pollId = setInterval(async () => {
            try {
                const sRes = await fetch(`/api/run/status/${runName}`);
                const sData = await sRes.json();
                outputEl.textContent = sData.output || '';
                outputEl.scrollTop = outputEl.scrollHeight;

                if (sData.status !== 'running') {
                    clearInterval(pollId);
                    const isOk = sData.status === 'completed';
                    statusArea.innerHTML = `<p style="color:var(${isOk ? '--success' : '--danger'})">
                        ${isOk ? '✅' : '❌'} Experiment ${sData.status}!</p>`;
                    btnStart.disabled = false;
                    btnStart.textContent = '▶ Start';
                    if (isOk) {
                        showToast('Experiment completed!', 'success');
                        loadExperiments();
                    }
                }
            } catch (e) { /* ignore poll errors */ }
        }, 2000);
    } catch (e) {
        statusArea.innerHTML = `<p style="color:var(--danger)">❌ Failed to start experiment</p>`;
        btnStart.disabled = false;
        btnStart.textContent = '▶ Start';
    }
}

// ─── Helpers ─────────────────────────────────────────────────────
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function formatTimestamp(ts) {
    if (!ts || ts.length < 15) return ts || '';
    // Format: 20260516_223103
    return `${ts.slice(0,4)}-${ts.slice(4,6)}-${ts.slice(6,8)} ${ts.slice(9,11)}:${ts.slice(11,13)}:${ts.slice(13,15)}`;
}
