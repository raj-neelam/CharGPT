// app.js — CharGPT Web Trainer main orchestrator
// Handles UI pages, training loop, inference with context highlighting

'use strict';

/* ── State ─────────────────────────────────────────────────────────────────── */
const state = {
    rawText: '',
    trainingText: '',
    tokenizer: null,
    encodedData: null,
    model: null,
    trainer: null,
    chart: null,
    archViz: null,
    isTraining: false,
    isPaused: false,
    modelReady: false,
    inferRunning: false,
    inferStop: false,
    inferGenerated: '',
    epochHistory: [],   // [{epoch, loss}]
};

/* ── Config Getters ────────────────────────────────────────────────────────── */
function getConfig() {
    return {
        contextLength: parseInt(document.getElementById('ctx-select').value),
        embDim: parseInt(document.getElementById('emb-select').value),
        numLayers: parseInt(document.getElementById('sl-layers').value),
        numHeads: parseInt(document.getElementById('sl-heads').value),
        dropoutRate: parseFloat(document.getElementById('sl-dropout').value),
        ffMul: parseInt(document.getElementById('sl-ffmul').value),
        epochs: parseInt(document.getElementById('sl-epochs').value),
        batchSize: parseInt(document.getElementById('sl-batch').value),
        lrStart: parseFloat(document.getElementById('sl-lr-start')?.value ?? 0.003),
        lrEnd: parseFloat(document.getElementById('sl-lr-end')?.value ?? 0.0005),
        temperature: parseFloat(document.getElementById('sl-temp').value),
        genLen: parseInt(document.getElementById('sl-genlen').value),
        inferGenLen: parseInt(document.getElementById('sl-infer-genlen').value),
    };
}

/* ── Complexity Score ──────────────────────────────────────────────────────── */
function computeComplexity(cfg) {
    // Model cost: quadratic in context (O(n²) attention)
    const { numLayers, embDim, contextLength, ffMul, batchSize } = cfg;
    const attn = numLayers * contextLength * contextLength * embDim;
    const ffn = numLayers * contextLength * embDim * embDim * ffMul;
    const raw = attn + ffn;
    const maxRaw = 8 * 512 * 512 * 256 + 8 * 512 * 256 * 256 * 4;
    let score = Math.min(80, Math.round(Math.sqrt(raw / maxRaw) * 80));
    // Batch penalty
    score += Math.min(10, Math.round((batchSize / 64) * 10));
    // Dataset size penalty: larger datasets = longer training
    const dataLen = state.trainingText ? state.trainingText.length : 0;
    score += Math.min(10, Math.round((dataLen / 500_000) * 10));
    return Math.min(100, score);
}

function updateComplexity() {
    const cfg = getConfig();
    const score = computeComplexity(cfg);
    const fill = document.getElementById('complexity-fill');
    const label = document.getElementById('complexity-label');
    const scoreEl = document.getElementById('complexity-score');

    // Param count
    const vocab = state.tokenizer ? state.tokenizer.vocabSize : 70;
    const params = 2 * vocab * cfg.embDim
        + cfg.numLayers * (3 * cfg.embDim * cfg.embDim + cfg.embDim * cfg.embDim
            + 2 * cfg.embDim * cfg.embDim * cfg.ffMul)
        + vocab * cfg.embDim;

    if (fill) {
        fill.style.width = score + '%';
        if (score < 25) fill.style.background = '#34c98e';
        else if (score < 50) fill.style.background = '#f0b429';
        else if (score < 75) fill.style.background = '#f47c3c';
        else fill.style.background = '#ef4444';
    }
    if (label) {
        const tags = ['✅ Lightweight', '⚡ Moderate', '⚠️ Heavy', '🔴 Very Heavy'];
        const idx = score < 25 ? 0 : score < 50 ? 1 : score < 75 ? 2 : 3;
        label.textContent = `${tags[idx]} — ${score}%`;
        label.style.color = score < 25 ? '#34c98e' : score < 50 ? '#f0b429' : score < 75 ? '#f47c3c' : '#ef4444';
    }
    if (scoreEl) scoreEl.textContent = `~${(params / 1000).toFixed(0)}K params`;

    const paramStr = `${params.toLocaleString()} parameters`;
    const pcd = document.getElementById('param-count-display');
    const pci = document.getElementById('param-count-inline');
    if (pcd) pcd.textContent = paramStr;
    if (pci) pci.textContent = paramStr;

    // Update arch viz
    if (state.archViz) {
        state.archViz.update({ ...cfg, vocabSize: vocab });
    }
}

/* ── Page Navigation ───────────────────────────────────────────────────────── */
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    const page = document.getElementById('page-' + pageId);
    const tab = document.querySelector(`.nav-tab[data-page="${pageId}"]`);
    if (page) page.classList.add('active');
    if (tab) tab.classList.add('active');
}

/* ── Theme ─────────────────────────────────────────────────────────────────── */
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('charGptTheme', theme);
    // Swap SVG icon (sun shown in dark mode, moon shown in light mode)
    const icon = document.getElementById('theme-icon');
    if (icon) {
        const use = icon.querySelector('use');
        if (use) use.setAttribute('href', theme === 'light' ? '#ic-moon' : '#ic-sun');
    }
    // Re-render arch on theme change (colors differ)
    if (state.archViz) state.archViz.render();
}
function toggleTheme() {
    const cur = document.documentElement.getAttribute('data-theme');
    setTheme(cur === 'light' ? 'dark' : 'light');
}

/* ── Toast ─────────────────────────────────────────────────────────────────── */
function toast(msg, type = 'info', duration = 3500) {
    const area = document.getElementById('toast-area');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    area.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    setTimeout(() => {
        el.classList.remove('show');
        setTimeout(() => el.remove(), 350);
    }, duration);
}

/* ── Model Badge ───────────────────────────────────────────────────────────── */
function setModelBadge(status, text) {
    const badge = document.getElementById('model-badge');
    if (!badge) return;
    badge.className = `model-badge ${status}`;
    badge.lastElementChild.textContent = text;
}

/* ── Data Loading ──────────────────────────────────────────────────────────── */
async function loadHarryPotter() {
    const stat = document.getElementById('data-load-status');
    if (stat) stat.textContent = 'Loading...';
    try {
        const res = await fetch('./data/harrypotter.txt');
        if (!res.ok) throw new Error('File not found');
        const fullText = await res.text();
        // Load first 1% of file
        const sliced = fullText.slice(0, Math.floor(fullText.length * 0.01));
        state.rawText = sliced;
        applyDataToTextarea(sliced);
        applyDataRange();
        toast(`Loaded ${(sliced.length / 1000).toFixed(1)}K chars (1% of Harry Potter)`, 'success');
        console.log(`[Data] HP loaded: ${sliced.length} / ${fullText.length} chars (1%)`);
        if (stat) stat.textContent = `${(sliced.length / 1000).toFixed(0)}K chars (1% of file)`;
    } catch (e) {
        if (stat) stat.textContent = 'harrypotter.txt not found';
        toast('harrypotter.txt not found — upload a file or paste text.', 'warn');
        console.warn('[Data] harrypotter.txt not found');
    }
}

function applyDataToTextarea(text) {
    const ta = document.getElementById('training-textarea');
    if (ta) {
        ta.value = text;
        updateDataStats();
    }
    state.rawText = text;
}

function updateDataStats() {
    const ta = document.getElementById('training-textarea');
    if (!ta) return;
    const len = ta.value.length;
    const el = document.getElementById('data-char-count');
    if (el) {
        el.textContent = `${(len / 1000).toFixed(1)}K chars`;
        el.className = 'data-stat' + (len < 5000 ? ' warning' : '');
    }
    const warnEl = document.getElementById('data-size-warn');
    if (warnEl) {
        warnEl.textContent = len < 5000 ? 'Very small dataset' : len > 500000 ? 'Large dataset — may be slow' : '';;
    }
    state.trainingText = ta.value;
}

/* ── Data Range Selection ──────────────────────────────────────────────────── */
function applyDataRange() {
    const rawText = state.rawText;
    if (!rawText) return;
    const startEl = document.getElementById('sl-data-start');
    const endEl = document.getElementById('sl-data-end');
    if (!startEl || !endEl) return;
    let startPct = parseFloat(startEl.value);
    let endPct = parseFloat(endEl.value);
    if (startPct >= endPct) { endPct = Math.min(100, startPct + 5); endEl.value = endPct; }
    document.getElementById('sl-data-start-val').textContent = `${startPct}%`;
    document.getElementById('sl-data-end-val').textContent = `${endPct}%`;
    const s = Math.floor(rawText.length * startPct / 100);
    const e = Math.floor(rawText.length * endPct / 100);
    const selectedText = rawText.slice(s, e);
    state.trainingText = selectedText;
    // Update textarea with full text but highlight selection
    const ta = document.getElementById('training-textarea');
    if (ta) ta.value = selectedText;
    updateDataStats();
}

async function handleFileUpload(file) {
    if (!file) return;
    console.log(`[Data] Uploading: ${file.name} (${file.type})`);
    toast(`Loading ${file.name}...`, 'info', 1500);

    const ext = file.name.split('.').pop().toLowerCase();

    if (ext === 'pdf') {
        await readPdfFile(file);
    } else if (ext === 'json') {
        const text = await file.text();
        try {
            const json = JSON.parse(text);
            const extracted = extractTextFromJSON(json);
            applyDataToTextarea(extracted);
            toast('JSON text extracted', 'success');
        } catch { applyDataToTextarea(text); toast('JSON loaded as raw text', 'info'); }
    } else if (ext === 'csv') {
        const text = await file.text();
        const lines = text.split('\n').map(l => l.replace(/"/g, '').split(',').join(' ')).join('\n');
        applyDataToTextarea(lines);
        toast('CSV text extracted', 'success');
    } else {
        // .txt and anything else
        const text = await file.text();
        applyDataToTextarea(text);
        toast(`Loaded ${(text.length / 1000).toFixed(1)}K chars`, 'success');
    }
}

function extractTextFromJSON(obj, depth = 0) {
    if (depth > 6) return '';
    if (typeof obj === 'string') return obj;
    if (typeof obj === 'number') return String(obj);
    if (Array.isArray(obj)) return obj.map(v => extractTextFromJSON(v, depth + 1)).join('\n');
    if (typeof obj === 'object' && obj !== null) {
        return Object.values(obj).map(v => extractTextFromJSON(v, depth + 1)).join('\n');
    }
    return '';
}

async function readPdfFile(file) {
    if (!window.pdfjsLib) {
        toast('PDF.js not loaded – paste text manually', 'warn');
        return;
    }
    try {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        let allText = '';
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            allText += content.items.map(s => s.str).join(' ') + '\n';
        }
        applyDataToTextarea(allText.trim());
        toast(`PDF: extracted ${pdf.numPages} pages`, 'success');
    } catch (e) {
        toast('PDF read failed — paste text manually', 'error');
        console.error('PDF error:', e);
    }
}

/* ── Build Model ───────────────────────────────────────────────────────────── */
async function buildModel() {
    const ta = document.getElementById('training-textarea');
    const text = ta ? ta.value.trim() : '';
    if (!text || text.length < 100) {
        toast('Need at least 100 characters of training data!', 'error');
        return false;
    }

    state.trainingText = text;

    console.log(`🏗️ [Build] Building tokenizer...`);
    setModelBadge('training', 'Building...');

    state.tokenizer = new CharTokenizer();
    state.tokenizer.build(text);
    state.encodedData = state.tokenizer.encode(text);
    const tokCount = state.encodedData.length;
    toast(`Tokenizer built — vocab: ${state.tokenizer.vocabSize} chars, ${tokCount} tokens`, 'info');
    console.log(`[Build] Vocab: ${state.tokenizer.vocabSize}, tokens: ${tokCount}`);

    const cfg = getConfig();

    // Validate heads vs embDim
    if (cfg.embDim % cfg.numHeads !== 0) {
        toast(`embDim (${cfg.embDim}) must be divisible by numHeads (${cfg.numHeads})`, 'error');
        setModelBadge('', 'Error');
        return false;
    }

    if (state.model) {
        try { state.model.dispose(); } catch (e) { }
        state.model = null;
    }

    try {
        state.model = buildTransformerModel({
            vocabSize: state.tokenizer.vocabSize,
            embDim: cfg.embDim,
            contextLength: cfg.contextLength,
            numLayers: cfg.numLayers,
            numHeads: cfg.numHeads,
            dropoutRate: cfg.dropoutRate,
            ffMul: cfg.ffMul,
        });
    } catch (e) {
        toast(`Build failed: ${e.message}`, 'error');
        console.error('Build error:', e);
        setModelBadge('', 'Error');
        return false;
    }

    state.modelReady = true;
    const paramStr = state.model.countParams().toLocaleString();
    toast(`Model built — ${paramStr} parameters`, 'success');
    updateArchViz();
    return true;
}

/* ── Arch Viz ───────────────────────────────────────────────────────────────── */
function updateArchViz() {
    if (!state.archViz) return;
    const cfg = getConfig();
    state.archViz.update({
        ...cfg,
        vocabSize: state.tokenizer ? state.tokenizer.vocabSize : 70,
    });
}

/* ── Training ──────────────────────────────────────────────────────────────── */
async function startTraining(further = false) {
    if (state.isTraining) return;
    if (!state.modelReady || !further) {
        const ok = await buildModel();
        if (!ok) return;
    }

    const cfg = getConfig();
    state.isTraining = true;
    state.isPaused = false;

    showPage('train');
    setModelBadge('training', 'Training');
    updateTrainButtons();

    // Init chart
    if (!state.chart) {
        state.chart = new LossChart('loss-canvas');
    }

    if (!further) {
        state.epochHistory = [];
        clearSampleCards();
        state.chart.reset();
        // 'Before Training' sample is generated via onUntrainedSample callback inside trainer
    }

    if (!state.trainer) state.trainer = new Trainer();

    const epochMax = cfg.epochs;
    let currentEpoch = state.epochHistory.length;
    const startEpoch = currentEpoch;
    let batchStep = 0;

    await state.trainer.train(state.model, state.encodedData, {
        contextLength: cfg.contextLength,
        batchSize: cfg.batchSize,
        epochs: cfg.epochs,
        lrStart: cfg.lrStart,
        lrEnd: cfg.lrEnd,
        vocabSize: state.tokenizer.vocabSize,

        onUntrainedSample: further ? null : async () => {
            await addSampleCard(0, null, 'Before Training');
        },

        onEpochStart: (epoch, lr) => {
            // epoch is 0-based; use epoch/epochMax so epoch 0 = 0%, last = (N-1)/N ~ 100%
            updateProgressBar('epoch-progress-fill', 'epoch-pct',
                (epoch / epochMax) * 100,
                `Epoch ${epoch + 1} / ${startEpoch + epochMax}`);
            updateStatRow('stat-lr', lr.toExponential(3));
        },

        onBatchEnd: (bi, total, wma, lr, rawLoss) => {
            // Use bi+1 so last batch reaches 100%
            updateProgressBar('batch-progress-fill', 'batch-pct', ((bi + 1) / total) * 100, null);
            batchStep++;
            state.chart.addPoint(batchStep, wma, rawLoss, lr);
            const wmaEl = document.getElementById('stat-wma');
            if (wmaEl) wmaEl.textContent = wma.toFixed(4);
        },

        onEpochEnd: async (epoch, testLoss) => {
            currentEpoch = epoch + 1;
            state.epochHistory.push({ epoch: epoch + 1, loss: testLoss });
            updateEpochTable();
            updateStatRow('stat-test', testLoss.toFixed(4));
            state.chart.addTestPoint(batchStep, testLoss);
            state.chart.addEpochMarker(batchStep);
            await addSampleCard(epoch + 1, testLoss, '');
        },

        onTrainingEnd: () => {
            state.isTraining = false;
            setModelBadge('ready', 'Trained');
            updateTrainButtons();
            updateProgressBar('epoch-progress-fill', 'epoch-pct', 100, 'Complete');
            updateProgressBar('batch-progress-fill', 'batch-pct', 100, null);
            toast('Training complete!', 'success');
        },
    });

    if (state.isTraining) {
        // Stopped early
        state.isTraining = false;
        setModelBadge('paused', 'Paused');
        updateTrainButtons();
        toast('Training stopped', 'warn');
    }
}

function stopTraining() {
    if (state.trainer) state.trainer.stop();
    state.isTraining = false;
    state.isPaused = false;
    setModelBadge('paused', 'Paused');
    updateTrainButtons();
}

function updateTrainButtons() {
    const start = document.getElementById('btn-train');
    const further = document.getElementById('btn-further');
    const stop = document.getElementById('btn-stop');

    if (start) start.disabled = state.isTraining;
    if (further) further.disabled = state.isTraining || !state.modelReady;
    if (stop) stop.disabled = !state.isTraining;
    if (further) further.style.display = state.modelReady ? '' : 'none';
}

/* ── Progress bars ──────────────────────────────────────────────────────────── */
function updateProgressBar(fillId, pctId, pct, label) {
    const fill = document.getElementById(fillId);
    const pctEl = document.getElementById(pctId);
    const v = Math.min(100, Math.max(0, pct));
    if (fill) fill.style.width = v + '%';
    if (pctEl && label !== null) pctEl.textContent = label || `${Math.round(v)}%`;
    else if (pctEl) pctEl.textContent = `${Math.round(v)}%`;
}

function updateStatRow(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function updateEpochTable() {
    const tbody = document.getElementById('epoch-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';
    [...state.epochHistory].reverse().forEach(({ epoch, loss }) => {
        const tr = document.createElement('tr');
        const isGood = epoch > 1 && loss < state.epochHistory[epoch - 2]?.loss;
        tr.innerHTML = `<td>${epoch}</td><td class="loss-val ${isGood ? 'loss-good' : ''}">${loss.toFixed(4)}</td>`;
        tbody.appendChild(tr);
    });
}

/* ── Sample Cards ───────────────────────────────────────────────────────────── */
function clearSampleCards() {
    const area = document.getElementById('sample-cards-area');
    if (!area) return;
    area.innerHTML = `<div class="empty-state">
      <div class="empty-icon"><svg style="width:32px;height:32px;opacity:.3" viewBox="0 0 20 20" fill="none"><use href="#ic-brain" stroke="currentColor"/></svg></div>
      <div>Press <strong>Build &amp; Train</strong> to start. Samples appear after each epoch.</div>
    </div>`;
}

async function addSampleCard(epoch, testLoss, subtitle) {
    const area = document.getElementById('sample-cards-area');
    if (!area) return;

    // Remove empty state
    const empty = area.querySelector('.empty-state');
    if (empty) empty.remove();

    // Create card (streaming)
    const card = document.createElement('div');
    card.className = 'sample-card';

    let lossColor = '#8b92b4';
    let lossStr = '—';
    if (testLoss !== null) {
        lossStr = testLoss.toFixed(4);
        lossColor = testLoss < 2 ? '#34c98e' : testLoss < 3 ? '#f0b429' : '#ef4444';
    }

    const epochLabel = epoch === 0 ? 'Untrained' : epoch < 0 ? 'Before Training' : `Epoch ${epoch}`;
    const now = new Date().toLocaleTimeString();

    card.innerHTML = `
      <div class="sample-card-hdr">
        <span class="sc-epoch">${epochLabel}</span>
        ${testLoss !== null ? `<span class="sc-loss" style="color:${lossColor};border-color:${lossColor}22">loss ${lossStr}</span>` : ''}
        <span class="sc-time">${now}</span>
      </div>
      <div class="sample-card-body" id="sc-body-${epoch}"></div>
    `;
    area.prepend(card);

    // Stream generated text
    if (!state.model || !state.tokenizer) {
        const body = card.querySelector('.sample-card-body');
        if (body) body.textContent = '(model not available)';
        return;
    }

    const cfg = getConfig();
    const seed = pickRandomSeed(state.trainingText, Math.min(12, cfg.contextLength));
    const body = card.querySelector('.sample-card-body');
    if (body) body.textContent = '';

    await generateText(
        state.model, state.tokenizer, seed,
        cfg.genLen, cfg.temperature, cfg.contextLength,
        (ch) => { if (body) body.textContent += ch; },
        () => false
    );
}

function pickRandomSeed(text, len) {
    if (!text || text.length < len + 5) return '';
    const start = Math.floor(Math.random() * (text.length - len - 1));
    return text.slice(start, start + len);
}

/* ── Inference Panel ────────────────────────────────────────────────────────── */
function setupInferencPanel() {
    const ta = document.getElementById('infer-input');
    const sendBtn = document.getElementById('infer-send');
    const stopBtn = document.getElementById('infer-stop');
    const moreBtn = document.getElementById('infer-more');
    const clearBtn = document.getElementById('infer-clear');
    const display = document.getElementById('infer-display');

    if (!ta || !sendBtn) return;

    ta.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runInference(); }
    });
    sendBtn.addEventListener('click', runInference);
    if (stopBtn) stopBtn.addEventListener('click', () => { state.inferStop = true; });
    if (moreBtn) moreBtn.addEventListener('click', runInferenceFurther);
    if (clearBtn) clearBtn.addEventListener('click', () => {
        ta.value = '';
        if (display) display.innerHTML = '';
        state.inferGenerated = '';
        if (moreBtn) moreBtn.disabled = true;
    });
}

function updateContextHighlight(prompt, generated, display, isGenerating = false) {
    if (!display) return;
    const cfg = getConfig();
    const ctxLen = cfg.contextLength;
    const glowCls = 'text-in-context' + (isGenerating ? ' generating' : '');

    if (!generated) {
        // No output yet — show blinking cursor while waiting for first token
        display.innerHTML = isGenerating ? `<span class="${glowCls}"> ▌</span>` : '';
        display.scrollTop = display.scrollHeight;
        return;
    }

    // Split: older text (outside context window) + glowing tail (inside context window)
    const contextTailLen = Math.max(0, ctxLen - prompt.length);
    const splitAt = Math.max(0, generated.length - contextTailLen);
    const older = generated.slice(0, splitAt);
    const inCtx = generated.slice(splitAt);

    display.innerHTML =
        (older ? `<span class="text-generated">${escHtml(older)}</span>` : '') +
        (inCtx ? `<span class="${glowCls}">${escHtml(inCtx)}</span>` : '');
    display.scrollTop = display.scrollHeight;
}

function setInferGenerating(display, on) {
    if (!display) return;
    display.querySelectorAll('.text-in-context').forEach(s =>
        s.classList.toggle('generating', on)
    );
}

function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '↵\n');
}

async function runInference() {
    if (!state.model || !state.tokenizer) {
        toast('Train the model first!', 'warn');
        return;
    }
    if (state.inferRunning) return;
    if (state.isTraining) {
        toast('Training in progress — wait a batch or stop first', 'warn');
        return;
    }

    const ta = document.getElementById('infer-input');
    const display = document.getElementById('infer-display');
    if (!ta) return;

    const prompt = ta.value;
    const cfg = getConfig();
    state.inferRunning = true;
    state.inferStop = false;
    state.inferGenerated = '';   // reset each new inference

    const sendBtn = document.getElementById('infer-send');
    const stopBtn = document.getElementById('infer-stop');
    if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.innerHTML = `<svg class="ic" width="13" height="13"><use href="#ic-stop"/></svg>`;
    }
    if (stopBtn) stopBtn.disabled = false;

    let generated = '';
    updateContextHighlight(prompt, generated, display, false);
    // Start glow on context
    setInferGenerating(display, true);

    await generateText(
        state.model, state.tokenizer, prompt,
        cfg.inferGenLen, cfg.temperature, cfg.contextLength,
        (ch) => {
            generated += ch;
            updateContextHighlight(prompt, generated, display, true);
        },
        () => state.inferStop,
        false // Do not echo startText
    );

    // Stop glow
    setInferGenerating(display, false);
    state.inferRunning = false;
    state.inferGenerated = generated;   // save for Continue
    const moreBtn = document.getElementById('infer-more');
    if (sendBtn) {
        sendBtn.disabled = false;
        sendBtn.innerHTML = `<svg class="ic" width="13" height="13"><use href="#ic-play"/></svg> Generate`;
    }
    if (stopBtn) stopBtn.disabled = true;
    if (moreBtn) moreBtn.disabled = false;
}

async function runInferenceFurther() {
    if (!state.model || !state.tokenizer) return;
    if (state.inferRunning || state.isTraining) return;
    const ta = document.getElementById('infer-input');
    const display = document.getElementById('infer-display');
    if (!ta) return;

    // Full context fed to the model = original prompt + all previously generated text
    const combined = ta.value + (state.inferGenerated || '');
    const cfg = getConfig();
    const ctxLen = cfg.contextLength;

    state.inferRunning = true;
    state.inferStop = false;

    const sendBtn = document.getElementById('infer-send');
    const stopBtn = document.getElementById('infer-stop');
    const moreBtn = document.getElementById('infer-more');
    if (sendBtn) sendBtn.disabled = true;
    if (moreBtn) { moreBtn.disabled = true; moreBtn.textContent = '…'; }
    if (stopBtn) stopBtn.disabled = false;

    // Helper: re-render display with context window glow
    // Shows only state.inferGenerated, split at contextLength from the end
    const renderWithContextGlow = (isGenerating) => {
        const full = state.inferGenerated || '';
        // The context window is the last ctxLen characters (from the combined prompt+generated perspective)
        // What the user sees in the display IS the generated text only.
        // We glow the tail of displayed text that falls within the context window minus the prompt.
        const promptLen = ta.value.length;
        const contextTailInGenerated = Math.max(0, ctxLen - promptLen);
        const splitAt = Math.max(0, full.length - contextTailInGenerated);
        const older = full.slice(0, splitAt);
        const inCtx = full.slice(splitAt);
        const glowClass = 'text-in-context' + (isGenerating ? ' generating' : '');
        display.innerHTML =
            (older ? `<span class="text-generated">${escHtml(older)}</span>` : '') +
            (inCtx ? `<span class="${glowClass}">${escHtml(inCtx)}</span>` : '');
        display.scrollTop = display.scrollHeight;
    };

    // Initial render — glow the existing tail before we start
    renderWithContextGlow(true);

    await generateText(
        state.model, state.tokenizer, combined,
        cfg.inferGenLen, cfg.temperature, ctxLen,
        (ch) => {
            state.inferGenerated = (state.inferGenerated || '') + ch;
            renderWithContextGlow(true);
        },
        () => state.inferStop,
        false // Do not echo startText
    );

    // Final render — remove generating glow
    renderWithContextGlow(false);
    state.inferRunning = false;
    if (sendBtn) { sendBtn.disabled = false; }
    if (stopBtn) stopBtn.disabled = true;
    if (moreBtn) {
        moreBtn.disabled = false;
        moreBtn.innerHTML = `<svg class="ic" width="13" height="13"><use href="#ic-play"/></svg> More`;
    }
}



/* ── Slider + Select Wiring ─────────────────────────────────────────────────── */
function wireSlidersAndSelects() {
    const bindings = [
        ['sl-layers', 'sl-layers-val', v => `${v} layers`],
        ['sl-heads', 'sl-heads-val', v => `${v} heads`],
        ['sl-dropout', 'sl-dropout-val', v => v],
        ['sl-ffmul', 'sl-ffmul-val', v => `×${v}`],
        ['sl-epochs', 'sl-epochs-val', v => `${v} epochs`],
        ['sl-batch', 'sl-batch-val', v => v],
        ['sl-temp', 'sl-temp-val', v => v],
        ['sl-genlen', 'sl-genlen-val', v => `${v} chars`],
        ['sl-infer-genlen', 'sl-infer-genlen-val', v => `${v} chars`],
    ];

    bindings.forEach(([id, valId, fmt]) => {
        const el = document.getElementById(id);
        const val = document.getElementById(valId);
        const val2 = document.getElementById(valId.replace('-val', '-val2'));
        if (!el) return;
        const update = () => {
            const txt = fmt(el.value);
            if (val) val.textContent = txt;
            if (val2) val2.textContent = txt;
            updateComplexity();
        };
        el.addEventListener('input', update);
        update();
    });

    // Selects
    ['ctx-select', 'emb-select'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', () => {
            updateComplexity();
            validateHeads();
        });
    });

    // Validate heads when heads or embDim changes
    document.getElementById('sl-heads').addEventListener('input', validateHeads);
}

function validateHeads() {
    const embDim = parseInt(document.getElementById('emb-select').value);
    const heads = parseInt(document.getElementById('sl-heads').value);
    const warn = document.getElementById('heads-warn');
    if (embDim % heads !== 0 && warn) {
        warn.textContent = `⚠️ ${embDim} ÷ ${heads} = not integer`;
        warn.style.color = 'var(--c-orange)';
    } else if (warn) {
        warn.textContent = `head dim = ${embDim / heads}`;
        warn.style.color = 'var(--c-dim)';
    }
}

/* ── Collapsible Config Sections ────────────────────────────────────────────── */
function wireCollapsibles() {
    document.querySelectorAll('.config-section-header').forEach(hdr => {
        hdr.addEventListener('click', () => {
            hdr.closest('.config-section').classList.toggle('open');
        });
    });
}

function setTrainParamsOpen(open) {
    const body = document.getElementById('train-params-body');
    const chev = document.getElementById('train-params-chevron');
    if (!body) return;
    body.style.display = open ? '' : 'none';
    if (chev) chev.style.transform = open ? '' : 'rotate(-90deg)';
}

/* ── Init ───────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 [App] CharacterGPT Trainer starting');

    // Restore theme
    const savedTheme = localStorage.getItem('charGptTheme') || 'dark';
    setTheme(savedTheme);

    // Nav tabs
    document.querySelectorAll('.nav-tab').forEach(btn => {
        btn.addEventListener('click', () => showPage(btn.dataset.page));
    });
    document.getElementById('theme-btn').addEventListener('click', toggleTheme);

    // Architecture viz
    state.archViz = new ArchitectureVisualizer('arch-diagram');

    // Wire up sliders
    wireSlidersAndSelects();
    wireCollapsibles();

    // Train params toggle
    document.getElementById('train-params-toggle')?.addEventListener('click', () => {
        const body = document.getElementById('train-params-body');
        if (body) {
            const isOpen = body.style.display !== 'none';
            setTrainParamsOpen(!isOpen);
        }
    });

    // Open first section by default
    document.querySelectorAll('.config-section')[0]?.classList.add('open');
    document.querySelectorAll('.config-section')[1]?.classList.add('open');

    // Data loading
    document.getElementById('btn-load-hp').addEventListener('click', async () => {
        await loadHarryPotter();
        const rangeRow = document.getElementById('data-range-row');
        if (rangeRow) rangeRow.style.display = '';
    });
    document.getElementById('file-input').addEventListener('change', async e => {
        if (e.target.files[0]) {
            await handleFileUpload(e.target.files[0]);
            const rangeRow = document.getElementById('data-range-row');
            if (rangeRow) rangeRow.style.display = '';
        }
    });
    document.getElementById('training-textarea').addEventListener('input', updateDataStats);

    // Data range sliders
    const wireRange = (id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => { applyDataRange(); });
    };
    wireRange('sl-data-start');
    wireRange('sl-data-end');

    // LR sliders — end LR must stay ≤ start LR
    const wireLR = (id, valId) => {
        const el = document.getElementById(id);
        const val = document.getElementById(valId);
        if (el && val) {
            el.addEventListener('input', () => {
                const v = parseFloat(el.value);
                // Clamp LR end to not exceed LR start
                if (id === 'sl-lr-end') {
                    const startEl = document.getElementById('sl-lr-start');
                    if (startEl && v > parseFloat(startEl.value)) {
                        el.value = startEl.value;
                    }
                }
                const clamped = parseFloat(el.value);
                val.textContent = clamped < 0.0001 ? clamped.toExponential(2) : clamped.toFixed(4);
            });
        }
    };
    wireLR('sl-lr-start', 'sl-lr-start-val');
    wireLR('sl-lr-end', 'sl-lr-end-val');

    // Drag-and-drop on textarea
    const ta = document.getElementById('training-textarea');
    ta.addEventListener('dragover', e => { e.preventDefault(); ta.style.borderColor = 'var(--c-accent)'; });
    ta.addEventListener('dragleave', () => { ta.style.borderColor = ''; });
    ta.addEventListener('drop', e => {
        e.preventDefault();
        ta.style.borderColor = '';
        const file = e.dataTransfer.files[0];
        if (file) handleFileUpload(file);
    });

    // Training page nav proceed button
    document.getElementById('btn-to-train').addEventListener('click', () => showPage('train'));

    // Train buttons — auto-collapse train params when training starts
    document.getElementById('btn-train').addEventListener('click', () => { setTrainParamsOpen(false); startTraining(false); });
    document.getElementById('btn-further').addEventListener('click', () => { setTrainParamsOpen(false); startTraining(true); });
    document.getElementById('btn-stop').addEventListener('click', stopTraining);
    // Stop also works from configure page
    document.getElementById('btn-to-configure').addEventListener('click', () => showPage('configure'));

    // Inference
    setupInferencPanel();

    // Initial arch
    updateComplexity();
    updateArchViz();
    updateTrainButtons();
    clearSampleCards();

    console.log('[App] Ready — no data loaded by default. Click "Load Harry Potter" or upload a file.');
});

window.appState = state;
