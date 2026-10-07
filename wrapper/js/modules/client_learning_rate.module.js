/**
 * client_learning_rate.module.js — App mixin for handling complex client learning rate UI states.
 *
 * This module extends the main App class prototype, providing methods to synchronize
 * the dynamic input grid for manual/random learning rates based on global configurations.
 */

window.AppClientLearningRateModule = {
    /**
     * Mirrors the currently effective client learning rate into the hidden config field.
     * @returns {number} Effective scalar learning rate.
     */
    _syncLearningRateBackingField() {
        const hiddenInput = document.getElementById('cfg-learning_rate');
        const modeEl = document.getElementById('cfg-client_lr_mode');
        const uniformInput = document.getElementById('cfg-client_lr_uniform');

        const fallback = Number.isFinite(parseFloat(hiddenInput?.value || ''))
            ? parseFloat(hiddenInput.value)
            : 0.001;
        const mode = String(modeEl?.value || 'uniform');

        let effective = Number.isFinite(parseFloat(uniformInput?.value || ''))
            ? parseFloat(uniformInput.value)
            : fallback;

        if (mode === 'manual') {
            const firstManual = parseFloat(document.getElementById('cfg-client_lr_manual_0')?.value || '');
            if (Number.isFinite(firstManual)) effective = firstManual;
        } else if (mode === 'random') {
            const firstMin = parseFloat(document.getElementById('cfg-client_lr_random_min_0')?.value || '');
            const firstMax = parseFloat(document.getElementById('cfg-client_lr_random_max_0')?.value || '');
            if (Number.isFinite(firstMin) && Number.isFinite(firstMax)) {
                effective = Math.min(firstMin, firstMax);
            } else if (Number.isFinite(firstMin)) {
                effective = firstMin;
            } else if (Number.isFinite(firstMax)) {
                effective = firstMax;
            }
        }

        if (!Number.isFinite(effective) || effective <= 0) {
            effective = fallback > 0 ? fallback : 0.001;
        }
        if (hiddenInput) hiddenInput.value = String(effective);
        return effective;
    },

    /**
     * Synchronizes the DOM input elements for client learning rates (Uniform, Manual, Random modes)
     * according to the currently selected mode and the total number of federated clients.
     * Rebuilds dynamic input grids and restores previous parameter values when toggling between settings.
     * @returns {void}
     */
    _syncClientLearningRateInputs() {
        const modeEl = document.getElementById('cfg-client_lr_mode');
        const nEl = document.getElementById('cfg-num_clients');
        if (!modeEl || !nEl) return;

        const mode = String(modeEl.value || 'uniform');
        const nClients = Math.max(1, parseInt(nEl.value || '1', 10) || 1);

        const mobileModeWrap = document.getElementById('client-lr-mode-mobile');
        if (mobileModeWrap) {
            if (!mobileModeWrap.dataset.bound) {
                mobileModeWrap.addEventListener('click', (e) => {
                    const btn = e.target?.closest?.('[data-lr-mode]');
                    if (!btn) return;
                    const next = String(btn.getAttribute('data-lr-mode') || 'uniform');
                    if (modeEl.value === next) return;
                    modeEl.value = next;
                    this._syncClientLearningRateInputs();
                });
                mobileModeWrap.dataset.bound = '1';
            }

            mobileModeWrap.querySelectorAll('[data-lr-mode]').forEach((btn) => {
                const btnMode = String(btn.getAttribute('data-lr-mode') || '');
                btn.classList.toggle('lr-mode-mobile-btn-active', btnMode === mode);
                btn.setAttribute('aria-pressed', btnMode === mode ? 'true' : 'false');
            });
        }

        const uniformWrap = document.getElementById('client-lr-uniform-wrap');
        const manualWrap = document.getElementById('client-lr-manual-wrap');
        const randomWrap = document.getElementById('client-lr-random-wrap');
        if (uniformWrap) uniformWrap.classList.toggle('hidden', mode !== 'uniform');
        if (manualWrap) manualWrap.classList.toggle('hidden', mode !== 'manual');
        if (randomWrap) randomWrap.classList.toggle('hidden', mode !== 'random');

        const uniformInput = document.getElementById('cfg-client_lr_uniform');
        const globalLr = parseFloat(document.getElementById('cfg-learning_rate')?.value || '0.001');
        const fallbackLr = Number.isFinite(globalLr) ? globalLr : 0.001;
        
        // Auto-seed uniform block if currently empty
        if (uniformInput && (uniformInput.value === '' || !Number.isFinite(parseFloat(uniformInput.value)))) {
            uniformInput.value = String(fallbackLr);
        }

        const manualGrid = document.getElementById('client-lr-manual-grid');
        if (manualGrid) {
            const prev = {};
            manualGrid.querySelectorAll('input[id^="cfg-client_lr_manual_"]').forEach((inp) => {
                prev[inp.id] = inp.value;
            });
            const items = [];
            for (let i = 0; i < nClients; i += 1) {
                const id = `cfg-client_lr_manual_${i}`;
                const value = prev[id] !== undefined ? prev[id] : String(fallbackLr);
                items.push(`
                    <div class="lr-client-card">
                        <label class="lr-client-card-label" for="${id}">Client ${i + 1}</label>
                        <input id="${id}" type="number" step="any" class="fl-input lr-compact-input lr-card-input" value="${this._escapeHtml(value)}">
                    </div>
                `);
            }
            manualGrid.innerHTML = items.join('');
        }

        const randomGrid = document.getElementById('client-lr-random-grid');
        if (randomGrid) {
            const prevMin = {};
            const prevMax = {};
            randomGrid.querySelectorAll('input[id^="cfg-client_lr_random_min_"]').forEach((inp) => {
                prevMin[inp.id] = inp.value;
            });
            randomGrid.querySelectorAll('input[id^="cfg-client_lr_random_max_"]').forEach((inp) => {
                prevMax[inp.id] = inp.value;
            });

            const rows = [];
            for (let i = 0; i < nClients; i += 1) {
                const minId = `cfg-client_lr_random_min_${i}`;
                const maxId = `cfg-client_lr_random_max_${i}`;
                const minVal = prevMin[minId] !== undefined ? prevMin[minId] : String(fallbackLr * 0.5);
                const maxVal = prevMax[maxId] !== undefined ? prevMax[maxId] : String(fallbackLr * 1.5);
                rows.push(`
                    <div class="lr-client-card lr-client-card-random">
                        <div class="lr-client-card-label">Client ${i + 1}</div>
                        <div class="lr-random-minmax-wrap">
                            <div class="lr-random-field">
                                <label class="block text-[10px] text-slate-500 mb-0.5">Min</label>
                                <input id="${minId}" type="number" step="any" class="fl-input lr-compact-input lr-card-input" value="${this._escapeHtml(minVal)}">
                            </div>
                            <div class="lr-random-field">
                                <label class="block text-[10px] text-slate-500 mb-0.5">Max</label>
                                <input id="${maxId}" type="number" step="any" class="fl-input lr-compact-input lr-card-input" value="${this._escapeHtml(maxVal)}">
                            </div>
                        </div>
                    </div>
                `);
            }
            randomGrid.innerHTML = rows.join('');
        }

        this._syncLearningRateBackingField();
    },

    /**
     * Hydrates the client learning rate setup UI blocks automatically using imported/saved configuration values.
     * Distinguishes arrays into specific mode buckets.
     * @param {Record<string, *>} cfg - Target configuration parameters dictionary properties.
     * @returns {void}
     */
    _hydrateClientLearningRateFromConfig(cfg) {
        const modeEl = document.getElementById('cfg-client_lr_mode');
        const uniformInput = document.getElementById('cfg-client_lr_uniform');
        if (!modeEl || !uniformInput) return;

        const raw = cfg?.client_learning_rates;
        let arr = [];
        if (Array.isArray(raw)) {
            arr = raw.map(v => parseFloat(v)).filter(v => Number.isFinite(v));
        } else if (typeof raw === 'string') {
            arr = raw.split(',').map(s => parseFloat(s.trim())).filter(v => Number.isFinite(v));
        }

        const nClients = Math.max(1, parseInt(document.getElementById('cfg-num_clients')?.value || '1', 10) || 1);
        
        if (arr.length <= 1) {
            modeEl.value = 'uniform';
            if (arr.length === 1) uniformInput.value = String(arr[0]);
            this._syncClientLearningRateInputs();
            return;
        }

        modeEl.value = 'manual';
        this._syncClientLearningRateInputs();
        for (let i = 0; i < nClients; i += 1) {
            const inp = document.getElementById(`cfg-client_lr_manual_${i}`);
            if (!inp) continue;
            inp.value = String(arr[Math.min(i, arr.length - 1)]);
        }
    },
};
