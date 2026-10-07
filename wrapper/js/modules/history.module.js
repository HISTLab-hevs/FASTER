window.AppHistoryModule = {
    /**
     * Wrapper for retrieving run configs and metrics ensuring momentary latency tolerance.
     * @param {string} runId Unique string matching the run's name footprint.
     * @param {number} [retries=2] Permissible successive re-attempts.
     * @param {number} [delayMs=220] Yielded timeout between failures.
     * @returns {Promise<{runId: string, metrics: Record<string, *>, cfg: Record<string, *>}>} Acquired combined run data.
     */
    async _fetchRunWithRetry(runId, retries = 2, delayMs = 220) {
        let lastErr = null;
        for (let i = 0; i <= retries; i++) {
            try {
                const [metrics, cfg] = await Promise.all([
                    api.getMetrics(runId),
                    api.getRunConfig(runId),
                ]);
                return { runId, metrics: metrics || {}, cfg: cfg || {} };
            } catch (e) {
                lastErr = e;
                if (i < retries) await new Promise(res => setTimeout(res, delayMs));
            }
        }
        throw lastErr;
    },

    /**
     * Checks whether a metrics payload contains a non-empty test split.
     * @param {Record<string, *>} metrics Metrics payload.
     * @returns {boolean} True when test metrics are available.
     */
    _historyHasTestMetrics(metrics) {
        const testAcc = metrics?.aggregated_test?.accuracy;
        const testLoss = metrics?.aggregated_test?.loss;
        return Array.isArray(testAcc) && testAcc.length > 0 && Array.isArray(testLoss) && testLoss.length > 0;
    },

    /**
     * Returns the metric definitions rendered in summary/detail views.
     * @returns {Array<{key: string, label: string, digits: number, direction: 'max'|'min', naText?: string, help?: string}>}
     */
    _historyMetricSpecs() {
        return [
            { key: 'accuracy', label: 'Accuracy', digits: 3, direction: 'max' },
            { key: 'loss', label: 'Loss', digits: 4, direction: 'min' },
            { key: 'precision', label: 'Precision', digits: 3, direction: 'max' },
            { key: 'recall', label: 'Recall', digits: 3, direction: 'max' },
            { key: 'f1', label: 'F1', digits: 3, direction: 'max' },
            {
                key: 'roc_auc',
                label: 'ROC AUC',
                digits: 3,
                direction: 'max',
                naText: 'N/A',
                help: 'ROC AUC may be unavailable when the evaluated split does not contain the class structure needed to compute it.',
            },
        ];
    },

    /**
     * Resolves a split bucket from a metrics payload.
     * @param {Record<string, *>} metrics Metrics payload.
     * @param {'validation'|'test'} scope Target split.
     * @returns {Record<string, Array<number>>} Split metrics bucket.
     */
    _historyMetricBucket(metrics, scope) {
        return scope === 'test'
            ? (metrics?.aggregated_test || {})
            : (metrics?.aggregated_val || {});
    },

    /**
     * Computes the best available scalar from a metric series.
     * @param {Array<number>} values Metric series.
     * @param {'max'|'min'} direction Optimizer direction.
     * @returns {number|null} Best scalar or null.
     */
    _historyBestSeriesValue(values, direction = 'max') {
        const nums = (values || [])
            .map(v => Number(v))
            .filter(v => Number.isFinite(v));
        if (!nums.length) return null;
        return direction === 'min' ? Math.min(...nums) : Math.max(...nums);
    },

    /**
     * Returns the final scalar from a metric series.
     * @param {Array<number>} values Metric series.
     * @returns {number|null} Final scalar or null.
     */
    _historyFinalSeriesValue(values) {
        if (!Array.isArray(values) || !values.length) return null;
        return values[values.length - 1];
    },

    /**
     * Formats one explicit metric value for detail cards and comparison tables.
     * @param {number|null|undefined} value Metric value.
     * @param {Record<string, *>} spec Metric spec.
     * @returns {string} Formatted display string.
     */
    _historyFormatMetricValue(value, spec) {
        if (value == null || Number.isNaN(Number(value)) || !Number.isFinite(Number(value))) {
            return spec?.naText || '-';
        }
        return this._formatMetric(value, spec?.digits ?? 4);
    },

    /**
     * Builds a split-aware label with an optional help tooltip.
     * @param {string} scopeLabel Split label such as Validation or Test.
     * @param {Record<string, *>} spec Metric spec.
     * @returns {string} Label HTML.
     */
    _historyMetricLabel(metricPrefix, scopeLabel, spec) {
        const effectivePrefix = metricPrefix === 'Best' && spec?.direction === 'min' ? 'Lowest' : metricPrefix;
        const text = `${effectivePrefix} ${scopeLabel} ${spec.label}`;
        const safeText = this._escapeHtml(text);
        if (!spec.help) return safeText;
        const safeHelp = this._escapeHtml(spec.help);
        return `<span class="label-with-help">${safeText}<span class="help-dot" tabindex="0" role="img" aria-label="${safeHelp}" data-tooltip="${safeHelp}">?</span></span>`;
    },

    /**
     * Returns the compact header label used in history comparison tables.
     * @param {Record<string, *>} spec Metric spec.
     * @returns {string} Header label HTML.
     */
    _historyComparisonMetricLabel(spec) {
        const labels = {
            accuracy: 'Best Accuracy',
            loss: 'Lowest Loss',
            precision: 'Best Precision',
            recall: 'Best Recall',
            f1: 'Best F1',
            roc_auc: 'Best ROC AUC',
        };
        const text = labels[spec?.key] || spec?.label || '-';
        const safeText = this._escapeHtml(text);
        if (!spec?.help) return safeText;
        const safeHelp = this._escapeHtml(spec.help);
        return `<span class="label-with-help">${safeText}<span class="help-dot" tabindex="0" role="img" aria-label="${safeHelp}" data-tooltip="${safeHelp}">?</span></span>`;
    },

    /**
     * Renders a detail/summary metric card.
     * @param {string} scopeLabel Split label such as Validation or Test.
     * @param {Record<string, *>} spec Metric spec.
     * @param {number|null|undefined} value Metric value.
     * @returns {string} Metric card HTML.
     */
    _historyMetricCard(metricPrefix, scopeLabel, spec, value) {
        return `<div class="ui-stat-card detail-insight-card"><div class="detail-insight-label">${this._historyMetricLabel(metricPrefix, scopeLabel, spec)}</div><div class="detail-insight-value">${this._historyFormatMetricValue(value, spec)}</div></div>`;
    },

    /**
     * Produces a shared explanatory note for final/best metric panels.
     * @param {string} metricPrefix Metric semantic prefix.
     * @returns {string} Short explanatory note.
     */
    _historyMetricNote(metricPrefix) {
        if (metricPrefix === 'Final') {
            return 'Final metric cards show the last recorded round value. The charts below still show the full trajectory.';
        }
        return 'Best metric cards show the strongest value observed over each series, or the lowest value for loss-style metrics. The charts below still show trajectories, while comparison views add mean lines and standard deviation bands.';
    },

    /**
     * Produces a table title for final/best summary tables.
     * @param {string} metricPrefix Metric semantic prefix.
     * @param {string} scopeLabel Split label such as Validation or Test.
     * @returns {string} Section title.
     */
    _historyMetricSectionTitle(metricPrefix, scopeLabel) {
        return `${metricPrefix} ${scopeLabel} Metrics`;
    },

    /**
     * Returns a column layout block for history summary tables.
     * @param {'repeat'|'compare'} kind Table variant.
     * @returns {string} Colgroup markup.
     */
    _historyMetricTableColgroup(kind = 'repeat') {
        if (kind === 'compare') {
            return `<colgroup>
                <col class="hist-summary-run-col">
                <col class="hist-summary-mode-col">
                <col class="hist-summary-metric-col">
                <col class="hist-summary-metric-col">
                <col class="hist-summary-metric-col">
                <col class="hist-summary-metric-col">
                <col class="hist-summary-metric-col">
                <col class="hist-summary-metric-col">
            </colgroup>`;
        }
        return `<colgroup>
            <col class="hist-summary-repeat-col">
            <col class="hist-summary-metric-col">
            <col class="hist-summary-metric-col">
            <col class="hist-summary-metric-col">
            <col class="hist-summary-metric-col">
            <col class="hist-summary-metric-col">
            <col class="hist-summary-metric-col">
        </colgroup>`;
    },

    /**
     * Builds a table for one split-aware summary section.
     * @param {string} title Summary section title.
     * @param {string} scopeLabel Split label such as Validation or Test.
     * @param {Array<{metrics: Record<string, *>}>} items Run payloads.
     * @param {'validation'|'test'} scope Target split.
     * @returns {string} HTML fragment.
     */
    _historyMetricSectionTable(title, scopeLabel, items, scope) {
        const specs = this._historyMetricSpecs();
        const rows = items.map((item, idx) => {
            const bucket = this._historyMetricBucket(item.metrics, scope);
            const cells = specs.map((spec) => {
                const raw = this._historyBestSeriesValue(bucket?.[spec.key], spec.direction);
                return `<td class="py-1 px-2 text-sm">${this._historyFormatMetricValue(raw, spec)}</td>`;
            }).join('');
            return `<tr class="border-t border-slate-700/40">
                <td class="py-1 px-2 text-xs font-mono text-slate-400">Repeat ${idx + 1}</td>
                ${cells}
            </tr>`;
        }).join('');

        const headers = specs.map((spec) => `<th class="py-1 px-2">${this._historyComparisonMetricLabel(spec)}</th>`).join('');
        return `<p class="font-semibold text-xs text-slate-400 mb-2 uppercase tracking-wide">${this._escapeHtml(title)}</p>
            <div class="hist-summary-table-wrap">
            <table class="w-full text-left table-fixed hist-summary-table">
                ${this._historyMetricTableColgroup('repeat')}
                <thead class="text-xs text-slate-500 uppercase">
                    <tr>
                        <th class="py-1 px-2">Repeat</th>
                        ${headers}
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
            </div>`;
    },

    /**
     * Resolves the displayed KPI scope for one selection set.
     * @param {Array<{metrics: Record<string, *>, cfg: Record<string, *>}>} series Loaded run payloads.
     * @returns {{scope: 'validation'|'test', scopeLabel: string, note: string, showTest: boolean}}
     */
    _resolveHistoryMetricScope(series) {
        const items = Array.isArray(series) ? series : [];
        const hasSingleRun = items.length === 1;
        const allHaveTest = items.length > 0 && items.every(({ metrics, cfg }) => (
            String(cfg?.evaluation_split_mode || 'train_val_test') !== 'train_val'
            && this._historyHasTestMetrics(metrics)
        ));
        const scope = allHaveTest ? 'test' : 'validation';
        const scopeLabel = scope === 'test' ? 'Test' : 'Validation';
        const showTest = allHaveTest;

        if (hasSingleRun) {
            const mode = String(items[0]?.cfg?.evaluation_split_mode || 'train_val_test');
            if (scope === 'test') {
                return {
                    scope,
                    scopeLabel,
                    showTest,
                    note: 'Displayed metrics: Test. This run includes test results, so summary cards use best-over-series test values.',
                };
            }
            return {
                scope,
                scopeLabel,
                showTest,
                note: mode === 'train_val'
                    ? 'Displayed metrics: Validation. This run uses train_val, so test metrics are not available. Summary cards use best-over-series validation values.'
                    : 'Displayed metrics: Validation. Test metrics are unavailable for this run. Summary cards use best-over-series validation values.',
            };
        }

        return {
            scope,
            scopeLabel,
            showTest,
            note: scope === 'test'
                ? 'Displayed metrics: Test. Every selected run includes test results, so summary cards use best-over-series test values.'
                : 'Displayed metrics: Validation. At least one selected run does not include test results, so summary cards use best-over-series validation values.',
        };
    },

    /**
     * Loads detailed data for one or more requested runs and prepares the UI.
     * @param {Array<string>|null} [selectedKeys=null] Targets to load. Uses global selection if omitted.
     * @returns {Promise<void>}
     */
    async _loadHistoryRun(selectedKeys = null) {
        const selected = (selectedKeys || this._state.historySelectedRuns || []).slice();
        if (selected.length < 2) { return; }
        if (selected.length > 5) { this._showToast('You can compare up to 5 runs', 'warn'); return; }

        const hasRunning = selected.some(key => this._runByKey(key)?.status === 'running');
        if (hasRunning) {
            this._showToast('Running runs cannot be loaded in history', 'warn');
            this._state.historySelectedRuns = selected.filter(key => this._runByKey(key)?.status !== 'running');
            this._populateRunsTable(this._sortedRuns());
            this._updateSelectedRunsHint();
            this._refreshComparisonAreaFromSelection();
            return;
        }

        const loadSeq = ++this._state.histLoadSeq;

        const statusEl = document.getElementById('hist-status');
        if (statusEl) { statusEl.textContent = `Loading ${selected.length} run(s)…`; statusEl.classList.remove('hidden'); }

        try {
            const series = await Promise.all(selected.map(async key => {
                const run = this._runByKey(key);
                const runId = run?.id || run?.name || key;
                const displayName = run?.name || runId;
                const fetched = await this._fetchRunWithRetry(runId);
                return { ...fetched, displayName };
            }));

            if (loadSeq !== this._state.histLoadSeq) return;

            const modes = series.map(s => (s.cfg?.evaluation_split_mode || 'train_val_test'));
            const mixedModes = new Set(modes).size > 1;
            const scopeInfo = this._resolveHistoryMetricScope(series);
            const splitWarn = document.getElementById('hist-split-warning');
            if (splitWarn) {
                splitWarn.textContent = scopeInfo.note + (mixedModes ? ' Mixed evaluation modes were detected.' : '');
                splitWarn.className = `hidden text-xs mb-3 px-3 py-2 rounded border ${scopeInfo.scope === 'test'
                    ? 'border-emerald-300 bg-emerald-50 text-emerald-800'
                    : 'border-amber-300 bg-amber-50 text-amber-800'}`;
                splitWarn.classList.remove('hidden');
            }
            document.getElementById('hist-test-acc-card')?.classList.toggle('hidden', !scopeInfo.showTest);
            document.getElementById('hist-test-loss-card')?.classList.toggle('hidden', !scopeInfo.showTest);

            if (!this._state.histChartsInit) {
                metricsCharts.initHistory();
                this._state.histChartsInit = true;
            }
            if (series.length === 1) {
                metricsCharts.updateHistory(series[0].metrics);
            } else {
                const map = {};
                series.forEach(s => { map[s.displayName || s.runId] = s.metrics; });
                metricsCharts.updateHistoryComparison(map);
            }

            const summaryEl = document.getElementById('hist-summary');
            if (summaryEl) {
                if (series.length === 1) {
                    summaryEl.innerHTML = this._buildSummaryTable(series[0].metrics, series[0].displayName || series[0].runId, {
                        mode: modes[0],
                    });
                } else {
                    summaryEl.innerHTML = this._buildComparisonSummary(series, {
                        showTest: scopeInfo.showTest,
                    });
                }
                summaryEl.classList.remove('hidden');
            }

            document.getElementById('hist-charts')?.classList.remove('hidden');

            const dlRow = document.getElementById('hist-download-row');
            const dlBtn = document.getElementById('hist-download-btn');
            if (dlRow && dlBtn) {
                if (series.length === 1) {
                    api.bindExportLink(dlBtn, series[0].runId, (e) => {
                        const msg = e.response?.data?.detail || e.message || 'Failed to download run archive';
                        this._showToast(msg, 'error');
                    });
                    dlRow.classList.remove('hidden');
                } else {
                    dlRow.classList.add('hidden');
                }
            }

            if (statusEl) statusEl.textContent = series.length === 1
                ? `Loaded: ${series[0].displayName || series[0].runId}`
                : `Compared: ${series.map(s => s.displayName || s.runId).join(', ')}`;
            this._state.histRunId = series[0].runId;
        } catch {
            if (loadSeq !== this._state.histLoadSeq) return;
            if (statusEl) statusEl.textContent = 'Failed to load run.';
            this._showToast('Failed to load run', 'error');
        }
    },

    /**
     * Safely destroys any existing detail Chart.js instances to avoid memory leaks.
     * @returns {void}
     */
    _destroyDetailCharts() {
        const charts = this._state.histDetailCharts || {};
        Object.keys(charts).forEach(k => {
            try { charts[k]?.destroy(); } catch {}
        });
        this._state.histDetailCharts = {};
    },

    /**
     * Closes the run detail modal overlay and cleans up bound charts.
     * @returns {void}
     */
    _closeRunDetailModal() {
        this._destroyDetailCharts();
        this._closeRunSystemInfoModal();
        document.getElementById('hist-run-detail-modal')?.classList.add('hidden');
    },

    /**
     * Opens inline editing for the run name.
     * User can edit directly on screen; Enter or blur confirms, Escape cancels.
     * @returns {void}
     */
    _openRenameRunDialog() {
        const nameSpan = document.getElementById('hist-run-detail-name');
        const currentDisplayName = nameSpan?.textContent?.trim() || '';
        const runId = String(this._state.runSystemInfoRunId || this._state.histRunId || '').trim();
        if (!currentDisplayName || !runId) {
            this._showToast('Could not determine run ID', 'error');
            return;
        }

        const MAX_NAME_LENGTH = 20;
        const input = document.createElement('input');
        input.type = 'text';
        input.value = currentDisplayName;
        input.style.cssText = `
            padding: 4px 8px;
            border: 2px solid #3b82f6;
            border-radius: 4px;
            font-weight: 500;
            font-size: inherit;
            font-family: inherit;
            background: white;
            color: #1f2937;
            outline: none;
        `;
        input.maxLength = MAX_NAME_LENGTH;
        input.setAttribute('title', `Press Enter to save, Escape to cancel (max ${MAX_NAME_LENGTH} chars)`);

        nameSpan.replaceWith(input);
        input.focus();
        input.select();

        let finalized = false;
        const confirmEdit = async () => {
            if (finalized) return;
            finalized = true;
            const trimmed = String(input.value || '').trim();

            // Validation
            if (!trimmed) {
                this._showToast('Run name cannot be empty', 'warn');
                input.replaceWith(nameSpan);
                return;
            }
            if (trimmed.length > MAX_NAME_LENGTH) {
                this._showToast(`Name too long (max ${MAX_NAME_LENGTH} chars, got ${trimmed.length})`, 'error');
                input.replaceWith(nameSpan);
                return;
            }
            if (!/^[A-Za-z0-9_.\- ]+$/.test(trimmed)) {
                this._showToast('Invalid characters (allowed: letters, numbers, _, -, ., space)', 'error');
                input.replaceWith(nameSpan);
                return;
            }
            if (trimmed === currentDisplayName) {
                input.replaceWith(nameSpan);
                return;
            }

            // API call
            try {
                await api.renameRun(runId, trimmed);
                nameSpan.textContent = trimmed;
                input.replaceWith(nameSpan);
                this._showToast(`Run renamed to: ${trimmed}`, 'success');
                this._refreshRunsList(true, { silent: true });
            } catch (e) {
                const errorMsg = e.response?.data?.detail || e.message || 'Failed to rename run';
                this._showToast(errorMsg, 'error');
                input.replaceWith(nameSpan);
            }
        };

        const cancelEdit = () => {
            if (finalized) return;
            finalized = true;
            input.replaceWith(nameSpan);
        };

        input.addEventListener('blur', confirmEdit);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                input.blur();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                cancelEdit();
            }
        });
    },

    /**
     * Calculates the mathematical average of an array of numbers, filtering invalid ones.
     * @param {Array<number>} values Array of numerics to reduce.
     * @returns {number|null} The computed subset mean, or null if unprocessable.
     */
    _meanFinite(values) {
        const clean = (values || []).map(v => Number(v)).filter(v => Number.isFinite(v));
        if (!clean.length) return null;
        return clean.reduce((s, v) => s + v, 0) / clean.length;
    },

    /**
     * Normalizes complex hardware usage metrics obtained from a run's runtime snapshot logs.
     * @param {Record<string, *>} metrics The core metrics object possessing the payload.
     * @returns {Record<string, *>} Structurally normalized usage averages and snapshots.
     */
    _normalizeResourceUsage(metrics) {
        const usage = metrics?.resource_usage && typeof metrics.resource_usage === 'object'
            ? { ...metrics.resource_usage }
            : {};

        const samples = Array.isArray(usage.samples) ? usage.samples : [];
        if (samples.length) {
            const cpu = [];
            const ramPct = [];
            const ramUsed = [];
            const ramTotal = [];
            const gpuLoad = [];
            const gpuTemp = [];
            const vramUsed = [];
            const vramTotal = [];
            const vramPct = [];
            samples.forEach(s => {
                const c = Number(s?.cpu_percent);
                const rp = Number(s?.ram_percent);
                const ru = Number(s?.ram_used_gb);
                const rt = Number(s?.ram_total_gb);
                if (Number.isFinite(c)) cpu.push(c);
                if (Number.isFinite(rp)) ramPct.push(rp);
                if (Number.isFinite(ru)) ramUsed.push(ru);
                if (Number.isFinite(rt)) ramTotal.push(rt);

                const gpus = Array.isArray(s?.gpus) ? s.gpus : [];
                if (gpus.length) {
                    const perLoad = gpus.map(g => Number(g?.load)).filter(Number.isFinite);
                    const perTemp = gpus.map(g => Number(g?.temperature)).filter(Number.isFinite);
                    const perUsed = gpus.map(g => Number(g?.memory_used_mb)).filter(Number.isFinite);
                    const perTotal = gpus.map(g => Number(g?.memory_total_mb)).filter(Number.isFinite);
                    if (perLoad.length) gpuLoad.push(perLoad.reduce((a, b) => a + b, 0) / perLoad.length);
                    if (perTemp.length) gpuTemp.push(perTemp.reduce((a, b) => a + b, 0) / perTemp.length);
                    if (perUsed.length && perTotal.length) {
                        const usedGb = perUsed.reduce((a, b) => a + b, 0) / 1024;
                        const totalGb = perTotal.reduce((a, b) => a + b, 0) / 1024;
                        if (Number.isFinite(usedGb) && Number.isFinite(totalGb) && totalGb > 0) {
                            vramUsed.push(usedGb);
                            vramTotal.push(totalGb);
                            vramPct.push((usedGb / totalGb) * 100);
                        }
                    }
                }
            });

            usage.cpu_percent_avg = this._meanFinite(cpu);
            usage.ram_percent_avg = this._meanFinite(ramPct);
            usage.ram_used_gb_avg = this._meanFinite(ramUsed);
            usage.ram_total_gb_avg = this._meanFinite(ramTotal);
            usage.gpu_load_avg = this._meanFinite(gpuLoad);
            usage.gpu_temp_avg = this._meanFinite(gpuTemp);
            usage.vram_used_gb_avg = this._meanFinite(vramUsed);
            usage.vram_total_gb_avg = this._meanFinite(vramTotal);
            usage.vram_usage_percent_avg = this._meanFinite(vramPct);
            usage.samples_count = usage.samples_count || samples.length;
        }

        return usage;
    },

    /**
     * Closes the detailed system hardware usage diagnostics modal overlay.
     * @returns {void}
     */
    _closeRunSystemInfoModal() {
        document.getElementById('hist-run-system-modal')?.classList.add('hidden');
    },

    /**
     * Renders and opens the specific hardware system resources tracking modal for a loaded run.
     * @returns {void}
     */
    _openRunSystemInfoModal() {
        const runId = this._state.runSystemInfoRunId || this._state.histRunId || '-';
        const info = this._state.runSystemInfo || {};
        const title = document.getElementById('hist-run-system-title');
        const meta = document.getElementById('hist-run-system-meta');
        const grid = document.getElementById('hist-run-system-grid');
        const empty = document.getElementById('hist-run-system-empty');
        if (!grid || !empty) return;

        if (title) title.textContent = `Run System Info — ${runId}`;
        if (meta) {
            const samples = Number(info.samples_count);
            const duration = Number(info.duration_s);
            const samplesTxt = Number.isFinite(samples) && samples > 0 ? `${samples} samples` : 'No samples';
            const durationTxt = Number.isFinite(duration) ? `${Math.round(duration)}s window` : 'Duration n/a';
            const hasGpu = Number.isFinite(Number(info.gpu_load_avg))
                || Number.isFinite(Number(info.gpu_temp_avg))
                || Number.isFinite(Number(info.vram_used_gb_avg))
                || Number.isFinite(Number(info.vram_total_gb_avg));
            const gpuNote = (!hasGpu && Number.isFinite(samples) && samples > 0)
                ? ' | No GPU samples recorded for this run'
                : '';
            meta.textContent = `${samplesTxt} | ${durationTxt}${gpuNote}`;
        }

        const withUnit = (value, digits, unit) => {
            const base = this._formatMetric(value, digits);
            return base === '-' ? '-' : `${base}${unit}`;
        };

        const cards = [
            ['CPU Avg', withUnit(info.cpu_percent_avg, 1, '%')],
            ['RAM Avg', withUnit(info.ram_percent_avg, 1, '%')],
            ['RAM Used Avg', withUnit(info.ram_used_gb_avg, 2, ' GB')],
            ['RAM Total Avg', withUnit(info.ram_total_gb_avg, 2, ' GB')],
            ['GPU Load Avg', withUnit(info.gpu_load_avg, 1, '%')],
            ['GPU Temp Avg', withUnit(info.gpu_temp_avg, 1, ' C')],
            ['VRAM Used Avg', withUnit(info.vram_used_gb_avg, 2, ' GB')],
            ['VRAM Total Avg', withUnit(info.vram_total_gb_avg, 2, ' GB')],
            ['VRAM Usage Avg', withUnit(info.vram_usage_percent_avg, 1, '%')],
        ];

        const available = cards.filter(([, value]) => value !== '-');
        if (!available.length) {
            grid.innerHTML = '';
            empty.classList.remove('hidden');
        } else {
            empty.classList.add('hidden');
            grid.innerHTML = cards.map(([k, v]) =>
                `<div class="ui-stat-card detail-insight-card"><div class="detail-insight-label">${k}</div><div class="detail-insight-value">${v}</div></div>`
            ).join('');
        }

        document.getElementById('hist-run-system-modal')?.classList.remove('hidden');
    },

    /**
     * Drives full process rendering of individual run analytics details upon selection.
     * @param {string} runKey The identifier of the run details to inspect.
     * @returns {Promise<void>}
     */
    async _openRunDetail(runKey) {
        const run = this._runByKey(runKey);
        if (!run) return;
        const runId = run.id || run.name;
        const runDisplayName = run.name || runId;
        const seq = (this._state.runDetailLoadSeq || 0) + 1;
        this._state.runDetailLoadSeq = seq;
        this._initCodeEditors?.();
        this._setEditorValue('runCfg', 'Loading config...');
        this._setEditorValue('runModel', 'Loading model code...');
        try {
            const { metrics, cfg } = await this._fetchRunWithRetry(runId);
            if (seq !== this._state.runDetailLoadSeq) return;
            const nameSpan = document.getElementById('hist-run-detail-name');
            if (nameSpan) {
                nameSpan.textContent = runDisplayName;
                nameSpan.title = 'Click to rename run';
            }
            const meta = document.getElementById('hist-run-detail-meta');
            if (meta) {
                const datasetLabel = (cfg.dataset_name === 'custom_csv' || cfg.dataset_name === 'custom_image_npz' || cfg.dataset_name === 'custom_image_folder')
                    ? (cfg.custom_dataset_name || run.dataset || cfg.dataset_name)
                    : (cfg.dataset_name || run.dataset || '-');
                meta.textContent = `Method: ${cfg.method || run.method || '-'} | Dataset: ${datasetLabel} | Eval: ${cfg.evaluation_split_mode || run.evaluation_split_mode || '-'} | Model: ${cfg.custom_model_name || run.model_name || 'DefaultNet'}`;
            }
            const dlBtn = document.getElementById('hist-run-detail-download-btn');
            if (dlBtn) {
                api.bindExportLink(dlBtn, runId, (e) => {
                    const msg = e.response?.data?.detail || e.message || 'Failed to download run archive';
                    this._showToast(msg, 'error');
                });
            }
            const cfgPre = document.getElementById('hist-run-detail-config');
            const cfgText = Object.keys(cfg || {}).length
                ? Object.entries(cfg || {}).map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n')
                : 'No config found for this run.';
            if (!this._setEditorValue('runCfg', cfgText) && cfgPre) cfgPre.textContent = cfgText;
            const codePre = document.getElementById('hist-run-detail-model-code');
            const codeText = (cfg.custom_model_code && cfg.custom_model_code.trim()) ? cfg.custom_model_code : 'Default model in use';
            if (!this._setEditorValue('runModel', codeText) && codePre) codePre.textContent = codeText;

            this._destroyDetailCharts();
            const valBucket = metrics?.aggregated_val || {};
            const testBucket = metrics?.aggregated_test || {};
            const specs = this._historyMetricSpecs();
            const valAcc = Array.isArray(valBucket.accuracy) ? valBucket.accuracy : [];
            const valLoss = Array.isArray(valBucket.loss) ? valBucket.loss : [];
            const testAcc = Array.isArray(testBucket.accuracy) ? testBucket.accuracy : [];
            const testLoss = Array.isArray(testBucket.loss) ? testBucket.loss : [];
            const evalMode = cfg?.evaluation_split_mode || run?.evaluation_split_mode || 'train_val_test';
            const showTest = evalMode !== 'train_val' && this._historyHasTestMetrics(metrics);
            const testAccCard = document.getElementById('hist-detail-test-acc-card');
            const testLossCard = document.getElementById('hist-detail-test-loss-card');
            if (testAccCard) testAccCard.classList.toggle('hidden', !showTest);
            if (testLossCard) testLossCard.classList.toggle('hidden', !showTest);
            const resourceUsage = this._normalizeResourceUsage(metrics);
            this._state.runSystemInfoRunId = runId;
            this._state.runSystemInfo = resourceUsage;
            const vramAvg = Number.isFinite(Number(resourceUsage?.vram_used_gb_avg))
                ? Number(resourceUsage.vram_used_gb_avg)
                : null;

            const rounds = Math.max(
                valAcc.length,
                valLoss.length,
                testAcc.length,
                testLoss.length,
                Array.isArray(valBucket.precision) ? valBucket.precision.length : 0,
                Array.isArray(valBucket.recall) ? valBucket.recall.length : 0,
                Array.isArray(valBucket.f1) ? valBucket.f1.length : 0,
                Array.isArray(valBucket.roc_auc) ? valBucket.roc_auc.length : 0,
                Array.isArray(testBucket.precision) ? testBucket.precision.length : 0,
                Array.isArray(testBucket.recall) ? testBucket.recall.length : 0,
                Array.isArray(testBucket.f1) ? testBucket.f1.length : 0,
                Array.isArray(testBucket.roc_auc) ? testBucket.roc_auc.length : 0,
            );
            const insights = document.getElementById('hist-run-detail-insights');
            if (insights) {
                const renderSection = (metricPrefix, scopeLabel, bucket) => {
                    const cards = specs.map((spec) => {
                        const series = Array.isArray(bucket?.[spec.key]) ? bucket[spec.key] : [];
                        const value = metricPrefix === 'Final'
                            ? this._historyFinalSeriesValue(series)
                            : this._historyBestSeriesValue(series, spec.direction);
                        return this._historyMetricCard(metricPrefix, scopeLabel, spec, value);
                    }).join('');
                    return `<div class="col-span-2 xl:col-span-3">
                        <p class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1">${this._historyMetricSectionTitle(metricPrefix, scopeLabel)}</p>
                        <p class="text-xs text-slate-500 mb-2">${this._historyMetricNote(metricPrefix)}</p>
                        <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">${cards}</div>
                    </div>`;
                };

                const repeatsCount = Array.isArray(metrics?.all_metrics) && metrics.all_metrics.length
                    ? metrics.all_metrics.length
                    : 1;
                const sections = [
                    `<div class="col-span-2 xl:col-span-3"><div class="ui-stat-card detail-insight-card"><div class="detail-insight-label">Rounds</div><div class="detail-insight-value">${rounds}</div></div></div>`,
                    `<div class="col-span-2 xl:col-span-3"><div class="ui-stat-card detail-insight-card"><div class="detail-insight-label">Repeats</div><div class="detail-insight-value">${repeatsCount}</div>${repeatsCount > 1 ? '<div class="text-xs text-slate-500 mt-1">Final cards and charts show the mean across repeats.</div>' : ''}</div></div>`,
                    renderSection('Final', 'Validation', valBucket),
                ];
                if (showTest) {
                    sections.push(renderSection('Final', 'Test', testBucket));
                }
                sections.push(`<div class="col-span-2 xl:col-span-3"><div class="ui-stat-card detail-insight-card"><div class="detail-insight-label">Run VRAM Avg</div><div class="detail-insight-value">${Number.isFinite(vramAvg) ? `${Number(vramAvg).toFixed(2)} GB` : '-'}</div></div></div>`);
                insights.innerHTML = sections.join('');
            }

            const labelsAcc = Array.from({ length: valAcc.length }, (_, i) => i + 1);
            const labelsLoss = Array.from({ length: valLoss.length }, (_, i) => i + 1);
            const labelsTAcc = Array.from({ length: testAcc.length }, (_, i) => i + 1);
            const labelsTLoss = Array.from({ length: testLoss.length }, (_, i) => i + 1);

            const accCtx = document.getElementById('hist-detail-val-acc');
            const lossCtx = document.getElementById('hist-detail-val-loss');
            const testAccCtx = document.getElementById('hist-detail-test-acc');
            const testLossCtx = document.getElementById('hist-detail-test-loss');
            if (accCtx) {
                this._state.histDetailCharts.acc = new Chart(accCtx, {
                    type: 'line',
                    data: { labels: labelsAcc, datasets: [{ label: 'Validation Accuracy', data: valAcc, borderColor: '#1d4ed8', tension: 0.3 }] },
                    options: { responsive: true, maintainAspectRatio: false, animation: false },
                });
            }
            if (lossCtx) {
                this._state.histDetailCharts.loss = new Chart(lossCtx, {
                    type: 'line',
                    data: { labels: labelsLoss, datasets: [{ label: 'Validation Loss', data: valLoss, borderColor: '#0f766e', tension: 0.3 }] },
                    options: { responsive: true, maintainAspectRatio: false, animation: false },
                });
            }
            if (testAccCtx && showTest) {
                this._state.histDetailCharts.testAcc = new Chart(testAccCtx, {
                    type: 'line',
                    data: { labels: labelsTAcc, datasets: [{ label: 'Test Accuracy', data: testAcc, borderColor: '#1d4ed8', tension: 0.3 }] },
                    options: { responsive: true, maintainAspectRatio: false, animation: false },
                });
            }
            if (testLossCtx && showTest) {
                this._state.histDetailCharts.testLoss = new Chart(testLossCtx, {
                    type: 'line',
                    data: { labels: labelsTLoss, datasets: [{ label: 'Test Loss', data: testLoss, borderColor: '#0f766e', tension: 0.3 }] },
                    options: { responsive: true, maintainAspectRatio: false, animation: false },
                });
            }

            const modal = document.getElementById('hist-run-detail-modal');
            modal?.classList.remove('hidden');
            requestAnimationFrame(() => {
                this._state.editors?.runCfg?.refresh();
                this._state.editors?.runModel?.refresh();
            });
        } catch (e) {
            this._reportNonBlockingIssue?.('run detail load', e, {
                dedupeMs: 10000,
            });
            if (seq !== this._state.runDetailLoadSeq) return;
            this._showToast('Failed to load run details', 'error');
        }
    },

    /**
     * Fetches and displays isolated model script python source code inside an overlay modal.
     * @param {string} runKey The targeted run identifier.
     * @returns {Promise<void>}
     */
    async _openModelCodeForRun(runKey) {
        const run = this._runByKey(runKey);
        if (!run) return;
        const runId = run.id || run.name;
        this._initCodeEditors?.();
        this._setEditorValue('modelPreview', 'Loading model code...');
        try {
            const cfg = await api.getRunConfig(runId);
            const title = document.getElementById('hist-model-code-title');
            if (title) title.textContent = `Model Code — ${runId} (${cfg.custom_model_name || run.model_name || 'DefaultNet'})`;
            const pre = document.getElementById('hist-model-code-pre');
            const codeText = (cfg.custom_model_code && cfg.custom_model_code.trim()) ? cfg.custom_model_code : 'Default model in use';
            if (!this._setEditorValue('modelPreview', codeText) && pre) pre.textContent = codeText;
            const modal = document.getElementById('hist-model-code-modal');
            modal?.classList.remove('hidden');
            requestAnimationFrame(() => {
                this._state.editors?.modelPreview?.refresh();
            });
        } catch (e) {
            this._reportNonBlockingIssue?.('model code load', e, {
                dedupeMs: 10000,
            });
            this._showToast('Failed to load model code', 'error');
        }
    },

    /**
     * Closes the python model source code viewer interactive modal overlay.
     * @returns {void}
     */
    _closeModelCodeModal() {
        document.getElementById('hist-model-code-modal')?.classList.add('hidden');
    },

    /**
     * Generates HTML markup for a comparison summary table assessing multiple side-by-side runs.
     * @param {Array<{runId: string, metrics: Record<string, *>, cfg: Record<string, *>}>} series Compiled run payloads.
     * @param {Record<string, *>} [opts={}] Tuning parameters for display configurations.
     * @returns {string} Fully baked HTML fragment for placement.
     */
    _buildComparisonSummary(series, opts = {}) {
        const items = Array.isArray(series) ? series : [];
        const includeTest = !!opts.showTest;
        const scopes = [
            { scope: 'validation', scopeLabel: 'Validation' },
            ...(includeTest ? [{ scope: 'test', scopeLabel: 'Test' }] : []),
        ];

        const bestByScope = Object.fromEntries(scopes.map(({ scope }) => {
            const bestByMetric = {};
            this._historyMetricSpecs().forEach((spec) => {
                const rowValues = items.map(({ metrics }) => this._historyBestSeriesValue(
                    this._historyMetricBucket(metrics, scope)?.[spec.key],
                    spec.direction,
                ));
                bestByMetric[spec.key] = this._historyBestSeriesValue(rowValues, spec.direction);
            });
            return [scope, bestByMetric];
        }));

        const renderMetricValue = (value, spec, bestValue) => {
            const formatted = this._historyFormatMetricValue(value, spec);
            if (value == null || bestValue == null) return formatted;
            if (!Number.isFinite(Number(value)) || !Number.isFinite(Number(bestValue))) return formatted;
            if (Math.abs(Number(value) - Number(bestValue)) > 1e-12) return formatted;
            return `<span class="hist-metric-best">${formatted}</span>`;
        };

        const buildRows = (scope) => items.map(({ runId, displayName, metrics, cfg }) => {
            const bucket = this._historyMetricBucket(metrics, scope);
            const mode = cfg?.evaluation_split_mode || 'train_val_test';
            const cells = this._historyMetricSpecs().map((spec) => {
                const raw = this._historyBestSeriesValue(bucket?.[spec.key], spec.direction);
                return `<td class="py-1 px-2 text-sm">${renderMetricValue(raw, spec, bestByScope?.[scope]?.[spec.key])}</td>`;
            }).join('');
            return `<tr class="border-t border-slate-700/40">
                <td class="py-1 px-2 text-xs font-mono text-slate-400">${this._escapeHtml(displayName || runId)}</td>
                <td class="py-1 px-2 text-xs">${this._escapeHtml(mode)}</td>
                ${cells}
            </tr>`;
        }).join('');

        const sections = scopes.map(({ scope, scopeLabel }) => {
            const title = `${this._historyMetricSectionTitle('Best', scopeLabel)} (${items.length} runs)`;
            const rows = buildRows(scope);
            const headers = this._historyMetricSpecs().map((spec) => `<th class="py-1 px-2">${this._historyComparisonMetricLabel(spec)}</th>`).join('');
            return `<p class="font-semibold text-xs text-slate-400 mb-2 uppercase tracking-wide">${this._escapeHtml(title)}</p>
                <div class="hist-summary-table-wrap mb-4">
                <table class="w-full text-left table-fixed hist-summary-table">
                    ${this._historyMetricTableColgroup('compare')}
                    <thead class="text-xs text-slate-500 uppercase">
                        <tr>
                            <th class="py-1 px-2">Run</th>
                            <th class="py-1 px-2">Mode</th>
                            ${headers}
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
                </div>`;
        }).join('');

        return `<div class="text-xs text-slate-500 mb-2">Comparison summary shows best-over-series metric values for each selected run. Validation stays explicit, test metrics appear only when all selected runs provide them, and ROC AUC can display as N/A when the split cannot be evaluated mathematically.</div>
            ${sections}`;
    },

    /**
     * Generates HTML markup for a solitary run summary highlighting repeats explicitly.
     * @param {Record<string, *>} metrics The parent metrics map targeting the summary.
     * @param {string} runId Associated readable name rendering into the DOM.
     * @param {Record<string, boolean|string>} [opts={}] Optional configuration context.
     * @returns {string} Stringified HTML markup wrapping the stats.
     */
    _buildSummaryTable(metrics, runId, opts = {}) {
        const repeats = Array.isArray(metrics?.all_metrics) && metrics.all_metrics.length ? metrics.all_metrics : [metrics];
        const hasTest = this._historyHasTestMetrics(metrics);
        const scopes = [
            { scope: 'validation', scopeLabel: 'Validation' },
            ...(hasTest ? [{ scope: 'test', scopeLabel: 'Test' }] : []),
        ];

        const renderRows = (scope) => {
            const specs = this._historyMetricSpecs();
            const repeatRows = repeats.map((r, i) => {
                const bucket = this._historyMetricBucket(r, scope);
                const cells = specs.map((spec) => {
                    const raw = this._historyBestSeriesValue(bucket?.[spec.key], spec.direction);
                    return `<td class="py-1 px-2 text-sm">${this._historyFormatMetricValue(raw, spec)}</td>`;
                }).join('');
                return `<tr class="border-t border-slate-700/40">
                    <td class="py-1 px-2 text-slate-400 text-xs">Repeat ${i + 1}</td>
                    ${cells}
                </tr>`;
            }).join('');

            // With more than one repeat, append a mean ± std row across repeats
            // so the run has a single aggregated figure per metric (the charts
            // already show the mean trajectory and std band).
            if (repeats.length < 2) return repeatRows;
            const meanCells = specs.map((spec) => {
                const values = repeats
                    .map((r) => this._historyBestSeriesValue(this._historyMetricBucket(r, scope)?.[spec.key], spec.direction))
                    .filter((v) => typeof v === 'number' && Number.isFinite(v));
                if (!values.length) {
                    return `<td class="py-1 px-2 text-sm">${this._historyFormatMetricValue(null, spec)}</td>`;
                }
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
                const std = Math.sqrt(variance);
                return `<td class="py-1 px-2 text-sm font-semibold">${this._historyFormatMetricValue(mean, spec)} <span class="text-slate-400 font-normal">± ${this._historyFormatMetricValue(std, spec)}</span></td>`;
            }).join('');
            return `${repeatRows}
                <tr class="border-t-2 border-slate-500/60 bg-slate-800/30">
                    <td class="py-1 px-2 text-slate-200 text-xs font-semibold">Mean ± Std</td>
                    ${meanCells}
                </tr>`;
        };

        return scopes.map(({ scope, scopeLabel }) => {
            const title = `Summary — ${runId} (${this._historyMetricSectionTitle('Best', scopeLabel)})`;
            const headers = this._historyMetricSpecs().map((spec) => `<th class="py-1 px-2">${this._historyComparisonMetricLabel(spec)}</th>`).join('');
            return `<p class="font-semibold text-xs text-slate-400 mb-2 uppercase tracking-wide">${this._escapeHtml(title)}</p>
                <p class="text-xs text-slate-500 mb-2">Each repeat row shows best-over-series values; the Mean ± Std row aggregates them across repeats. The charts preserve the per-repeat trajectories and the mean band.</p>
                <div class="hist-summary-table-wrap mb-4">
                <table class="w-full text-left table-fixed hist-summary-table">
                    ${this._historyMetricTableColgroup('repeat')}
                    <thead class="text-xs text-slate-500 uppercase">
                        <tr>
                            <th class="py-1 px-2">Repeat</th>
                            ${headers}
                        </tr>
                    </thead>
                    <tbody>${renderRows(scope)}</tbody>
                </table>
                </div>`;
        }).join('');
    },

    /**
     * Alternate wrapper specifically triggering UI delete mode via history header actions.
     * @returns {Promise<void>}
     */
    async _deleteHistoryRun() {
        this._toggleDeleteMode();
    },
};
