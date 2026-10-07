/**
 * charts.js — Chart.js helpers for live metrics and history charts.
 *
 * This module is responsible for managing all Chart.js instances across the 
 * "Live Monitor" and "History & Analysis" tabs. It includes utilities for 
 * computing statistical aggregations (mean, standard deviation) across 
 * repeated runs and standardizing chart palettes.
 *
 * Globals exported:
 *   @type {MetricsCharts} metricsCharts - A global singleton for managing UI charts.
 */

// ── Shared light-theme defaults ───────────────────────────────────────────────

/**
 * Global default configuration options for Chart.js instances.
 * @constant {Object}
 * @private
 */
const LIGHT_OPTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
        legend: {
            position: 'top',
            labels: { color: '#334155', font: { size: 11 }, boxWidth: 12 },
        },
        tooltip: {
            backgroundColor: 'rgba(255,255,255,0.98)',
            titleColor: '#0f172a',
            bodyColor: '#334155',
            borderColor: '#cbd5e1',
            borderWidth: 1,
        },
    },
    scales: {
        x: {
            grid: { color: '#e5edf9' },
            ticks: { color: '#475569', font: { size: 10 } },
            title: { display: true, text: 'Round', color: '#64748b', font: { size: 11 } },
        },
        y: {
            grid: { color: '#e5edf9' },
            ticks: { color: '#475569', font: { size: 10 } },
        },
    },
};

// ── Palette for multi-repeat lines ────────────────────────────────────────────

/**
 * Standardized color palette used to distinguish individual repeat series or 
 * compared runs in the history charts.
 * @constant {Array<string>}
 * @private
 */
const PALETTE = [
    '#1d4ed8', '#0f766e', '#7c3aed', '#c2410c',
    '#0369a1', '#b45309', '#15803d', '#be123c',
];

/**
 * Specifications for the four standard history charts.
 * Format: [ htmlCanvasId, chartLabel, yAxisLabel, extractionFunction ]
 * @constant {Array<Array<any>>}
 * @private
 */
const HISTORY_CHART_SPECS = [
    ['hist-val-acc',   'Global Validation Accuracy', 'Accuracy', m => _safeGet(m, 'aggregated_val', 'accuracy')],
    ['hist-val-loss',  'Global Validation Loss',     'Loss',     m => _safeGet(m, 'aggregated_val', 'loss')],
    ['hist-test-acc',  'Global Test Accuracy',       'Accuracy', m => _safeGet(m, 'aggregated_test', 'accuracy')],
    ['hist-test-loss', 'Global Test Loss',           'Loss',     m => _safeGet(m, 'aggregated_test', 'loss')],
];

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Normalizes the metrics payload to always return an array of repeats.
 * Single-run payloads are wrapped in an array.
 * 
 * @private
 * @param {Object} m - The raw metrics object.
 * @returns {Array<Object>} An array of metric dictionaries per repeat.
 */
function _normalizeRepeats(m) {
    if (!m || !Object.keys(m).length) return [];
    if (Array.isArray(m.all_metrics) && m.all_metrics.length) return m.all_metrics;
    return [m];
}

/**
 * Safely resolves an array path from a nested object, preventing exceptions 
 * on missing or malformed keys.
 * 
 * @private
 * @param {Object} obj - Source object to traverse.
 * @param {...string} keys - Nested property keys.
 * @returns {Array<*>} Resolved array, or an empty array if invalid.
 */
function _safeGet(obj, ...keys) {
    let cur = obj;
    for (const k of keys) {
        if (cur == null || typeof cur !== 'object') return [];
        cur = cur[k];
    }
    return Array.isArray(cur) ? cur : [];
}

/**
 * Computes the element-wise topological mean across multiple numeric arrays.
 * 
 * @private
 * @param {Array<Array<number>>} arrays - Collection of numeric series (e.g. repeats).
 * @returns {Array<number|null>} A single array of mean values, padded with nulls where data is missing.
 */
function _mean(arrays) {
    if (!arrays.length) return [];
    const len = Math.max(...arrays.map(a => a.length));
    return Array.from({ length: len }, (_, i) => {
        const vals = arrays.map(a => a[i]).filter(v => v != null && isFinite(v));
        return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
    });
}

/**
 * Computes the element-wise standard deviation across multiple numeric arrays.
 * 
 * @private
 * @param {Array<Array<number>>} arrays - Collection of numeric series.
 * @returns {Array<number|null>} A single array of standard deviations.
 */
function _stdev(arrays) {
    if (!arrays.length) return [];
    const len = Math.max(...arrays.map(a => a.length));
    return Array.from({ length: len }, (_, i) => {
        const vals = arrays.map(a => a[i]).filter(v => v != null && isFinite(v));
        if (vals.length < 2) return 0;
        const mu = vals.reduce((s, v) => s + v, 0) / vals.length;
        const variance = vals.reduce((s, v) => s + ((v - mu) ** 2), 0) / vals.length;
        return Math.sqrt(variance);
    });
}

/**
 * Builds a compact label for history comparison legends based on the run ID.
 * Extracts the last 6 digits typically associated with the timestamp.
 * 
 * @private
 * @param {string} runId - The full unified run identifier string.
 * @param {number} index - The 0-based series index, used for the prefix (R1, R2).
 * @returns {string} Short formatted label.
 */
function _shortRunLabel(runId, index) {
    const name = String(runId || '');
    // Default run ids end in a timestamp (…_HHMMSS); show a compact tail for those.
    // Custom display names have no such suffix, so show the full name (capped) instead.
    const m = name.match(/_(\d{6})$/);
    if (m) return `R${index + 1}-${m[1]}`;
    const label = name.length > 24 ? `${name.slice(0, 23)}…` : name;
    return `R${index + 1}-${label}`;
}

// ── ChartManager ──────────────────────────────────────────────────────────────

/**
 * Low-level wrapper managing Chart.js DOM instances.
 * Ensures canvases are correctly refreshed and memory is freed upon destruction.
 */
class ChartManager {
    /**
     * Initializes the instance dictionary.
     */
    constructor() {
        /** @type {Object<string, Object>} Map of DOM IDs to Chart instances. */
        this.instances = {};
    }

    /**
     * Creates or recreates a Chart.js instance attached to a target canvas ID.
     * 
     * @param {string} id - The DOM ID of the target `<canvas>` element.
     * @param {Array<Object>} datasets - Array of dataset definitions.
     * @param {string} [yTitle=''] - Optional title for the Y-axis.
     * @param {Object} [extraOpts={}] - Base settings overrides.
     * @returns {Object|null} The raw Chart.js instantiated object, or null if the canvas was missing.
     */
    create(id, datasets, yTitle = '', extraOpts = {}) {
        const canvas = document.getElementById(id);
        if (!canvas) return null;
        if (this.instances[id]) {
            this.instances[id].destroy();
        }
        const opts = JSON.parse(JSON.stringify(LIGHT_OPTS));
        if (yTitle) opts.scales.y.title = { display: true, text: yTitle, color: '#64748b', font: { size: 11 } };
        Object.assign(opts, extraOpts);

        this.instances[id] = new Chart(canvas, {
            type: 'line',
            data: { labels: [], datasets },
            options: opts,
        });
        return this.instances[id];
    }

    /**
     * Updates labels and dataset values for an active chart without destroying it.
     * 
     * @param {string} id - The DOM ID of the target canvas.
     * @param {Array<number|string>} labels - X-axis ticks.
     * @param {Array<Array<number|null>>} dataArrays - Data arrays, mapped by order to existing datasets.
     */
    update(id, labels, dataArrays) {
        const chart = this.instances[id];
        if (!chart) return;
        chart.data.labels = labels;
        dataArrays.forEach((data, i) => {
            if (chart.data.datasets[i]) chart.data.datasets[i].data = data;
        });
        chart.update('none'); // Update without heavy animations
    }

    /**
     * Destroys a single chart instance and frees its memory.
     * @param {string} id - Canvas ID.
     */
    destroy(id) {
        if (this.instances[id]) {
            this.instances[id].destroy();
            delete this.instances[id];
        }
    }

    /**
     * Destroys all currently managed chart instances globally.
     */
    destroyAll() {
        Object.keys(this.instances).forEach(id => this.destroy(id));
    }
}

// ── MetricsCharts ─────────────────────────────────────────────────────────────

/**
 * Higher-level abstraction linking raw federated learning metrics 
 * directly to the UI rendering elements.
 */
class MetricsCharts {
    /**
     * Instantiates the core ChartManager dependency.
     */
    constructor() {
        /** @type {ChartManager} */
        this.manager = new ChartManager();
    }

    // ── Live charts (Training Monitor tab) ────────────────────────────────────

    /**
     * Initializes the two main live monitoring charts (Accuracy & Loss).
     * Binds them to `live-acc` and `live-loss` canvas elements.
     */
    initLive() {
        this.manager.create('live-acc', [
                {
                    label: 'Global Validation Accuracy',
                    data: [],
                    borderColor: '#0f766e',
                    backgroundColor: 'rgba(15,118,110,0.08)',
                    fill: false,
                    tension: 0.35,
                    pointRadius: 2.5,
                    borderWidth: 2,
                },
                {
                    label: 'Global Test Accuracy',
                    data: [],
                    borderColor: '#b45309',
                    backgroundColor: 'rgba(180,83,9,0.08)',
                    fill: false,
                    tension: 0.35,
                    pointRadius: 2.5,
                    borderWidth: 2,
                },
        ], 'Accuracy');

        this.manager.create('live-loss', [
                {
                    label: 'Global Validation Loss',
                    data: [],
                    borderColor: '#0f766e',
                    backgroundColor: 'rgba(15,118,110,0.08)',
                    fill: false,
                    tension: 0.35,
                    pointRadius: 2.5,
                    borderWidth: 2,
                },
                {
                    label: 'Global Test Loss',
                    data: [],
                    borderColor: '#b45309',
                    backgroundColor: 'rgba(180,83,9,0.08)',
                    fill: false,
                    tension: 0.35,
                    pointRadius: 2.5,
                    borderWidth: 2,
                },
        ], 'Loss');
    }

    /**
     * Injects updated live payload data straight into the active monitor charts.
     * @param {Object} metrics - Live run metrics dictionary.
     */
    updateLive(metrics) {
        const valAcc = _safeGet(metrics, 'aggregated_val', 'accuracy');
        const testAcc = _safeGet(metrics, 'aggregated_test', 'accuracy');

        const valLoss = _safeGet(metrics, 'aggregated_val', 'loss');
        const testLoss = _safeGet(metrics, 'aggregated_test', 'loss');

        const labels = Array.from(
            {
                length: Math.max(
                    valAcc.length,
                    testAcc.length,
                    valLoss.length,
                    testLoss.length,
                ),
            },
            (_, i) => i + 1,
        );
        this.manager.update('live-acc', labels, [valAcc, testAcc]);
        this.manager.update('live-loss', labels, [valLoss, testLoss]);
    }

    // ── History charts (History & Analysis tab) ───────────────────────────────

    /**
     * Initializes the six analytics canvases residing in the History tab.
     */
    initHistory() {
        for (const [id, label, yTitle] of HISTORY_CHART_SPECS) {
            this.manager.create(id, [{ label, data: [], borderColor: '#1d4ed8', tension: 0.35, pointRadius: 2, borderWidth: 2 }], yTitle);
        }
    }

    /**
     * Updates the history analytics charts for a single selected run.
     * Supports multi-repeat configurations automatically by plotting all repeats 
     * accompanied by a calculated dashed mean line.
     * 
     * @param {Object} metrics - Historical run metrics payload.
     */
    updateHistory(metrics) {
        const repeats = _normalizeRepeats(metrics);
        if (!repeats.length) return;

        for (const [id, , , extract] of HISTORY_CHART_SPECS) {
            const chart = this.manager.instances[id];
            if (!chart) continue;

            const allSeries = repeats.map(extract);
            const meanSeries = _mean(allSeries);

            const datasets = allSeries.map((data, i) => ({
                label: `Repeat ${i + 1}`,
                data,
                borderColor: PALETTE[i % PALETTE.length],
                backgroundColor: 'transparent',
                tension: 0.35,
                pointRadius: 2,
                borderWidth: 1.5,
            }));

            // Include dotted mean line when multiple repeats are tracked.
            if (repeats.length > 1) {
                datasets.push({
                    label: 'Mean',
                    data: meanSeries,
                    borderColor: '#0f172a',
                    backgroundColor: 'transparent',
                    borderDash: [5, 3],
                    tension: 0.35,
                    pointRadius: 0,
                    borderWidth: 2,
                });
            }

            const len = Math.max(...allSeries.map(a => a.length), 0);
            chart.data.labels = Array.from({ length: len }, (_, i) => i + 1);
            chart.data.datasets = datasets;
            chart.update('none');
        }
    }

    /**
     * Maps comparison analytics data across multiple distinct runs.
     * Generates a comprehensive view incorporating each run's mean line, 
     * and a semi-transparent band showing the cumulative ±1 standard deviation.
     * 
     * @param {Record<string, Object>} runMetricsMap - Map associating run IDs to their specific metrics payloads.
     */
    updateHistoryComparison(runMetricsMap) {
        const runNames = Object.keys(runMetricsMap || {}).slice(0, 5);
        if (!runNames.length) return;

        for (const [id, , , extract] of HISTORY_CHART_SPECS) {
            const chart = this.manager.instances[id];
            if (!chart) continue;

            const datasets = [];
            let maxLen = 0;
            const runMeanSeries = [];

            runNames.forEach((name, i) => {
                const metrics = runMetricsMap[name] || {};
                const repeats = _normalizeRepeats(metrics);
                const allSeries = repeats.map(extract);
                const meanSeries = _mean(allSeries);
                runMeanSeries.push(meanSeries);
                maxLen = Math.max(maxLen, meanSeries.length);

                const color = PALETTE[i % PALETTE.length];
                const label = _shortRunLabel(name, i);
                
                datasets.push({
                    label,
                    data: meanSeries,
                    borderColor: color,
                    backgroundColor: 'transparent',
                    tension: 0.35,
                    pointRadius: 2,
                    borderWidth: 2,
                });
            });

            // Global statistics (aggregate mean + stdev over multiple runs)
            const globalMean = _mean(runMeanSeries);
            const globalStd = _stdev(runMeanSeries);
            const upper = globalMean.map((m, idx) => (m == null ? null : m + (globalStd[idx] || 0)));
            const lower = globalMean.map((m, idx) => (m == null ? null : m - (globalStd[idx] || 0)));
            maxLen = Math.max(maxLen, globalMean.length, upper.length, lower.length);

            // Shaded std-deviation area
            datasets.push({
                label: '_std_upper',
                data: upper,
                borderColor: 'transparent',
                backgroundColor: 'transparent',
                tension: 0.25,
                pointRadius: 0,
                borderWidth: 0,
            });
            datasets.push({
                label: '±1σ (runs)',
                data: lower,
                borderColor: 'transparent',
                backgroundColor: 'rgba(30,64,175,0.16)', // Brand navy fill
                fill: '-1',
                tension: 0.25,
                pointRadius: 0,
                borderWidth: 0,
            });
            
            // Global mean trace
            datasets.push({
                label: 'Mean (runs)',
                data: globalMean,
                borderColor: '#0f172a',
                backgroundColor: 'transparent',
                borderDash: [6, 3],
                tension: 0.25,
                pointRadius: 0,
                borderWidth: 2,
            });

            chart.data.labels = Array.from({ length: maxLen }, (_, i) => i + 1);
            chart.data.datasets = datasets;

            // Hide the structural '_std_upper' helper dataset from the legend.
            chart.options.plugins.legend.labels.filter = (item, data) => {
                const ds = data.datasets?.[item.datasetIndex];
                return ds?.label !== '_std_upper';
            };
            chart.update('none');
        }
    }
}

// ── Single global instance ────────────────────────────────────────────────────

/** 
 * Exported global singleton chart manager bound to the window space.
 * @type {MetricsCharts} 
 */
const metricsCharts = new MetricsCharts();
