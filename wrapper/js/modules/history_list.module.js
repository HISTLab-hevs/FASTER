window.AppHistoryListModule = {
    /**
     * Determines the maximum number of items per history page.
     * @returns {number} The page size.
     */
    _historyPageSize() {
        return 10;
    },

    /**
     * Computes a paginated slice from an array of items.
     * @param {Array<*>} items The full array of items.
     * @param {number|string} page The target page number (1-indexed).
     * @param {number} [pageSize=10] The number of items per page.
     * @returns {{page: number, total: number, pages: number, items: Array<*>}} Pagination slice details.
     */
    _getPageSlice(items, page, pageSize = 10) {
        const total = Array.isArray(items) ? items.length : 0;
        const pages = Math.max(1, Math.ceil(total / pageSize));
        const safePage = Math.min(Math.max(1, Number(page) || 1), pages);
        const start = (safePage - 1) * pageSize;
        const end = start + pageSize;
        return {
            page: safePage,
            total,
            pages,
            items: (items || []).slice(start, end),
        };
    },

    /**
     * Renders pagination controls for the history or other data tables.
     * @param {string} containerId The DOM element ID to render the pager into.
     * @param {number} page The current page number.
     * @param {number} pages The total number of pages available.
     * @param {string} setterName The App method name to call on navigate.
     * @returns {void}
     */
    _renderPager(containerId, page, pages, setterName) {
        const el = document.getElementById(containerId);
        if (!el) return;
        if ((pages || 1) <= 1) {
            el.innerHTML = '';
            return;
        }
        const prevDisabled = page <= 1 ? 'disabled' : '';
        const nextDisabled = page >= pages ? 'disabled' : '';
        const prevPage = Math.max(1, page - 1);
        const nextPage = Math.min(pages, page + 1);
        el.innerHTML = `
            <button class="hist-page-btn" ${prevDisabled} data-page-setter="${this._escapeHtml(setterName)}" data-page-value="${prevPage}">Prev</button>
            <span class="hist-page-label">Page ${page} / ${pages}</span>
            <button class="hist-page-btn" ${nextDisabled} data-page-setter="${this._escapeHtml(setterName)}" data-page-value="${nextPage}">Next</button>
        `;
    },

    /**
     * Updates the current view page for the history table and repopulates the runs.
     * @param {number|string} page The target page number representation.
     * @returns {void}
     */
    _setHistTablePage(page) {
        this._state.histTablePage = Math.max(1, Number(page) || 1);
        this._refreshRunsList(false, { silent: true });
    },

    /**
     * Returns the active server-side sort state for the history table.
     * @returns {{key: string, dir: string}} Current sort configuration.
     */
    _historySortState() {
        const current = this._state.histTableSort || {};
        return {
            key: String(current.key || 'run_ts'),
            dir: String(current.dir || 'desc'),
        };
    },

    /**
     * Merge one or more run rows into the lookup cache used by detail/compare flows.
     * @param {Array<Record<string, *>>} runs Run summary rows.
     * @returns {void}
     */
    _mergeHistoryLookup(runs) {
        const next = { ...(this._state.historyRunLookup || {}) };
        (runs || []).forEach((run) => {
            const key = String(run?.id || run?.name || '').trim();
            if (key) next[key] = run;
        });
        this._state.historyRunLookup = next;
    },

    /**
     * Refreshes cached summaries for selected/detail runs that may not be on the current page.
     * @returns {Promise<void>}
     */
    async _syncHistorySelectionCache() {
        const tracked = Array.from(new Set([
            ...(this._state.historySelectedRuns || []),
            this._state.histRunId || '',
        ].filter(Boolean)));
        if (!tracked.length) return;

        const rows = (await api.getRunsByIds(tracked)).map((r) => this._normalizeHistoryRun(r));
        const valid = new Set(rows.filter((r) => !['running', 'queued', 'pending'].includes(String(r.status || '').toLowerCase())).map((r) => r.id || r.name));
        this._mergeHistoryLookup(rows);
        this._state.historySelectedRuns = (this._state.historySelectedRuns || []).filter((runId) => valid.has(runId));
        if (this._state.histRunId && !rows.some((r) => (r.id || r.name) === this._state.histRunId)) {
            this._state.histRunId = null;
        }
    },

    /**
     * Debounces server-driven history list refresh while typing in search controls.
     * @returns {void}
     */
    _scheduleHistoryListRefresh() {
        clearTimeout(this._state.histSearchDebounceTimer);
        this._state.histSearchDebounceTimer = setTimeout(() => {
            this._refreshRunsList(false, { silent: true });
        }, 180);
    },

    /**
     * Validates and triggers a refresh of the comparison dashboard based on the selected run keys.
     * @returns {void}
     */
    _refreshComparisonAreaFromSelection() {
        const selected = (this._state.historySelectedRuns || []).slice();
        const empty = document.getElementById('hist-compare-empty');
        const statusEl = document.getElementById('hist-status');
        const splitWarn = document.getElementById('hist-split-warning');
        const summaryEl = document.getElementById('hist-summary');
        const chartsEl = document.getElementById('hist-charts');
        const dlRow = document.getElementById('hist-download-row');

        if (selected.length < 2) {
            summaryEl?.classList.add('hidden');
            chartsEl?.classList.add('hidden');
            dlRow?.classList.add('hidden');
            splitWarn?.classList.add('hidden');
            if (statusEl) {
                statusEl.classList.add('hidden');
                statusEl.textContent = '';
            }
            if (empty) {
                empty.textContent = 'Select at least 2 completed runs to compare their metrics here.';
                empty.classList.remove('hidden');
            }
            return;
        }

        empty?.classList.add('hidden');
        this._loadHistoryRun(selected);
    },

    /**
     * Fetches the latest history runs from the API and refreshes the list state.
     * @returns {Promise<void>}
     */
    async _refreshRunsList(forceRefresh = false, options = {}) {
        const silent = !!options?.silent;
        const btn = document.getElementById('hist-refresh-btn');
        const icon = btn?.querySelector('i');
        const statusEl = document.getElementById('hist-status');
        const sort = this._historySortState();
        if (!silent && btn) btn.disabled = true;
        if (!silent && icon) icon.classList.add('hist-refresh-spin');
        if (!silent && statusEl) {
            statusEl.textContent = forceRefresh ? 'Refreshing runs from database...' : 'Refreshing runs...';
            statusEl.classList.remove('hidden');
        }
        try {
            const payload = await api.getRunsPage({
                search: this._state.histSearchQuery || '',
                searchField: this._state.histSearchField || 'all',
                allUsers: !!this._state.isAdmin,
                refreshDb: !!forceRefresh,
                page: this._state.histTablePage || 1,
                pageSize: this._historyPageSize(),
                sortKey: sort.key,
                sortDir: sort.dir,
            });
            const pageRuns = (payload.runs || []).map((r) => this._normalizeHistoryRun(r));
            this._state.runsCache = pageRuns;
            this._state.histRunsPagination = payload.pagination || {
                page: 1,
                page_size: this._historyPageSize(),
                pages: 1,
                total: pageRuns.length,
            };
            this._state.histTablePage = Math.max(1, Number(this._state.histRunsPagination.page) || 1);
            this._mergeHistoryLookup(pageRuns);
            await this._syncHistorySelectionCache();
            this._updateHistorySortHeader();
            this._populateRunsTable(this._sortedRuns());
            this._updateSelectedRunsHint();
            this._refreshComparisonAreaFromSelection();
            if (!silent && statusEl) {
                statusEl.textContent = `Runs refreshed (${this._state.histRunsPagination.total || pageRuns.length})`;
            }
        } catch (e) {
            this._reportNonBlockingIssue?.('history refresh', e, {
                statusId: 'hist-status',
                statusMessage: 'Failed to refresh runs.',
                dedupeMs: 15000,
            });
            if (!silent && statusEl) {
                statusEl.textContent = 'Failed to refresh runs.';
            }
        } finally {
            if (!silent && btn) btn.disabled = false;
            if (!silent && icon) icon.classList.remove('hist-refresh-spin');
        }
    },

    /**
     * Starts background polling for History.
     * @returns {void}
     */
    _startHistoryAutoRefresh() {
        if (this._state.historyAutoLoadTimer) return;
        this._state.historyAutoLoadTimer = setInterval(() => {
            if (this._state.tab !== 3) return;
            this._refreshRunsList(true, { silent: true });
        }, 15000);
    },

    /**
     * Stops background polling for History.
     * @returns {void}
     */
    _stopHistoryAutoRefresh() {
        clearInterval(this._state.historyAutoLoadTimer);
        this._state.historyAutoLoadTimer = null;
        clearTimeout(this._state.histSearchDebounceTimer);
        this._state.histSearchDebounceTimer = null;
    },

    /**
     * Normalizes a raw run record from the server into a standard history row object.
     * @param {Record<string, *>} run Raw run object from the API.
     * @returns {Record<string, *>} Formatted run object with standardized properties.
     */
    _normalizeHistoryRun(run) {
        const name = run?.name || run?.id || '';
        const runTs = this._extractRunTs(name);
        const createdAt = this._formatCreatedAt(run?.created_at);
        const createdAtSort = this._createdAtSortValue(run?.created_at);
        const derivedDate = this._formatRunTs(runTs);
        return {
            id: run?.id || run?.name,
            name,
            owner: run?.owner || (name.includes('_') ? name.split('_', 1)[0] : 'unknown'),
            method: run?.method || '',
            dataset: run?.dataset || '',
            status: run?.status || 'completed',
            evaluation_split_mode: run?.evaluation_split_mode || 'train_val_test',
            model_name: run?.model_name || 'DefaultNet',
            best_accuracy: run?.best_accuracy,
            run_ts: runTs,
            run_sort_ts: createdAtSort || runTs || 0,
            created_at: run?.created_at || '',
            run_date: createdAt || derivedDate,
        };
    },

    /**
     * Format backend ISO datetime into a stable local display string.
     * @param {string} value ISO datetime string.
     * @returns {string} Formatted timestamp or empty string.
     */
    _formatCreatedAt(value) {
        const raw = String(value || '').trim();
        if (!raw) return '';
        const d = new Date(raw);
        if (!Number.isFinite(d.getTime())) return '';
        const pad = (n) => String(n).padStart(2, '0');
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    },

    /**
     * Converts backend ISO datetime into a sortable numeric timestamp.
     * @param {string} value ISO datetime string.
     * @returns {number} Millisecond timestamp or 0 when invalid.
     */
    _createdAtSortValue(value) {
        const raw = String(value || '').trim();
        if (!raw) return 0;
        const ts = Date.parse(raw);
        return Number.isFinite(ts) ? ts : 0;
    },

    /**
     * Formats a raw run timestamp into a readable date string.
     * @param {string|number} ts Raw timestamp string.
     * @returns {string} Formatted timestamp string or fallback.
     */
    _formatRunTs(ts) {
        if (!ts) return '-';
        const s = String(ts);
        if (s.length !== 14) return '-';
        return `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)} ${s.slice(8, 10)}:${s.slice(10, 12)}:${s.slice(12, 14)}`;
    },

    /**
     * Retrieves a run object from the local history cache.
     * @param {string} key The run ID or name.
     * @returns {Record<string, *>|null} The cached run object or null if missing.
     */
    _runByKey(key) {
        const direct = (this._state.runsCache || []).find((r) => (r.id || r.name) === key || r.name === key);
        if (direct) return direct;
        return this._state.historyRunLookup?.[key] || null;
    },

    /**
     * Extracts a numeric timestamp for sorting by inspecting a run name suffix.
     * @param {string} name The run name.
     * @returns {number} The derived timestamp integer or 0.
     */
    _extractRunTs(name) {
        const m = String(name || '').match(/(?:^|_)(\d{8})_(\d{6})(?:_|$)/);
        if (!m) return 0;
        return Number(`${m[1]}${m[2]}`);
    },

    /**
     * Returns the cached runs filtered by the current search query and sorted by table rules.
     * @returns {Array<Record<string, *>>} Filtered and sorted run objects.
     */
    _sortedRuns() {
        return [...(this._state.runsCache || [])];
    },

    /**
     * Changes the current history table sorting logic.
     * @param {string} key The column name used for sorting.
     * @returns {void}
     */
    _setHistoryTableSort(key) {
        const current = this._state.histTableSort || { key: null, dir: 'asc' };
        const defaultDir = key === 'run_ts' ? 'desc' : 'asc';
        const nextDir = current.key === key ? (current.dir === 'asc' ? 'desc' : 'asc') : defaultDir;
        this._state.histTableSort = { key, dir: nextDir };
        this._state.histTablePage = 1;
        this._updateHistorySortHeader();
        this._refreshRunsList(false, { silent: true });
    },

    /**
     * Updates header icons to reflect current active sorting parameters.
     * @returns {void}
     */
    _updateHistorySortHeader() {
        document.querySelectorAll('[data-hist-sort]').forEach((th) => {
            const key = th.getAttribute('data-hist-sort') || '';
            const marker = th.querySelector('[data-sort-indicator]');
            if (!marker) return;
            const active = this._state.histTableSort?.key === key;
            marker.textContent = active ? (this._state.histTableSort.dir === 'asc' ? ' ▲' : ' ▼') : '';
        });
    },

    /**
     * Renders the runs history table and selection hint.
     * @returns {void}
     */
    _renderHistoryRunChecklist() {
        this._populateRunsTable(this._sortedRuns());
        this._updateSelectedRunsHint();
    },

    /**
     * Regenerates the history runs table body with the provided rows.
     * @param {Array<Record<string, *>>} runs Array of sorted run objects to display.
     * @returns {void}
     */
    _populateRunsTable(runs) {
        const tbody = document.getElementById('runs-table-body');
        if (!tbody) return;
        const canSeeOwner = !!this._state.isAdmin;
        const emptyColspan = canSeeOwner ? 10 : 9;
        const pagination = this._state.histRunsPagination || {
            page: Math.max(1, Number(this._state.histTablePage) || 1),
            pages: 1,
            total: Array.isArray(runs) ? runs.length : 0,
        };
        this._state.histTablePage = Math.max(1, Number(pagination.page) || 1);
        this._renderPager('hist-table-pagination', this._state.histTablePage, Math.max(1, Number(pagination.pages) || 1), '_setHistTablePage');
        if (!runs.length) {
            tbody.innerHTML = `<tr><td colspan="${emptyColspan}" class="px-4 py-4 text-center text-slate-500 text-sm">No runs found.</td></tr>`;
            return;
        }
        const selectedSet = new Set(this._state.historySelectedRuns || []);
        tbody.innerHTML = runs.map((r) => {
            const key = r.id || r.name;
            const selected = selectedSet.has(key);
            const status = String(r.status || '').toLowerCase();
            const statusBadgeTone =
                status === 'running'
                    ? 'running'
                    : (status === 'queued' || status === 'pending')
                        ? 'queued'
                    : (status === 'failed' || status === 'stopped' ? 'failed' : 'completed');
            const statusBadgeClass =
                statusBadgeTone === 'running'
                    ? 'ui-badge-warn'
                    : (statusBadgeTone === 'queued' ? 'ui-badge-queued'
                        : (statusBadgeTone === 'failed' ? 'ui-badge-danger' : 'ui-badge-success'));
            const selectable = status !== 'running' && status !== 'queued' && status !== 'pending' && status !== 'failed';
            const ownerRaw = r.owner || 'unknown';
            const ownerDisp = ownerRaw.length > 10 ? `${ownerRaw.slice(0, 10)}...` : ownerRaw;
            const ownerCell = canSeeOwner
                ? `<td class="text-slate-400 text-xs" title="${this._escapeHtml(ownerRaw)}">${this._escapeHtml(ownerDisp)}</td>`
                : '';
            return `
            <tr class="border-t border-slate-700/40 cursor-pointer hover:bg-slate-700/20 ${this._state.histRunId === key ? 'hist-row-selected' : ''} ${selected ? 'hist-row-compare-selected' : ''} ${r.status === 'running' ? 'hist-row-disabled' : ''}"
                data-run-row="${this._escapeHtml(String(r.id || r.name))}">
                <td class="font-mono text-xs text-slate-300">${r.name}</td>
                ${ownerCell}
                <td class="text-slate-400">${r.method || '—'}</td>
                <td class="text-slate-400">${r.dataset || '—'}</td>
                <td class="text-slate-300">
                    <button class="underline decoration-dotted inline-block max-w-[170px] truncate align-bottom" title="${this._escapeHtml(r.model_name || 'DefaultNet')}" data-run-model-code="${this._escapeHtml(String(r.id || r.name))}">${r.model_name || 'DefaultNet'}</button>
                </td>
                <td>
                    <span class="ui-badge-pill ${r.evaluation_split_mode === 'train_val' ? 'ui-badge-info' : 'ui-badge-success'}">
                        ${r.evaluation_split_mode === 'train_val' ? 'train_val' : 'train_val_test'}
                    </span>
                </td>
                <td class="text-slate-300">${r.best_accuracy != null ? r.best_accuracy.toFixed(3) : '—'}</td>
                <td class="text-slate-400 text-xs whitespace-nowrap">${r.run_date || '—'}</td>
                <td>
                    <div class="hist-status-compare-cell">
                        <span class="hist-run-status-badge ui-badge-pill ${statusBadgeClass}">
                            ${r.status}
                        </span>
                        ${selectable
                            ? `<button class="compare-toggle-btn ${selected ? 'active' : ''}" title="Toggle compare selection" data-run-compare="${this._escapeHtml(String(key))}">Compare</button>`
                            : ''}
                    </div>
                </td>
                <td class="text-right">
                    ${this._state.histDeleteMode && r.status !== 'running'
                        ? `<button class="run-del-x" title="Delete run" aria-label="Delete run ${this._escapeHtml(r.id || r.name)}" data-run-delete="${this._escapeHtml(String(r.id || r.name))}">X</button>`
                        : ''}
                </td>
            </tr>
        `;
        }).join('');
    },

    /**
     * Handles run row click events to open specific details.
     * @param {string} runKey The identifier of the run.
     * @returns {void}
     */
    _onRunRowClicked(runKey) {
        if (this._state.histDeleteMode) return;
        const run = this._runByKey(runKey);
        if (run?.status === 'running') {
            this._showToast('Running runs cannot be selected in history', 'warn');
            return;
        }
        this._state.histRunId = runKey;
        this._populateRunsTable(this._sortedRuns());
        this._openRunDetail(runKey);
    },

    /**
     * Selects exactly one run for comparison.
     * @param {string} runName The specific name of the run.
     * @returns {void}
     */
    _selectSingleRun(runName) {
        const run = this._runByKey(runName);
        if (run) this._mergeHistoryLookup([run]);
        this._state.historySelectedRuns = [runName];
        this._populateRunsTable(this._sortedRuns());
    },

    /**
     * Toggles inclusion of a run inside the comparison pool.
     * @param {string} runKey The identifier of the toggled run.
     * @returns {void}
     */
    _toggleRunSelection(runKey) {
        const selected = new Set(this._state.historySelectedRuns || []);
        if (selected.has(runKey)) {
            selected.delete(runKey);
        } else {
            if (selected.size >= 5) {
                this._showToast('You can compare up to 5 runs', 'warn');
                return;
            }
            const run = this._runByKey(runKey);
            if (run?.status === 'running') {
                this._showToast('Running runs cannot be selected in history', 'warn');
                return;
            }
            if (run) this._mergeHistoryLookup([run]);
            selected.add(runKey);
        }
        this._state.historySelectedRuns = Array.from(selected);
        this._populateRunsTable(this._sortedRuns());
        this._updateSelectedRunsHint();
        this._refreshComparisonAreaFromSelection();
    },

    /**
     * Activates or disables delete actions in the history table.
     * @returns {void}
     */
    _toggleDeleteMode() {
        const wrap = document.getElementById('hist-runs-table-wrap');
        const delBtn = document.getElementById('hist-delete-btn');
        if (wrap?.classList.contains('hidden')) {
            this._showToast('Expand runs table before enabling delete mode', 'warn');
            return;
        }
        this._state.histDeleteMode = !this._state.histDeleteMode;
        wrap?.classList.toggle('hist-delete-mode', !!this._state.histDeleteMode);
        delBtn?.classList.toggle('delete-mode-active', !!this._state.histDeleteMode);
        this._populateRunsTable(this._sortedRuns());
        this._showToast(this._state.histDeleteMode ? 'Delete mode enabled: click red X in a row' : 'Delete mode disabled', 'info');
    },

    /**
     * Asks user for confirmation and attempts deletion of a run.
     * @param {string} runKey The identifier for the chosen run.
     * @returns {Promise<void>}
     */
    async _requestDeleteRun(runKey) {
        const run = this._runByKey(runKey);
        const runId = run?.id || run?.name || runKey;
        const ok = await this._confirmDialog(`Delete run "${runId}"? This cannot be undone.`);
        if (!ok) return;
        try {
            await api.deleteRun(runId);
            this._state.historySelectedRuns = (this._state.historySelectedRuns || []).filter((r) => r !== runKey && r !== runId);
            this._showToast(`Deleted: ${runId}`, 'success');
            await this._refreshRunsList();

            if (this._state.histRunId === runId || !(this._state.historySelectedRuns || []).length) {
                this._state.histRunId = null;
                document.getElementById('hist-charts')?.classList.add('hidden');
                document.getElementById('hist-summary')?.classList.add('hidden');
                document.getElementById('hist-download-row')?.classList.add('hidden');
                const st = document.getElementById('hist-status');
                if (st) st.textContent = 'Run deleted.';
            } else {
                this._refreshComparisonAreaFromSelection();
            }
            this._refreshComparisonAreaFromSelection();
        } catch (e) {
            this._showToast(e.response?.data?.detail || 'Delete failed', 'error');
        }
    },

    /**
     * Synchronizes the comparison selection hint with stored run context.
     * @returns {void}
     */
    _updateSelectedRunsHint() {
        const el = document.getElementById('hist-selected-runs');
        if (!el) return;
        const selected = this._state.historySelectedRuns || [];
        const selectedNames = selected.map((k) => this._runByKey(k)?.name || k);
        el.textContent = selected.length
            ? `Comparison selection (${selected.length}/5): ${selectedNames.join(', ')}`
            : 'Comparison selection: none';
    },

    /**
     * Collapses or shows the history table panel dynamically.
     * @returns {void}
     */
    _toggleHistoryTable() {
        const wrap = document.getElementById('hist-runs-table-wrap');
        const btn = document.getElementById('hist-table-toggle');
        if (!wrap || !btn) return;
        const hidden = wrap.classList.toggle('hidden');
        btn.textContent = hidden ? 'Show Runs Table' : 'Hide Runs Table';

        if (hidden && this._state.histDeleteMode) {
            this._state.histDeleteMode = false;
            wrap.classList.remove('hist-delete-mode');
            document.getElementById('hist-delete-btn')?.classList.remove('delete-mode-active');
            this._populateRunsTable(this._sortedRuns());
            this._showToast('Delete mode reset because runs table was collapsed', 'info');
        }
    },
};
