/**
 * jobs.module.js — App mixin for parsing and rendering backend system job queues.
 *
 * Provides methods for identifying live jobs, parsing progress payloads,
 * rendering the UI tables, and handling inline job actions (stop/delete).
 */

window.AppJobsModule = {
    /**
     * Determines if a job entity represents an active process (running or queued).
     * Compares the expected Job Manager queue/status fields returned by the backend.
     * 
     * @param {Object} job - System job response object.
     * @returns {boolean} True if the job is structurally active/pending.
     */
    _isActiveJob(job) {
        if (!job || typeof job !== 'object') return false;

        const kind = String(job.kind || '').toLowerCase();
        if (['running', 'queued', 'pending'].includes(kind)) return true;
        if (['done', 'finished', 'completed', 'failed', 'cancelled', 'canceled', 'stopped', 'error'].includes(kind)) return false;

        const status = String(job.status || '').toLowerCase();
        if (['running', 'queued', 'pending', 'active', 'in_progress', 'processing'].includes(status)) return true;
        if (['done', 'finished', 'completed', 'failed', 'cancelled', 'canceled', 'stopped', 'terminated', 'success', 'error'].includes(status)) return false;

        const position = Number(job.position);
        if (Number.isFinite(position) && position >= 0) return true;

        return false;
    },

    /**
     * Configures the total rows per pagination block within the Jobs table UI.
     * @returns {number} Results chunk limit.
     */
    _jobsPageSize() {
        return 10;
    },

    /**
     * Transitions the Jobs queue list table to a specific validated sequence page block.
     * @param {number|string} page - Target page representation.
     */
    _setJobsPage(page) {
        this._state.jobsPage = Math.max(1, Number(page) || 1);
        this._renderJobsQueue(this._state.lastJobsPayload || []);
    },

    /**
     * Emits structurally formatted navigation nodes representing backward/forward pagination progression.
     * @param {number} page - Active page.
     * @param {number} pages - Total resolved pages boundaries.
     */
    _renderJobsPager(page, pages) {
        const el = document.getElementById('jobs-pagination');
        if (!el) return;
        if ((pages || 1) <= 1) {
            el.innerHTML = '';
            return;
        }
        const prev = Math.max(1, page - 1);
        const next = Math.min(pages, page + 1);
        el.innerHTML = `
            <button class="hist-page-btn" ${page <= 1 ? 'disabled' : ''} data-jobs-page="${prev}">Prev</button>
            <span class="hist-page-label">Page ${page} / ${pages}</span>
            <button class="hist-page-btn" ${page >= pages ? 'disabled' : ''} data-jobs-page="${next}">Next</button>
        `;
    },

    /**
     * Render queue cards for active/running jobs.
     * Locked rows are masked for non-admin users when the job belongs to others.
     * @param {Array<Object>} jobs Active jobs payload from backend.
     * @returns {void}
     */
    _renderJobsQueue(jobs) {
        const box = document.getElementById('jobs-queue-list');
        const pagerEl = document.getElementById('jobs-pagination');
        if (!box) return;

        const list = (Array.isArray(jobs) ? jobs : []).filter((j) => this._isActiveJob(j));
        this._state.lastJobsPayload = list;
        this._state.jobsIndex = {};

        if (!list.length) {
            box.innerHTML = '<div class="ui-empty-state">No running or pending jobs.</div>';
            if (pagerEl) pagerEl.innerHTML = '';
            return;
        }

        const pageSize = this._jobsPageSize();
        const totalPages = Math.max(1, Math.ceil(list.length / pageSize));
        const page = Math.min(Math.max(1, this._state.jobsPage || 1), totalPages);
        this._state.jobsPage = page;
        const start = (page - 1) * pageSize;
        const pageItems = list.slice(start, start + pageSize);

        this._renderJobsPager(page, totalPages);

        box.innerHTML = pageItems.map((j, idx) => {
            const key = j.run_id || j.job_id || `job_${idx}`;
            this._state.jobsIndex[key] = j;
            const status = j.kind === 'running' ? 'running' : `pending #${j.position ?? '-'}`;
            if (j.locked) {
                const maskedId = j.job_id || j.run_id || key;
                const pctRaw = j?.summary?.progress_pct;
                const pct = Number.isFinite(Number(pctRaw))
                    ? Math.max(0, Math.min(100, Number(pctRaw)))
                    : (j.kind === 'running' ? 35 : 5);
                return `<div class="border border-slate-700/40 rounded-lg p-3 mb-2 bg-slate-900/10">
                    <div class="text-xs font-semibold text-slate-400">${status}</div>
                    <div class="text-sm text-slate-300 mt-1">${maskedId}</div>
                    <div class="mt-2">
                        <div class="flex items-center justify-between text-[11px] text-slate-500 mb-1">
                            <span>Completion</span>
                            <span>${pct.toFixed(1)}%</span>
                        </div>
                        <div class="progress-track" style="height:6px">
                            <div class="progress-fill" style="width:${pct}%;background:linear-gradient(90deg,#64748b,#334155)"></div>
                        </div>
                    </div>
                </div>`;
            }

            const summary = j.summary || {};
            const runId = j.run_id || '';
            const jobId = j.job_id || '';
            const viewId = runId || jobId;
            const pct = typeof summary.progress_pct === 'number' ? Math.max(0, Math.min(100, summary.progress_pct)) : null;

            const progressRow = j.kind === 'running' && pct != null
                ? `<div class="mt-2">
                    <div class="flex items-center justify-between text-[11px] text-slate-500 mb-1">
                        <span>Completion</span>
                        <span>${pct.toFixed(1)}%</span>
                    </div>
                    <div class="progress-track" style="height:6px">
                        <div class="progress-fill" style="width:${pct}%;background:linear-gradient(90deg,#0f766e,#1d4ed8)"></div>
                    </div>
                </div>`
                : '';

            const actions = j.kind === 'running'
                ? `<button data-job-action="stop" data-job-id="${runId || viewId}" class="ui-btn-compact ui-btn-xs ui-btn-danger">Stop</button>`
                : `<button data-job-action="delete" data-job-id="${jobId || viewId}" class="ui-btn-compact ui-btn-xs ui-btn-danger">Delete</button>`;

            return `<div class="border border-slate-700/40 rounded-lg p-3 mb-2 bg-slate-900/10">
                <div class="jobs-row">
                    <div>
                        <div class="text-xs font-semibold text-slate-400">${status}</div>
                        <div class="text-sm text-slate-300 mt-1">${viewId}</div>
                        <div class="text-xs text-slate-500 mt-1">Method: ${summary.method || '-'} | Dataset: ${summary.dataset_name || '-'}</div>
                        ${progressRow}
                    </div>
                    <div class="jobs-actions">
                        <button data-job-action="view" data-job-id="${viewId}" class="ui-btn-compact ui-btn-xs ui-btn-neutral">Details</button>
                        ${actions}
                    </div>
                </div>
            </div>`;
        }).join('');
    },

    /**
     * Handle click actions inside queue cards (view, stop, delete).
     * @param {MouseEvent} event Delegated click event.
     * @returns {Promise<void>}
     */
    async _onJobsPanelClick(event) {
        const btn = event.target.closest('[data-job-action]');
        if (!btn) return;
        const action = btn.getAttribute('data-job-action');
        const jobId = String(btn.getAttribute('data-job-id') || '').trim();
        const item = this._state.jobsIndex?.[jobId]
            || Object.values(this._state.jobsIndex || {}).find((j) => {
                const runId = String(j?.run_id || '').trim();
                const queuedId = String(j?.job_id || '').trim();
                return jobId && (jobId === runId || jobId === queuedId);
            });
        if (!item || item.locked) return;

        if (action === 'view') {
            this._state.myJobConfig = item.config || {};
            this._openMyJobModal();
            return;
        }

        if (action === 'stop') {
            const runId = String(item.run_id || jobId).trim();
            const rawJobId = String(item.job_id || '').trim();
            const ok = await this._confirmDialog(`Stop running job ${runId}?`);
            if (!ok) return;
            try {
                await api.stopRun(runId, rawJobId);
                this._showToast(`Stopped: ${runId}`, 'warn');
                this._pollSystem();
            } catch (e) {
                this._showToast(e.response?.data?.detail || 'Failed to stop job', 'error');
            }
            return;
        }

        if (action === 'delete') {
            const queuedId = String(item.job_id || item.run_id || jobId).trim();
            const ok = await this._confirmDialog(`Delete queued job ${queuedId}?`);
            if (!ok) return;
            try {
                await api.cancelQueuedJob(queuedId);
                this._showToast(`Deleted queued job: ${queuedId}`, 'success');
                this._pollSystem();
            } catch (e) {
                this._showToast(e.response?.data?.detail || 'Failed to delete queued job', 'error');
            }
        }
    }
};
