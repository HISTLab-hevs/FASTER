/**
 * runtime.module.js — App mixin for centralizing global system runtime and active session handling.
 *
 * Interfaces with background processes polling hardware metrics, alerts, and 
 * handles active-job delegation when queuing schedules tasks automatically.
 */

window.AppRuntimeModule = {
    /**
     * Scrolls dashboard main container to top.
     * @returns {void}
     */
    _scrollDashboardTop() {
        const main = document.querySelector('.app-shell main');
        if (!main) return;
        main.scrollTo({ top: 0, behavior: 'auto' });
    },

    /**
     * Build a lightweight signature of runtime state used to detect queue changes.
     * @param {Object} status Latest system-status payload.
     * @returns {string} Deterministic signature string for change detection.
     */
    _analyticsSystemSignature(status) {
        const jobs = Array.isArray(status?.jobs) ? status.jobs : [];
        const parts = jobs
            .map((j) => `${j.kind || ''}:${j.run_id || j.job_id || ''}:${j.owner || ''}`)
            .sort();
        const activeRun = String(status?.user_job?.active?.run_id || '');
        parts.push(`active:${activeRun}`);
        return parts.join('|');
    },

    /**
     * Refresh admin overview when runtime queue/activity changes are detected.
     * @param {Object} status Latest system-status payload.
     * @returns {void}
     */
    _maybeRefreshAdminOverviewFromSystem(status) {
        if (!this._state.isAdmin || this._state.tab !== 5) return;
        const sig = this._analyticsSystemSignature(status);
        const changed = sig !== String(this._state.adminOverviewLastSignature || '');
        const neverLoaded = !this._state.adminAnalyticsPayload;
        if (!changed && !neverLoaded) return;

        this._state.adminOverviewLastSignature = sig;
        this._refreshAdminOverview?.();
    },

    /**
     * Syncs monitor state from system-status payload when backend reports an active user run.
     * This handles the handover case where one run finishes and another queued run starts automatically.
     * @param {Record<string, *>} status - System status payload containing 'user_job' mappings.
     * @returns {void}
     */
    _syncRunFromSystemStatus(status) {
        const active = status?.user_job?.active;
        const activeRunId = active?.run_id || null;
        if (!activeRunId) return;

        const stopGuardRunId = String(this._state.monitorStopGuardRunId || '').trim();
        if (stopGuardRunId && activeRunId === stopGuardRunId) {
            return;
        }
        if (stopGuardRunId && activeRunId !== stopGuardRunId) {
            this._state.monitorStopGuardRunId = null;
        }

        const alreadyMonitoring =
            this._state.currentRunId === activeRunId &&
            this._state.trainingStatus === 'running';
        if (alreadyMonitoring) return;

        this._stopPolling();
        this._state.currentRunId = activeRunId;
        this._state.currentConfig = active?.config || null;
        this._state.trainingStatus = 'running';
        this._setTrainingButtons(true);
        this._resetMonitoringView(active?.config || null);
        this._updateClientEpochProgressFromLogs('');
        this._hydrateMonitoringFromRun?.(activeRunId, active?.config || null);

        // Sync monitor state for the newly active run but leave the user on their
        // current tab — they can open the monitor themselves. (Previously this
        // force-switched to the live tab whenever a run finished/handed over.)
        this._startPolling();
        this._showToast(`Monitoring run: ${activeRunId}`, 'info');
    },

    // ── Tab switching ─────────────────────────────────────────────────────────

    /**
     * Switch dashboard tab and run tab-specific initialization hooks.
     * @param {number} n Target tab index.
     * @returns {void}
     */
    _switchTab(n) {
        const maxTab = this._state.isAdmin ? 6 : 4;
        const safeTab = Math.min(Math.max(0, Number(n) || 0), maxTab);
        n = safeTab;
        if (n !== 3) this._stopHistoryAutoRefresh?.();
        if (n !== 5) this._stopAdminOverviewPolling?.();
        if (this._state.customDatasetUpload?.active && this._state.tab !== n) {
            this._requestCancelDatasetUpload?.('Dataset upload canceled because you changed tab.');
        }
        this._state.tab = n;
        try {
            if (Number.isInteger(n) && n >= 0) localStorage.setItem('fl_dashboard_tab', String(n));
        } catch (e) {
            this._reportNonBlockingIssue?.('dashboard tab persistence', e, {
                dedupeMs: 20000,
            });
        }

        // Update nav buttons
        document.querySelectorAll('.nav-tab').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.tab) === n);
        });

        // Resolve slug and update URL
        const allTabs = [...(UI_DATA.navTabs || []), ...(UI_DATA.adminNavTabs || [])];
        const tabData = allTabs.find(t => t[2] === n);
        const slug = tabData ? tabData[3] : '';

        if (slug) {
            const newPath = `/Faster/${slug}`;
            if (window.location.pathname !== newPath) {
                history.pushState({ tab: n }, '', newPath);
            }
        }

        // Update browser tab title to reflect the current section
        const sectionName = tabData ? tabData[1] : '';
        document.title = sectionName ? `FASTER — ${sectionName}` : 'FASTER';

        // Show/hide content
        document.querySelectorAll('.tab-content').forEach(el => {
            el.classList.toggle('hidden', parseInt(el.dataset.tab) !== n);
        });

        // Tab-specific init
        document.getElementById('auth-resource-panel')?.classList.add('hidden');
        if (n === 0) {
            this._renderDefaultDatasetsRows?.();
            this._refreshDatasetsManager?.();
        }
        if (n === 2 && !this._state.liveChartsInit) {
            metricsCharts.initLive();
            this._state.liveChartsInit = true;
        }
        if (n === 2) {
            requestAnimationFrame(() => this._scrollDashboardTop());
        }
        if (n === 3) {
            this._refreshRunsList(true, { silent: false });
            this._startHistoryAutoRefresh?.();
            if (!this._state.histChartsInit) {
                metricsCharts.initHistory();
                this._state.histChartsInit = true;
            }
        }
        if (n === 4) {
            this._startSystemPolling();
        }
        if (n === 5 && this._state.isAdmin) {
            this._refreshAdminOverview?.({ silent: false, showLoading: true });
            this._startAdminOverviewPolling?.();
        }
        if (n === 6 && this._state.isAdmin) {
            this._refreshAdminUsers?.();
        }
    },

    // ── System status ─────────────────────────────────────────────────────────

    /**
     * Start periodic system polling while runtime tab is open.
     * @returns {void}
     */
    _startSystemPolling() {
        if (this._state.systemTimer) return;
        this._pollSystem();
        this._state.systemTimer = setInterval(() => this._pollSystem(), 10000);
    },

    /**
     * Start periodic alert polling.
     * @returns {void}
     */
    _startAlertPolling() {
        if (this._state.alertTimer) return;
        this._pollAlerts();
        this._state.alertTimer = setInterval(() => this._pollAlerts(), 5000);
    },

    /**
     * Poll alerts and synchronize runtime monitor from backend status.
     * @returns {Promise<void>}
     */
    async _pollAlerts() {
        try {
            const alerts = await api.getAlerts();
            (alerts || []).forEach(a => this._showToast(a.message || 'Runtime alert', a.level || 'warn'));

            // Keep monitor aligned with backend scheduler transitions.
            const status = await api.getSystemStatus();
            this._syncRunFromSystemStatus(status);
            if ((this._state.tab || 0) === 3) {
                this._renderJobsQueue(status.jobs || []);
            }
        } catch (e) {
            this._reportNonBlockingIssue?.('runtime alerts polling', e, {
                dedupeMs: 20000,
            });
        }
    },

    /**
     * Poll system metrics, update resource widgets, and refresh queue cards.
     * @returns {Promise<void>}
     */
    async _pollSystem() {
        try {
            const s = await api.getSystemStatus();
            const bar = (id, pct) => { const el = document.getElementById(id); if (el) el.style.width = `${pct}%`; };
            const label = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };

            bar('cpu-bar', s.cpu_percent);
            label('cpu-label', `${s.cpu_percent}% utilisation`);

            bar('ram-bar', s.ram_percent);
            label('ram-label', `${s.ram_used_gb.toFixed(1)} GB / ${s.ram_total_gb.toFixed(0)} GB  (${s.ram_percent}%)`);

            const gpuEl = document.getElementById('gpu-block');
            if (gpuEl) {
                if (s.gpu_info?.available) {
                    gpuEl.innerHTML = s.gpu_info.gpus.map(g => `
                        <div class="mb-3 last:mb-0">
                            <p class="text-xs font-medium text-slate-300 mb-1">GPU ${g.id}: ${g.name}</p>
                            <div class="grid grid-cols-2 gap-2 text-xs text-slate-400">
                                <span>Temp: ${g.temperature}°C</span>
                                <span>Load: ${g.load}%</span>
                                <span>VRAM: ${(g.memory_used_mb / 1024).toFixed(1)} / ${(g.memory_total_mb / 1024).toFixed(1)} GB</span>
                            </div>
                            <div class="progress-track mt-1.5" style="height:4px">
                                <div class="progress-fill" style="width:${g.load}%;background:linear-gradient(90deg,#0d9488,#14b8a6)"></div>
                            </div>
                        </div>`).join('');
                } else {
                    gpuEl.textContent = s.gpu_info?.message || 'No GPU detected';
                }
            }

            this._renderJobsQueue(s.jobs || []);
            this._syncRunFromSystemStatus(s);
            this._maybeRefreshAdminOverviewFromSystem(s);
        } catch (e) {
            this._reportNonBlockingIssue?.('system status polling', e, {
                dedupeMs: 20000,
            });
        }
    },

    // ── Auth expired (called by api._call on 401 Unauthorized) ────────────────────────

    /**
     * Handle expired authentication state and reset UI/session polling.
     * @returns {void}
     */
    onAuthExpired() {
        this._stopAllPolling();
        metricsCharts.manager.destroyAll();
        try {
            Object.values(this._state.adminCharts || {}).forEach((chart) => chart?.destroy?.());
        } catch (e) {
            this._reportNonBlockingIssue?.('admin chart teardown', e, {
                dedupeMs: 20000,
            });
        }
        this._showLanding();
        this._showToast('Session expired — please sign in again', 'warn');
    },

    // ── Toast notifications ───────────────────────────────────────────────────

    /**
     * Show a transient toast notification.
     * @param {string} message Notification text.
     * @param {string} [type='info'] Visual variant (info, success, warn, error).
     * @returns {void}
     */
    _showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;
        const el = document.createElement('div');
        el.className = `toast toast-${type} fade-in`;
        el.textContent = message;
        container.appendChild(el);
        setTimeout(() => el.remove(), 4500);
    },
};
