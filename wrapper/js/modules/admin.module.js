window.AppAdminModule = {
    /** @returns {number} Constants pagination targets */
    _adminPageSize() {
        return 10;
    },

    /**
     * Updates the current page for the admin jobs table.
     * @param {number|string} page Requested page number.
     * @returns {void}
     */
    _setAdminJobsPage(page) {
        this._state.adminJobsPage = Math.max(1, Number(page) || 1);
        this._refreshAdminOverview({ silent: true, showLoading: false });
    },

    /**
     * Toggle loading state for the admin analytics panel.
     * @param {boolean} loading Whether the overview is currently loading.
     * @param {string} [message] Optional status text shown near the loading bar.
     */
    _setAdminLoading(loading, message = '') {
        const wrap = document.getElementById('admin-loading-wrap');
        const text = document.getElementById('admin-loading-text');
        const refreshBtn = document.getElementById('admin-analytics-refresh');

        if (wrap) {
            wrap.classList.toggle('hidden', !loading);
            wrap.setAttribute('aria-busy', loading ? 'true' : 'false');
        }
        if (text && message) {
            text.textContent = message;
        }
        if (refreshBtn) {
            refreshBtn.disabled = !!loading;
        }
    },

    /**
     * Validates Chart creation logic overrides logic structures properties models frameworks variables fields format references structures.
     * @param {string} chartKey - Storage format variables identifiers references mappings components structures.
     * @param {string} canvasId - Element mapping framework component.
     * @param {Object} config - Representation mappings variables formatting arrays references mappings metrics targets settings targets.
     */
    _upsertAdminChart(chartKey, canvasId, config) {
        if (typeof Chart === 'undefined') return;
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        this._state.adminCharts = this._state.adminCharts || {};
        const noAnim = {
            animation: false,
            transitions: {
                active: { animation: { duration: 0 } },
                resize: { animation: { duration: 0 } },
                show: { animation: { duration: 0 } },
                hide: { animation: { duration: 0 } },
            },
        };
        const finalConfig = {
            ...config,
            options: {
                ...(config?.options || {}),
                ...noAnim,
            },
        };

        const prev = this._state.adminCharts[chartKey];
        if (prev && prev.config?.type === finalConfig.type) {
            prev.data = finalConfig.data;
            prev.options = finalConfig.options;
            prev.update('none');
            return;
        }

        try { prev?.destroy?.(); } catch { }
        this._state.adminCharts[chartKey] = new Chart(canvas, finalConfig);
    },

    /**
     * Orchestrates validation logic rendering mapped components array elements variables representations properties framework strings limits metrics formatting schemas schemas validation implementations representations formats settings updates mappings definitions format representations logic frameworks structures targets structures layouts logic.
     * @param {Object} analytics - Analytics array dictionary string format.
     */
    _renderAdminCharts(analytics) {
        if (typeof Chart === 'undefined') return;

        const safe = analytics || {};
        const status = safe.runs_by_status || { labels: [], values: [] };
        const users = safe.top_users || { labels: [], values: [] };
        const methods = safe.top_methods || { labels: [], values: [] };
        const durations = safe.run_duration_buckets || { labels: [], values: [] };
        const pendingByMethod = safe.pending_by_method || { labels: [], values: [] };
        const resource = safe.resource_usage || {};

        this._upsertAdminChart('status', 'admin-status-chart', {
            type: 'doughnut',
            data: {
                labels: status.labels || [],
                datasets: [{
                    label: 'Runs',
                    data: status.values || [],
                    backgroundColor: ['#2563eb', '#0f766e', '#ea580c', '#dc2626', '#7c3aed', '#475569'],
                }],
            },
            options: { responsive: true, maintainAspectRatio: false },
        });

        this._upsertAdminChart('users', 'admin-users-chart', {
            type: 'bar',
            data: {
                labels: users.labels || [],
                datasets: [{
                    label: 'Runs',
                    data: users.values || [],
                    backgroundColor: '#14b8a6',
                }],
            },
            options: { responsive: true, maintainAspectRatio: false },
        });

        this._upsertAdminChart('methods', 'admin-methods-chart', {
            type: 'bar',
            data: {
                labels: methods.labels || [],
                datasets: [{
                    label: 'Runs',
                    data: methods.values || [],
                    backgroundColor: '#3b82f6',
                }],
            },
            options: { responsive: true, maintainAspectRatio: false },
        });

        this._upsertAdminChart('duration', 'admin-duration-chart', {
            type: 'bar',
            data: {
                labels: durations.labels || [],
                datasets: [{
                    label: 'Runs',
                    data: durations.values || [],
                    backgroundColor: '#f59e0b',
                }],
            },
            options: { responsive: true, maintainAspectRatio: false },
        });

        this._upsertAdminChart('queue', 'admin-queue-chart', {
            type: 'bar',
            data: {
                labels: pendingByMethod.labels || [],
                datasets: [{
                    label: 'Pending jobs',
                    data: pendingByMethod.values || [],
                    backgroundColor: '#0ea5e9',
                }],
            },
            options: { responsive: true, maintainAspectRatio: false },
        });

        this._upsertAdminChart('resource', 'admin-resource-chart', {
            type: 'bar',
            data: {
                labels: ['CPU %', 'RAM %', 'GPU load %'],
                datasets: [{
                    label: 'Average usage',
                    data: [
                        Number(resource.cpu_avg || 0),
                        Number(resource.ram_avg || 0),
                        Number(resource.gpu_avg || 0),
                    ],
                    backgroundColor: ['#2563eb', '#0f766e', '#a855f7'],
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 0, max: 100 },
                },
            },
        });
    },

    /**
     * Translates arrays framework targets fields constraints validation models limits structures representations settings variables frameworks representations models fields strings lists format arrays formats strings.
     */
    _renderAdminJobsTable() {
        const jobsBody = document.getElementById('admin-jobs-body');
        if (!jobsBody) return;

        const pagination = this._state.adminJobsPagination || {
            page: Math.max(1, Number(this._state.adminJobsPage) || 1),
            pages: 1,
            total: Array.isArray(this._state.adminJobsCache) ? this._state.adminJobsCache.length : 0,
        };
        this._state.adminJobsPage = Math.max(1, Number(pagination.page) || 1);
        this._renderPager('admin-jobs-pagination', this._state.adminJobsPage, Math.max(1, Number(pagination.pages) || 1), '_setAdminJobsPage');

        const jobs = this._state.adminJobsCache || [];
        jobsBody.innerHTML = jobs.length
            ? jobs.map(j => {
                const id = j.run_id || j.job_id || '-';
                const jobId = j.job_id || '';
                const method = j.summary?.method || '-';
                const action = j.kind === 'running'
                    ? `<button data-admin-stop-run="${this._escapeHtml(id)}" data-admin-stop-job-id="${this._escapeHtml(jobId)}" class="ui-btn-compact ui-btn-xs ui-btn-danger">Stop</button>`
                    : '<span class="text-xs text-slate-500">-</span>';
                return `
                    <tr class="border-t border-slate-700/40">
                        <td class="text-slate-400 text-xs">${this._escapeHtml(j.kind || '-')}</td>
                        <td class="font-mono text-xs text-slate-300">${this._escapeHtml(id)}</td>
                        <td class="text-slate-400 text-xs">${this._escapeHtml(j.owner || '-')}</td>
                        <td class="text-slate-400 text-xs">${this._escapeHtml(method)}</td>
                        <td class="text-slate-300 text-xs">${action}</td>
                    </tr>
                `;
            }).join('')
            : '<tr><td colspan="5" class="px-4 py-4"><div class="ui-empty-state text-center">No running or pending jobs.</div></td></tr>';
    },

    /**
     * Executes GET references formats arrays mappings framework structures elements validation fields representations variables constraints layouts bindings targets logic lists definitions constraints representations variables format limits models settings validation.
     */
    async _refreshAdminOverview(options = {}) {
        if (!this._state.isAdmin) return;
        if (this._state.adminOverviewLoading) return;

        const silent = !!options?.silent;
        const showLoading = !silent && (options?.showLoading !== false);
        const setText = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = String(value);
        };

        let data;
        this._state.adminOverviewLoading = true;
        if (showLoading) {
            this._setAdminLoading(true, 'Loading admin analytics...');
        }
        try {
            const searchQuery = String(this._state.adminAnalyticsSearchQuery || '').trim().toLowerCase();
            const searchField = String(this._state.adminAnalyticsSearchField || 'all');
            data = await api.adminPlatformOverview(
                searchQuery,
                searchField,
                this._state.adminJobsPage,
                this._adminPageSize(),
            );
            if (showLoading) {
                this._setAdminLoading(true, 'Rendering dashboard charts...');
            }
        } catch (e) {
            if (!silent) {
                this._showToast(e.response?.data?.detail || e.message || 'Failed to load admin overview', 'error');
            }
            this._state.adminOverviewLoading = false;
            if (showLoading) {
                this._setAdminLoading(false);
            }
            return;
        }

        const successRatio = Number(data.analytics?.run_outcomes?.success_ratio || 0);
        const queuePressure = Number(data.analytics?.queue_kpis?.queue_pressure || 0);

        setText('admin-total-users', data.totals?.users ?? '-');
        setText('admin-total-runs', data.totals?.runs ?? '-');
        setText('admin-running-jobs', data.totals?.running_jobs ?? '-');
        setText('admin-pending-jobs', data.totals?.pending_jobs ?? '-');
        setText('admin-success-ratio', `${successRatio.toFixed(1)}% | Queue ${queuePressure.toFixed(1)}%`);

        const scopeEl = document.getElementById('admin-analytics-scope');
        if (scopeEl) {
            const sampled = Number(data.analytics?.sampled_runs || 0);
            const resources = Number(data.analytics?.resource_scanned_runs || 0);
            scopeEl.textContent = `Sampled runs: ${sampled} | Resource scan: ${resources}`;
        }

        this._state.adminJobsCache = data.jobs || [];
        this._state.adminJobsPagination = data.jobs_pagination || {
            page: 1,
            page_size: this._adminPageSize(),
            pages: 1,
            total: this._state.adminJobsCache.length,
        };
        this._state.adminJobsPage = Math.max(1, Number(this._state.adminJobsPagination.page) || 1);
        this._state.adminAnalyticsPayload = data.analytics || {};
        this._state.adminOverviewLastRefreshTs = Date.now();

        try {
            this._renderAdminCharts(this._state.adminAnalyticsPayload);
            this._renderAdminJobsTable();
        } catch (e) {
            console.warn('Admin overview render error:', e);
        } finally {
            this._state.adminOverviewLoading = false;
            if (showLoading) {
                this._setAdminLoading(false);
            }
        }
    },

    /**
     * Debounces analytics refresh while typing in search controls.
     * @returns {void}
     */
    _scheduleAdminOverviewRefresh() {
        clearTimeout(this._state.adminSearchDebounceTimer);
        this._state.adminSearchDebounceTimer = setTimeout(() => {
            this._refreshAdminOverview({ silent: false, showLoading: true });
        }, 350);
    },

    /**
     * Starts background polling for admin analytics while the admin tab is open.
     * @returns {void}
     */
    _startAdminOverviewPolling() {
        if (this._state.adminOverviewTimer) return;
        this._state.adminOverviewTimer = setInterval(() => {
            if (!this._state.isAdmin || this._state.tab !== 5) return;
            this._refreshAdminOverview({ silent: true, showLoading: false });
        }, 20000);
    },

    /**
     * Stops background polling for admin analytics.
     * @returns {void}
     */
    _stopAdminOverviewPolling() {
        clearInterval(this._state.adminOverviewTimer);
        this._state.adminOverviewTimer = null;
        clearTimeout(this._state.adminSearchDebounceTimer);
        this._state.adminSearchDebounceTimer = null;
    },

    /**
     * Execution mapped array formats dictionary lists variables elements structures references mappings frameworks strings models format validations updates parameters schemas definition properties constraints formatting logics requirements fields references targets elements arrays formats constraints fields configurations settings updates variables formats validations updates logic objects.
     */
    async _refreshAdminUsers() {
        if (!this._state.isAdmin) return;
        try {
            const users = await api.adminListUsers();
            const body = document.getElementById('admin-users-body');
            if (!body) return;
            body.innerHTML = users.length
                ? users.map(u => `
                    <tr class="border-t border-slate-700/40">
                        <td class="px-4 py-3 text-slate-300 text-xs">${this._escapeHtml(u.identifier || '-')}</td>
                        <td class="px-4 py-3 text-slate-400 text-xs">${this._escapeHtml(u.email || '-')}</td>
                        <td class="px-4 py-3 text-slate-400 text-xs">${this._escapeHtml(u.role || 'user')}</td>
                        <td class="px-4 py-3 text-slate-400 text-xs">${this._escapeHtml(u.created_by || '-')}</td>
                        <td class="px-4 py-3 text-slate-400 text-xs">${this._escapeHtml(u.created_at || '-')}</td>
                        <td class="px-4 py-3 text-slate-400 text-xs align-middle">
                            ${String(u.role || 'user') === 'admin'
                        ? '<span>protected admin</span>'
                        : `<div style="display: flex; justify-content: flex-start; align-items: center; gap: 8px; width: 100%; white-space: nowrap;">
                                    <button data-admin-promote-user="${this._escapeHtml(u.identifier || '')}" class="admin-action-btn admin-promote-btn">Promote</button>
                                    <button data-admin-delete-user="${this._escapeHtml(u.identifier || '')}" class="admin-action-btn admin-delete-btn">Delete</button>
                                </div>`}
                        </td>
                    </tr>
                `).join('')
                : '<tr><td colspan="6" class="px-4 py-4 text-center text-slate-500 text-sm">No users found.</td></tr>';
        } catch (e) {
            this._showToast(e.response?.data?.detail || 'Failed to load users', 'error');
        }
    },

    /**
     * Creates a new user from the admin user-management form.
     * @returns {Promise<void>}
     */
    async _handleAdminCreateUser() {
        const identifier = String(document.getElementById('admin-new-user-identifier')?.value || '').trim();
        const email = String(document.getElementById('admin-new-user-email')?.value || '').trim().toLowerCase();
        const password = String(document.getElementById('admin-new-user-password')?.value || '');
        const role = String(document.getElementById('admin-new-user-role')?.value || 'user');
        const msg = document.getElementById('admin-create-user-msg');
        if (!identifier) {
            this._showMsg(msg, 'Identifier is required', 'error');
            return;
        }
        if (!email) {
            this._showMsg(msg, 'Email is required', 'error');
            return;
        }
        if (!password) {
            this._showMsg(msg, 'Password is required', 'error');
            return;
        }
        const pwdErrors = this._passwordPolicyErrors(password);
        if (pwdErrors.length) {
            this._showMsg(msg, `${this._passwordPolicyText()} Missing: ${pwdErrors.join(', ')}.`, 'error');
            return;
        }
        try {
            const res = await api.adminCreateUser(identifier, email, password, role);
            this._showMsg(msg, `${res.message || 'User created'}`, 'success');
            document.getElementById('admin-new-user-identifier').value = '';
            document.getElementById('admin-new-user-email').value = '';
            document.getElementById('admin-new-user-password').value = '';
            this._refreshAdminUsers();
            this._refreshAdminOverview();
        } catch (e) {
            this._showMsg(msg, e.response?.data?.detail || 'User creation failed', 'error');
        }
    },

    /**
     * Bind variables variables definitions configurations parameters formats schemas schemas definitions schemas implementations.
     * @param {string} identifier - Validation username formats strings implementations variables variables strings format formulations models fields requirements structures representations representation schemas mappings bindings formats representations formats fields framework structure format constraints format models logic fields objects templates schemas mapping constraints parameters representations formulations.
     */
    async _handleAdminPromoteUser(identifier) {
        const ok = await this._confirmDialog(`Promote user ${identifier} to admin?`);
        if (!ok) return;
        try {
            await api.adminSetUserRole(identifier, 'admin');
            this._showToast(`User promoted: ${identifier}`, 'success');
            this._refreshAdminUsers();
            this._refreshAdminOverview();
        } catch (e) {
            this._showToast(e.response?.data?.detail || 'Promotion failed', 'error');
        }
    },

    /**
     * Structures fields targets fields formats strings format components configurations mapping definitions schemas fields arrays strings models validations strings formulations configuration format logic components properties bindings settings properties structures settings representation limits components bindings structures definitions formats formats setups variables validations.
     * @param {string} identifier - String matched validation properties formats variables formats target parameters references updates targets components fields lists schemas framework parameters dependencies formats matrices mapping algorithms implementations logic references formats limits targets bindings.
     */
    async _handleAdminDeleteUser(identifier) {
        const ok = await this._confirmDialog(`Delete user ${identifier}? This cannot be undone.`);
        if (!ok) return;
        try {
            await api.adminDeleteUser(identifier);
            this._showToast(`User deleted: ${identifier}`, 'success');
            this._refreshAdminUsers();
            this._refreshAdminOverview();
        } catch (e) {
            this._showToast(e.response?.data?.detail || 'Delete failed', 'error');
        }
    },

    /**
     * Initializes elements arrays structures sequences metrics limits attributes limitations structures constraints mappings models templates variables properties updates frameworks limitations parameters strings context schemas implementations mappings targets vectors definitions schemas metrics limits implementations lists frameworks components algorithms variables vectors representations components rules components properties.
     */
    _bindAdminEvents() {
        document.getElementById('admin-analytics-refresh')?.addEventListener('click', () => {
            this._refreshAdminOverview({ silent: false, showLoading: true });
        });
        document.getElementById('admin-analytics-search')?.addEventListener('input', (e) => {
            this._state.adminAnalyticsSearchQuery = String(e.target?.value || '').trim().toLowerCase();
            this._state.adminJobsPage = 1;
            this._scheduleAdminOverviewRefresh();
        });
        document.getElementById('admin-analytics-search-field')?.addEventListener('change', (e) => {
            this._state.adminAnalyticsSearchField = String(e.target?.value || 'all');
            this._state.adminJobsPage = 1;
            this._state.adminJobsPagination = null;
            this._scheduleAdminOverviewRefresh();
        });

        document.getElementById('admin-create-user-form')?.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this._handleAdminCreateUser();
        });

        document.getElementById('admin-users-body')?.addEventListener('click', async (e) => {
            const promoteBtn = e.target.closest('[data-admin-promote-user]');
            if (promoteBtn) {
                const identifier = String(promoteBtn.getAttribute('data-admin-promote-user') || '');
                if (identifier) await this._handleAdminPromoteUser(identifier);
                return;
            }
            const deleteBtn = e.target.closest('[data-admin-delete-user]');
            if (deleteBtn) {
                const identifier = String(deleteBtn.getAttribute('data-admin-delete-user') || '');
                if (identifier) await this._handleAdminDeleteUser(identifier);
            }
        });

        document.getElementById('admin-jobs-body')?.addEventListener('click', async (e) => {
            const btn = e.target.closest('[data-admin-stop-run]');
            if (!btn) return;
            const runId = btn.getAttribute('data-admin-stop-run') || '';
            const jobId = btn.getAttribute('data-admin-stop-job-id') || '';
            if (!runId) return;
            const ok = await this._confirmDialog(`Stop run ${runId}?`);
            if (!ok) return;
            try {
                await api.stopRun(runId, jobId);
                this._showToast(`Stopped: ${runId}`, 'warn');
                this._refreshAdminOverview();
                this._pollSystem();
            } catch (err) {
                this._showToast(err.response?.data?.detail || 'Stop failed', 'error');
            }
        });

        document.getElementById('admin-jobs-pagination')?.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-page-setter][data-page-value]');
            if (!btn || btn.disabled) return;
            const setterName = String(btn.getAttribute('data-page-setter') || '').trim();
            const page = Number(btn.getAttribute('data-page-value') || 1);
            const handler = setterName && typeof this[setterName] === 'function' ? this[setterName] : null;
            if (!handler) return;
            handler.call(this, page);
        });
    }
};
