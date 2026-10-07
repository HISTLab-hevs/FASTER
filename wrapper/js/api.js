/**
 * api.js — SPA API client using a simple REST endpoint.
 *
 * All calls are routed through a single endpoint:
 *   POST /api   { "command": "...", "token": "...", "data": {...} }
 *
 * The backend (`web_backend.app:app`) mounts `POST /api`, which delegates to
 * the handler facade exported by `web_backend.api_handlers`.
 *
 * Exports a single global `api` instance. No external libraries required.
 */

class APIClient {
    /**
     * Initializes the API client and attempts to restore the authentication token from local storage.
     */
    constructor() {
        /** @private {string} */
        this._token = localStorage.getItem('fl_token') || '';
        /** @private {?Function} */
        this._authExpiredHandler = null;
    }

    // ── Token management ────────────────────────────────────────────────────

    /**
     * Sets the authentication token in memory and local storage.
     * @param {string} t - The raw JWT or bearer token.
     */
    setToken(t) { 
        this._token = t; 
        localStorage.setItem('fl_token', t); 
    }

    /**
     * Clears the current authentication token, logging the user out locally.
     */
    clearToken() { 
        this._token = ''; 
        localStorage.removeItem('fl_token'); 
    }

    /**
     * Checks whether a valid authentication token exists locally.
     * @returns {boolean} True if authenticated, false otherwise.
     */
    isAuthenticated() { 
        return !!this._token; 
    }

    /**
     * Registers a callback invoked when the backend reports an expired/invalid session.
     * @param {?Function} handler Callback to invoke on auth expiry.
     * @returns {void}
     */
    setAuthExpiredHandler(handler) {
        this._authExpiredHandler = typeof handler === 'function' ? handler : null;
    }

    // ── Core REST call ───────────────────────────────────────────────────────

    /**
     * Executes a POST request to the internal /api endpoint.
     * Handles generic network errors, HTTP errors, and token expiration.
     * 
     * @private
     * @param {string} command - Valid command strings processed by the backend API handler layer (e.g. 'login', 'get_runs').
     * @param {Object} [data={}] - Command payload/arguments.
     * @returns {Promise<Object>} The JSON payload returned from the server.
     * @throws {Error} Throws annotated Error on network failure or HTTP >= 400.
     */
    async _call(command, data = {}) {
        let res;
        try {
            res = await fetch('/api', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ command, token: this._token, data }),
            });
        } catch (netErr) {
            throw new Error('Cannot reach server: ' + netErr.message);
        }

        if (!res.ok) {
            let detail = `HTTP ${res.status}`;
            try { 
                const j = await res.json(); 
                detail = j.error || j.detail || detail; 
            } catch {}
            const e = new Error(detail);
            e.response = { data: { detail }, status: res.status };
            throw e;
        }

        const result = await res.json();

        if (result.error === 'Unauthorized') {
            this.clearToken();
            this._authExpiredHandler?.();
            const e = new Error('Unauthorized');
            e.status = 401;
            throw e;
        }

        if (result.error) {
            const e = new Error(result.error);
            e.response = { data: { detail: result.error }, status: 400 };
            throw e;
        }

        return result;
    }

    // ── Auth ─────────────────────────────────────────────────────────────────

    /**
     * Authenticate a user and store their token.
     * @param {string} identifier - Username or Email.
     * @param {string} password - User password.
     * @returns {Promise<{token: string, username: string, email: string, role: string, is_admin: boolean}>} User session details.
     */
    async login(identifier, password) {
        const r = await this._call('login', { identifier, password });
        this.setToken(r.token);
        return {
            token: r.token,
            username: r.username,
            email: r.email || '',
            role: r.role || 'user',
            is_admin: !!r.is_admin,
        };
    }

    /**
     * Retrieve the current authenticated user's profile info.
     * @returns {Promise<Object>} User details.
     */
    async me() {
        return this._call('me');
    }

    /**
     * Change user password while fully authenticated.
     * @param {string} oldPassword - The current user password.
     * @param {string} newPassword - The designated new user password.
     * @returns {Promise<Object>} Success response.
     */
    async changePasswordAuthenticated(oldPassword, newPassword, confirmNewPassword) {
        return this._call('change_password_authenticated', {
            old_password: oldPassword,
            new_password: newPassword,
            confirm_new_password: confirmNewPassword,
        });
    }

    // ── Runs ─────────────────────────────────────────────────────────────────

    /**
     * Retrieves all federated learning runs available to the user.
     * @returns {Promise<Array<Object>>} List of run descriptors.
     */
    async getRuns(refreshDb = false) {
        const r = await this._call('get_runs', { refresh_db: !!refreshDb });
        return r.runs || [];
    }

    /**
     * Retrieves filtered runs based on query arguments.
     * @param {string} search - Keyword matching.
     * @param {boolean} allUsers - Request all users (Admins only).
     * @returns {Promise<Array<Object>>} List of matching run descriptors.
     */
    async getRunsFiltered(search = '', allUsers = false, refreshDb = false) {
        const r = await this._call('get_runs', { search, all_users: !!allUsers, refresh_db: !!refreshDb });
        return r.runs || [];
    }

    /**
     * Retrieves one server-filtered, server-paginated runs page for history rendering.
     * @param {Object} options Paged query options.
     * @returns {Promise<Object>} Response payload including runs and pagination metadata.
     */
    async getRunsPage(options = {}) {
        return this._call('get_runs', {
            search: String(options.search || ''),
            search_field: String(options.searchField || 'all'),
            all_users: !!options.allUsers,
            refresh_db: !!options.refreshDb,
            page: Number(options.page || 1),
            page_size: Number(options.pageSize || 10),
            sort_key: String(options.sortKey || 'run_ts'),
            sort_dir: String(options.sortDir || 'desc'),
        });
    }

    /**
     * Retrieves compact run summaries for a specific list of run identifiers.
     * @param {Array<string>} ids Run identifiers to resolve.
     * @returns {Promise<Array<Object>>} Visible run summaries in request order.
     */
    async getRunsByIds(ids = []) {
        const r = await this._call('get_runs_by_ids', { ids: Array.isArray(ids) ? ids : [] });
        return r.runs || [];
    }

    /**
     * Retrieves live status of a single run.
     * @param {string} runId - Unified identifier for the run.
     * @returns {Promise<{status: string, active: boolean, running: boolean, stop_reason: string|null}>}
     */
    async getRunStatus(runId) {
        const r = await this._call('get_run_status', { run_id: runId });
        const status = String(r.status || (r.running ? 'running' : 'unknown'));
        const active = typeof r.active === 'boolean'
            ? r.active
            : ['queued', 'running', 'pending'].includes(status);
        return {
            status,
            active,
            running: !!r.running,
            stop_reason: r.stop_reason || null,
        };
    }

    /**
     * Starts an immediate execution of a training configuration.
     * @param {Object} config - Normalized parameters configuration dictionary.
     * @returns {Promise<Object>} Response detailing initialized run state.
     */
    async startRun(config) {
        return this._call('start_run', { config });
    }

    /**
     * Queues a training configuration for delayed/sequential execution.
     * @param {Object} config - Normalized parameters configuration dictionary.
     * @returns {Promise<Object>} Response detailing enqueued job state.
     */
    async queueRun(config) {
        return this._call('queue_run', { config });
    }

    /**
     * Validate against potential naming clashes across stored runs.
     * @param {Array<string>} names - Batch of configured run names to check.
     * @returns {Promise<Array<string>>} List of names that already exist.
     */
    async validateRunNames(names) {
        const r = await this._call('validate_run_names', { names: Array.isArray(names) ? names : [] });
        return r.conflicts || [];
    }

    /**
     * Send a signal to prematurely abort an executing run.
     * @param {string} runId - Unified identifier for the run.
     * @returns {Promise<Object>} Stop confirmation structure.
     */
    async stopRun(runId, jobId = '') {
        return this._call('stop_run', { run_id: runId, job_id: jobId });
    }

    /**
     * Removes an unmet job sequence from the execution queue.
     * @param {string} jobId - Queue string ID.
     * @returns {Promise<Object>} Action response status.
     */
    async cancelQueuedJob(jobId) {
        return this._call('cancel_queued_job', { job_id: jobId });
    }

    /**
     * Retrieves validation/test metrics extracted from the run.
     * @param {string} runId - Target run ID.
     * @returns {Promise<Object>} Metrics dictionary, populated per epoch/method.
     */
    async getMetrics(runId) {
        const r = await this._call('get_metrics', { run_id: runId });
        return r.metrics || {};
    }

    /**
     * Fetch standard terminal output segments corresponding to an execution run.
     * @param {string} runId - Target run ID.
     * @param {number} n - Number of final tail lines to pull (e.g. 100).
     * @returns {Promise<{logs: string}>} Sequential raw textual log output.
     */
    async getLogs(runId, n = 100) {
        const r = await this._call('get_logs', { run_id: runId, n });
        return { logs: r.logs || '' };
    }

    /**
     * Fetches configuration variables mapped into a particular static run.
     * @param {string} runId - Target run ID.
     * @returns {Promise<Object>} Full parameter mapping config object.
     */
    async getRunConfig(runId) {
        const r = await this._call('get_run_config', { run_id: runId });
        return r.config || {};
    }

    /**
     * Get standalone URL for archiving run components into a .zip.
     * @param {string} runId - Unified identifier for the run.
     * @returns {string} Same-origin export endpoint URL.
     */
    getExportUrl(runId) {
        return `/fl-download/${encodeURIComponent(runId)}.zip`;
    }

    /**
     * Trigger one authenticated ZIP download, preferring Authorization headers
     * over query-string tokens to reduce leakage through URLs.
     * @param {string} runId - Unified identifier for the run.
     * @returns {Promise<void>} Resolves after the browser download has been triggered.
     */
    async downloadRun(runId) {
        const overlay = this._showDownloadOverlay();
        try {
            let res;
            const headers = this._token ? { Authorization: `Bearer ${this._token}` } : {};
            try {
                res = await fetch(this.getExportUrl(runId), {
                    method: 'GET',
                    headers,
                });
            } catch (netErr) {
                throw new Error('Cannot reach server: ' + netErr.message);
            }

            if (!res.ok) {
                let detail = `HTTP ${res.status}`;
                try {
                    const payload = await res.json();
                    detail = payload.error || payload.detail || detail;
                } catch {}

                if (detail === 'Unauthorized' || res.status === 401) {
                    this.clearToken();
                    this._authExpiredHandler?.();
                }

                const e = new Error(detail);
                e.response = { data: { detail }, status: res.status };
                throw e;
            }

            const blob = await res.blob();
            const href = URL.createObjectURL(blob);
            const link = document.createElement('a');
            const dispo = res.headers.get('Content-Disposition') || '';
            const match = dispo.match(/filename=\"?([^\";]+)\"?/i);
            link.href = href;
            link.download = match?.[1] || `${runId}.zip`;
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.setTimeout(() => URL.revokeObjectURL(href), 1000);
        } finally {
            overlay?.remove();
        }
    }

    /**
     * Show a centered, non-dismissible spinner overlay that blocks interaction
     * for the duration of a ZIP download. Returns the element so the caller can
     * remove it once finished (or failed).
     * @returns {?HTMLElement} The overlay element, or null outside the browser.
     */
    _showDownloadOverlay() {
        if (typeof document === 'undefined') return null;
        const overlay = document.createElement('div');
        overlay.className = 'download-overlay';
        overlay.setAttribute('role', 'alert');
        overlay.setAttribute('aria-busy', 'true');
        overlay.innerHTML =
            '<div class="download-overlay__spinner"></div>' +
            '<div class="download-overlay__label">Preparing download…</div>';
        document.body.appendChild(overlay);
        return overlay;
    }

    /**
     * Wire one anchor to perform header-authenticated run export when clicked.
     * A plain href is kept for progressive enhancement, while the preferred path
     * uses Authorization headers to avoid leaking tokens into URLs.
     * @param {?HTMLAnchorElement} linkEl - Anchor element to bind.
     * @param {string} runId - Unified identifier for the run.
     * @param {?Function} onError - Optional async error callback.
     * @returns {void}
     */
    bindExportLink(linkEl, runId, onError = null) {
        if (!linkEl) return;
        linkEl.href = this.getExportUrl(runId);
        linkEl.setAttribute('download', `${runId}.zip`);
        linkEl.onclick = async (event) => {
            event.preventDefault();
            try {
                await this.downloadRun(runId);
            } catch (e) {
                if (typeof onError === 'function') {
                    onError(e);
                    return;
                }
                throw e;
            }
        };
    }

    /**
     * Permanently purges all metrics, configurations, and logs targeting a specific run.
     * @param {string} runId - Unified identifier for the run.
     * @returns {Promise<Object>} Deletion acknowledgment state.
     */
    async deleteRun(runId) {
        return this._call('delete_run', { run_id: runId });
    }

    /**
     * Rename an existing run.
     * @param {string} runId - Unified identifier for the run.
        * @param {string} newName - New name for the run (max 20 chars).
     * @returns {Promise<Object>} Rename acknowledgment state.
     */
    async renameRun(runId, newName) {
        return this._call('rename_run', { run_id: runId, new_name: newName });
    }

    // ── Scenarios ─────────────────────────────────────────────────────────────

    /**
     * List user-saved YAML setup scenarios.
     * @returns {Promise<Array<string>>} Sequence of active configuration scenarios mapped on system.
     */
    async getScenarios() {
        const r = await this._call('list_scenarios');
        return r.scenarios || [];
    }

    /**
     * Read the raw config parameter constraints saved under a particular scenario.
     * @param {string} name - Exact scenario name reference.
     * @returns {Promise<Object>} Scoped setup object fields.
     */
    async getScenario(name) {
        const r = await this._call('get_scenario', { name });
        return r.config || {};
    }

    /**
     * Snapshot an executed/configured parameter map out to a server-side YAML component.
     * @param {Object} config - Mapping field limits.
     * @param {string} name - Base name strings for identifying the resulting config save logic.
     * @returns {Promise<Object>} Response validating successful save execution.
     */
    async saveScenario(config, name = '') {
        return this._call('save_scenario', { config, name });
    }

    /**
     * Delete one user-owned saved scenario.
     * @param {string} name - Scenario reference such as my/demo.yaml.
     * @returns {Promise<Object>} API response.
     */
    async deleteScenario(name) {
        return this._call('delete_scenario', { name });
    }

    /**
     * Convert YAML user payload logic directly onto standard config bindings format.
     * @param {string} yamlText - Full multi-line YAML text representing scenario arguments.
     * @returns {Promise<Object>} Safe dictionary matching valid federated parameters.
     */
    async parseScenarioYaml(yamlText) {
        const r = await this._call('parse_scenario_yaml', { yaml_text: yamlText });
        return r.config || {};
    }

    /**
     * List uploaded custom datasets for current user.
     * @returns {Promise<Array<Object>>} Dataset metadata rows.
     */
    async listCustomDatasets() {
        const r = await this._call('list_custom_datasets');
        return r.datasets || [];
    }

    /**
     * Rename a custom dataset display name by dataset reference.
     * @param {string} datasetRef Stable dataset reference.
     * @param {string} newName New display name.
     * @returns {Promise<Object>} API response.
     */
    async renameCustomDataset(datasetRef, newName) {
        return this._call('rename_custom_dataset', { dataset_ref: datasetRef, new_name: newName });
    }

    /**
     * Update custom dataset description text.
     * @param {string} datasetRef Stable dataset reference.
     * @param {string} description Free-text description.
     * @returns {Promise<Object>} API response.
     */
    async updateCustomDatasetDescription(datasetRef, description) {
        return this._call('update_custom_dataset_description', { dataset_ref: datasetRef, description });
    }

    /**
     * Delete a custom dataset by reference.
     * @param {string} datasetRef Stable dataset reference.
     * @returns {Promise<Object>} API response.
     */
    async deleteCustomDataset(datasetRef) {
        return this._call('delete_custom_dataset', { dataset_ref: datasetRef });
    }

    /**
     * Retrieve Job Manager execution diagnostics.
     * @returns {Promise<Object>} Diagnostic payload.
     */
    async getJobManagerDiagnostics() {
        return this._call('job_manager_diagnostics');
    }

    /**
     * Stream one dataset file through multipart upload to avoid loading full contents into JS strings.
     *
     * @param {File} file - Dataset file selected by the user.
     * @param {Object} options - Upload configuration dictionary.
     * @param {Function} options.onProgress - Triggered with upload ratio updates.
     * @param {AbortSignal} options.signal - Optional AbortController signal.
     * @param {string} options.description - Optional dataset description.
     * @returns {Promise<Object>} Success execution payload.
     */
    uploadCustomDatasetFileWithProgress(file, options = {}) {
        const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
        const onUploadComplete = typeof options.onUploadComplete === 'function' ? options.onUploadComplete : null;
        const signal = options.signal || null;
        const description = String(options.description || '');

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            let settled = false;

            const finalizeReject = (error) => {
                if (settled) return;
                settled = true;
                reject(error);
            };

            const finalizeResolve = (value) => {
                if (settled) return;
                settled = true;
                resolve(value);
            };

            const buildAbortError = () => {
                const e = new Error('Upload canceled');
                e.name = 'AbortError';
                return e;
            };

            xhr.open('POST', '/api/upload-custom-dataset', true);
            if (this._token) xhr.setRequestHeader('Authorization', `Bearer ${this._token}`);

            xhr.upload.onprogress = (evt) => {
                if (!onProgress || !evt.lengthComputable || evt.total <= 0) return;
                onProgress(evt.loaded / evt.total);
            };
            xhr.upload.onload = () => {
                onProgress?.(1);
                onUploadComplete?.();
            };

            xhr.onerror = () => {
                finalizeReject(new Error('Cannot reach server'));
            };

            xhr.onabort = () => {
                finalizeReject(buildAbortError());
            };

            xhr.onload = () => {
                const status = xhr.status || 0;
                let payload;
                try {
                    payload = xhr.responseText ? JSON.parse(xhr.responseText) : {};
                } catch {
                    payload = {};
                }

                if (status < 200 || status >= 300) {
                    const detail = payload.error
                        || payload.detail
                        || (status === 413
                            ? 'Dataset upload is larger than the server limit. Increase the configured upload size limit or use a smaller archive.'
                            : `HTTP ${status}`);
                    const e = new Error(detail);
                    e.response = { data: { detail }, status };
                    finalizeReject(e);
                    return;
                }

                if (payload.error === 'Unauthorized') {
                    this.clearToken();
                    this._authExpiredHandler?.();
                    const e = new Error('Unauthorized');
                    e.status = 401;
                    finalizeReject(e);
                    return;
                }

                if (payload.error) {
                    const e = new Error(payload.error);
                    e.response = { data: { detail: payload.error }, status: 400 };
                    finalizeReject(e);
                    return;
                }

                finalizeResolve(payload);
            };

            if (signal) {
                if (signal.aborted) {
                    xhr.abort();
                    return;
                }
                const onAbort = () => xhr.abort();
                signal.addEventListener('abort', onAbort, { once: true });
                xhr.addEventListener('loadend', () => {
                    try { signal.removeEventListener('abort', onAbort); } catch {}
                }, { once: true });
            }

            try {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('description', description);
                xhr.send(formData);
            } catch (err) {
                finalizeReject(err instanceof Error ? err : new Error('Upload failed'));
            }
        });
    }

    // ── Defaults ──────────────────────────────────────────────────────────────

    /**
     * Ask server for baseline global parameters configuration structures mappings components defined via YAML.
     * @returns {Promise<Object>} Base dictionary formats defaults parameters logic strings.
     */
    async getDefaults() {
        const r = await this._call('get_defaults');
        return r.defaults || {};
    }

    /**
     * Update runtime logic for baseline parameter options mapping across forms layout settings formats.
     * @param {Object} config - Mapping field updates elements representation formats components updates format.
     * @returns {Promise<Object>} Logic responses validation formats confirmation definition components definition variables bindings definitions formats variables format representations. 
     */
    async saveDefaults(config) {
        return this._call('save_defaults', { config });
    }

    // ── System ────────────────────────────────────────────────────────────────

    /**
     * Extract active global memory constraints metrics, CPU bindings and load variables components defined via background task loops arrays structures formats variables formats.
     * @returns {Promise<Object>} Logic references dictionaries definition.
     */
    async getSystemStatus() {
        return this._call('system_status');
    }

    /**
     * Retrieves queued frontend alerts generated by the backend.
     * @returns {Promise<Array<Object>>} Alert objects for non-blocking UI notifications.
     */
    async getAlerts() {
        const r = await this._call('get_alerts');
        return r.alerts || [];
    }

    /**
     * Retrieves the full user list for the admin workspace.
     * @returns {Promise<Array<Object>>} Persisted user records visible to admins.
     */
    async adminListUsers() {
        const r = await this._call('admin_list_users');
        return r.users || [];
    }

    /**
     * Creates a user account from the admin panel.
     * @param {string} identifier - Login identifier for the new account.
     * @param {string} email - User email address.
     * @param {string} password - Initial password.
     * @param {string} role - Either `user` or `admin`.
     * @returns {Promise<Object>} Backend confirmation payload.
     */
    async adminCreateUser(identifier, email, password, role = 'user') {
        return this._call('admin_create_user', { identifier, email, password, role });
    }

    /**
     * Admin execution reference logic target specific account bindings removal variables references.
     * @param {string} identifier - String matched references username variables structures fields formatting.
     * @returns {Promise<Object>} Valid variables mapping response logic implementations validation elements models components.
     */
    async adminDeleteUser(identifier) {
        return this._call('admin_delete_user', { identifier });
    }

    /**
     * Change target role reference format models via specific update bindings bindings targets structures mappings structures formats components formats representations formats structures elements implementations references elements bindings definitions formats mappings bindings elements formats structures fields mappings definitions logic representations forms mappings variables structures mappings mappings references format mappings settings models format bindings structures.
     * @param {string} identifier - Username string reference.
     * @param {string} role - Target string binding assignment properties configurations targets mappings bindings updates definitions structures mappings structures validation logic definition settings implementations.
     * @returns {Promise<Object>} Valid server structure confirmations payload representations mappings constraints formats mapping references fields definitions format representations implementations targets mappings logic elements logic.
     */
    async adminSetUserRole(identifier, role) {
        return this._call('admin_set_user_role', { identifier, role });
    }

    /**
     * Load core aggregate runtime metrics formats configurations implementations components mappings arrays formatting validation elements definitions updates definitions references.
     * @param {string} search - Target term matching validations fields logic templates formats constraints.
     * @param {string} searchField - Key targets definitions.
     * @returns {Promise<Object>} Structured summary configuration elements components representations.
     */
    async adminPlatformOverview(search = '', searchField = 'all', jobsPage = 1, jobsPageSize = 10) {
        return this._call('admin_platform_overview', {
            search,
            search_field: searchField,
            jobs_page: jobsPage,
            jobs_page_size: jobsPageSize,
        });
    }
}

// Global instance mapping logic.
// eslint-disable-next-line no-unused-vars -- consumed by later classic scripts.
const api = new APIClient();
