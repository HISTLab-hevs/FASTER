/**
 * app.js — FederatedLearningApp (Main orchestrator)
 *
 * This file serves as the main entry point and orchestrator for the FASTER SPA.
 * It manages the primary application state, UI routing, authentication 
 * lifecycles, and core admin interactions. 
 *
 * Note: Domain-specific logic is injected into the App prototype via 
 * modules (e.g. AppSetupModule, AppHistoryModule).
 */

const UI_DATA = window.UI_DATA || {
    paramDescriptions: {},
    landingFeatures: [],
    navTabs: [],
    methodOptions: [],
    datasetOptions: [],
    evaluationSplitModeOptions: [
        ['train_val_test', 'Train/Val/Test'],
        ['train_val', 'Train/Val only'],
    ],
    gpInitializationOptions: [],
    mutationTypeOptions: [],
    crossoverTypeOptions: [],
    selectionTypeOptions: [],
    historyChartCards: [],
};

// eslint-disable-next-line no-unused-vars -- consumed by render.module.js as a classic-script global.
const TEMPLATE_VERSION = '20260421_monitor_setup_03';

/**
 * Main application class controlling global state and UI orchestration.
 */
class App {
    /**
     * Initializes global state objects, timers, UI caching indices, and binds routing listeners.
     */
    constructor() {
        let sidebarCollapsed = false;
        try {
            sidebarCollapsed = localStorage.getItem('fl_sidebar_collapsed') === '1';
        } catch { }

        /**
         * Core application state context. 
         * @type {Object}
         */
        this._state = {
            username: null,
            userEmail: '',
            isAdmin: false,
            tab: 0,
            currentRunId: null,
            currentConfig: null,
            lastLiveMetrics: null,
            trainingStatus: 'idle',
            monitorStopGuardRunId: null,
            logTimer: null,
            metricsTimer: null,
            systemTimer: null,
            histRunId: null,
            runsCache: [],
            historyRunLookup: {},
            histRunsPagination: null,
            historySelectedRuns: [],
            histDeleteMode: false,
            liveChartsInit: false,
            histChartsInit: false,
            logCollapsed: false,
            myJobConfig: null,
            historyAutoLoadTimer: null,
            histLoadSeq: 0,
            histTablePage: 1,
            histSearchDebounceTimer: null,
            histComparePage: 1,
            jobsPage: 1,
            customModelFileName: '',
            setupMode: 'simple',
            histDetailCharts: {},
            alertTimer: null,
            logResizeObserver: null,
            editors: {},
            jobsIndex: {},
            scenarioNameResolver: null,
            sidebarCollapsed,
            customDatasetRef: '',
            customDatasetFileName: '',
            customDatasetStatusTimer: null,
            customDatasetUpload: null,
            customDatasetType: 'csv', // 'csv', 'image_npz', or 'image_folder'
            customDatasetFormat: 'csv', // 'csv', 'tsv', 'npz', or 'zip'
            datasetUploadUnloadBound: false,
            datasetsCustomCache: [],
            datasetsSearchField: 'all',
            datasetsSearchQuery: '',
            datasetsSortField: 'dataset_name_asc',
            helpTooltipInit: false,
            helpTooltipEl: null,
            roundTrainScheduleManual: [],
            roundClientAllocationManual: [],
            roundClientAllocationActiveRound: 0,
            histSearchQuery: '',
            histSearchField: 'all',
            adminAnalyticsSearchQuery: '',
            adminAnalyticsSearchField: 'all',
            adminJobsPage: 1,
            adminJobsPagination: null,
            adminCharts: {},
            adminJobsCache: [],
            adminAnalyticsPayload: null,
            adminOverviewLastSignature: '',
            adminOverviewLastRefreshTs: 0,
            adminOverviewTimer: null,
            adminOverviewLoading: false,
            adminSearchDebounceTimer: null,
            dashboardResizeBound: false,
            nonBlockingIssueTimestamps: {},
        };

        this._templateCache = {};
        this._boundPopStateHandler = () => this._onPopState();
        this._confirmResolver = null;
        this._datasetUploadBeforeUnloadBound = false;
    }

    /**
     * Normalize backend-provided custom dataset format metadata.
     * @param {string} rawFormat Precise user-visible format from backend metadata.
     * @param {string} [rawType=''] Backend dataset type/family marker.
     * @returns {'csv'|'tsv'|'npz'|'zip'} Normalized visible format.
     */
    _normalizeCustomDatasetFormat(rawFormat, rawType = '') {
        const fmt = String(rawFormat || '').trim().toLowerCase();
        if (fmt === 'csv' || fmt === 'tsv' || fmt === 'npz' || fmt === 'zip') return fmt;
        const txt = String(rawType || '').trim().toLowerCase();
        if (txt === 'image_npz' || txt === 'custom_image_npz') return 'npz';
        if (txt === 'image_folder' || txt === 'custom_image_folder') return 'zip';
        return 'csv';
    }

    /**
     * Convert one custom dataset format into the execution family used by the form.
     * @param {string} rawFormat Precise format marker.
     * @param {string} [rawType=''] Backend dataset type/family marker.
     * @returns {'csv'|'image_npz'|'image_folder'} Execution family expected by current UI logic.
     */
    _normalizeCustomDatasetType(rawFormat, rawType = '') {
        const fmt = this._normalizeCustomDatasetFormat(rawFormat, rawType);
        if (fmt === 'npz') return 'image_npz';
        if (fmt === 'zip') return 'image_folder';
        return 'csv';
    }

    /**
     * Build the explicit format label shown to the user.
     * @param {string} rawFormat Precise format marker.
     * @param {string} [rawType=''] Backend dataset type/family marker.
     * @returns {string} Visible uppercase label such as CSV, TSV, or NPZ.
     */
    _customDatasetFormatLabel(rawFormat, rawType = '') {
        return this._normalizeCustomDatasetFormat(rawFormat, rawType).toUpperCase();
    }

    /**
     * Boots the application. Evaluates existing session authentication 
     * and routes to either the authenticated dashboard or public landing page.
     * @returns {Promise<void>}
     */
    async boot() {
        api.setAuthExpiredHandler(() => this.onAuthExpired?.());
        window.addEventListener('popstate', this._boundPopStateHandler);

        if (api.isAuthenticated()) {
            try {
                const me = await api.me();
                this._state.username = me.username;
                this._state.userEmail = me.email || '';
                this._state.isAdmin = !!me.is_admin || String(me.role || '') === 'admin';
                await this._showDashboard();
                await this._initDashboard();

                // Select the correct tab from URL if present
                const path = window.location.pathname;
                const parts = path.split('/').filter(Boolean);
                if (parts.length >= 2 && parts[0].toLowerCase() === 'faster') {
                    const slug = parts[1];
                    const allTabs = [...(UI_DATA.navTabs || []), ...(UI_DATA.adminNavTabs || [])];
                    const tabData = allTabs.find(t => t[3] === slug);
                    if (tabData) {
                        this._switchTab(tabData[2]);
                    } else if (['Documentation', 'Creators', 'Contribute'].includes(slug)) {
                        this._openAuthResource(slug);
                    }
                }
            } catch {
                api.clearToken();
                await this._showPublicPage(this._publicRouteFromPath());
            }
        } else {
            await this._showPublicPage(this._publicRouteFromPath());
        }
    }

    // ── Event wiring ─────────────────────────────────────────────────────────

    /**
     * Internal delegation triggering DOM event registrations for landing pages.
     * (Provided by mixin modules)
     * @protected
     * @returns {void}
     */
    _attachLandingEvents() {
        this._attachLandingEventsModular?.();
    }

    /**
     * Internal delegation triggering DOM event registrations for the private dashboard.
     * (Provided by mixin modules)
     * @protected
     * @returns {void}
     */
    _attachDashboardEvents() {
        this._attachDashboardEventsModular?.();
    }

    /**
     * Bootstraps the global mouse-over tooltip popover system for UI tooltips.
     * Only initializes once.
     * @protected
     * @returns {void}
     */
    _initHelpTooltips() {
        if (this._state.helpTooltipInit) return;
        this._state.helpTooltipInit = true;

        const pop = document.createElement('div');
        pop.className = 'help-popover';
        pop.setAttribute('aria-hidden', 'true');
        document.body.appendChild(pop);
        this._state.helpTooltipEl = pop;

        const hide = () => {
            pop.classList.remove('show');
            pop.setAttribute('aria-hidden', 'true');
        };

        const show = (dot) => {
            const text = dot?.getAttribute('data-tooltip') || dot?.getAttribute('aria-label') || '';
            if (!text) return;
            pop.textContent = text;
            pop.setAttribute('aria-hidden', 'false');
            pop.classList.add('show');

            const rect = dot.getBoundingClientRect();
            const vw = window.innerWidth;
            const vh = window.innerHeight;
            const margin = 8;
            const popRect = pop.getBoundingClientRect();

            let top = rect.top - popRect.height - 10;
            if (top < margin) {
                top = rect.bottom + 10;
            }
            if (top + popRect.height > vh - margin) {
                top = Math.max(margin, vh - popRect.height - margin);
            }

            const preferredLeft = rect.left + rect.width / 2 - popRect.width / 2;
            const left = Math.min(Math.max(margin, preferredLeft), vw - popRect.width - margin);

            pop.style.top = `${Math.round(top)}px`;
            pop.style.left = `${Math.round(left)}px`;
        };

        document.addEventListener('mouseenter', (e) => {
            const dot = e.target?.closest?.('.help-dot');
            if (dot) show(dot);
        }, true);
        document.addEventListener('focusin', (e) => {
            const dot = e.target?.closest?.('.help-dot');
            if (dot) show(dot);
        });
        document.addEventListener('mouseleave', (e) => {
            const dot = e.target?.closest?.('.help-dot');
            if (dot) hide();
        }, true);
        document.addEventListener('focusout', (e) => {
            const dot = e.target?.closest?.('.help-dot');
            if (dot) hide();
        });
        window.addEventListener('scroll', hide, true);
        window.addEventListener('resize', hide);
        document.addEventListener('click', (e) => {
            if (!e.target?.closest?.('.help-dot')) hide();
        });
    }

    // ── Auth handlers ─────────────────────────────────────────────────────────

    /**
     * Collects form inputs to execute login via Auth module.
     * Transitions user to dashboard upon success.
     * @returns {Promise<void>}
     */
    async _handleLogin() {
        const u = document.getElementById('login-user').value.trim();
        const p = document.getElementById('login-pass').value;
        const msg = document.getElementById('login-msg');
        if (!u || !p) { this._showMsg(msg, 'Email or username and password are required', 'error'); return; }

        const result = await auth.login(u, p);
        if (result.success) {
            this._state.username = result.user?.username || u;
            this._state.userEmail = result.user?.email || '';
            this._state.isAdmin = !!result.user?.is_admin;
            await this._showDashboard();
            await this._initDashboard();
            this._switchTab(3);
        } else {
            this._showMsg(msg, result.message, 'error');
        }
    }

    /**
     * Executes strict client-side teardown for active session.
     * Clears loop intervals, cached runs, destroys DOM charts, and bounces to Landing.
     * @returns {Promise<void>}
     */
    async _handleLogout() {
        this._stopAllPolling();
        this._teardownLogAutoScroll();
        metricsCharts.manager.destroyAll();
        try { Object.values(this._state.adminCharts || {}).forEach((chart) => chart?.destroy?.()); } catch { }
        auth.logout();

        let sidebarCollapsed = false;
        try { sidebarCollapsed = localStorage.getItem('fl_sidebar_collapsed') === '1'; } catch { }

        // Reset state object
        this._state = {
            username: null, userEmail: '', isAdmin: false, tab: 0, currentRunId: null, currentConfig: null,
            trainingStatus: 'idle', logTimer: null, metricsTimer: null, systemTimer: null,
            histRunId: null, runsCache: [], historyRunLookup: {}, histRunsPagination: null, historySelectedRuns: [], histDeleteMode: false,
            liveChartsInit: false, histChartsInit: false, logCollapsed: false, myJobConfig: null,
            historyAutoLoadTimer: null, histLoadSeq: 0, histSearchDebounceTimer: null, customModelFileName: '', setupMode: 'simple',
            histDetailCharts: {}, alertTimer: null, logResizeObserver: null, editors: {}, jobsIndex: {},
            setupStep: 1, setupSteps: [], scenarioNameResolver: null, sidebarCollapsed,
            customDatasetRef: '', customDatasetFileName: '', customDatasetStatusTimer: null,
            customDatasetUpload: null, customDatasetType: 'csv', customDatasetFormat: 'csv', datasetUploadUnloadBound: false,
            datasetsCustomCache: [],
            datasetsSearchField: 'all', datasetsSearchQuery: '', datasetsSortField: 'dataset_name_asc',
            histSearchQuery: '', histSearchField: 'all',
            adminAnalyticsSearchQuery: '', adminAnalyticsSearchField: 'all',
            adminJobsPage: 1, adminJobsPagination: null, adminCharts: {}, adminJobsCache: [], adminAnalyticsPayload: null,
            adminOverviewLastSignature: '', adminOverviewLastRefreshTs: 0,
            adminOverviewTimer: null, adminOverviewLoading: false, adminSearchDebounceTimer: null,
            dashboardResizeBound: true, nonBlockingIssueTimestamps: {}
        };
        history.pushState(null, '', '/Faster/Signin');
        await this._showLanding();
        this._showToast('Logged out', 'info');
    }

    /**
     * Triggers the UI-blocking custom confirmation dialog overlay.
     * @param {string} message - Text prompting user verification.
     * @returns {Promise<boolean>} True if 'OK' selected.
     */
    _confirmDialog(message) {
        const modal = document.getElementById('confirm-modal');
        const msgEl = document.getElementById('confirm-modal-msg');
        if (!modal || !msgEl) return Promise.resolve(false);
        msgEl.textContent = message;
        modal.classList.remove('hidden');
        return new Promise(resolve => {
            this._confirmResolver = resolve;
        });
    }

    /**
     * Handles programmatic closure/resolving of the active Confirmation Dialog window.
     * @param {boolean} answer - Resolved boolean mapping UI selection.
     */
    _resolveConfirm(answer) {
        const modal = document.getElementById('confirm-modal');
        if (modal) modal.classList.add('hidden');
        if (this._confirmResolver) {
            this._confirmResolver(answer);
            this._confirmResolver = null;
        }
    }

    /**
     * Invokes prompt overlay requesting string identifiers (used for new Scenarios).
     * @param {string} defaultName - Baseline suggestion to pre-populate input block.
     * @returns {Promise<string|null>} Resolves string parameter on commit, or null upon cancellation.
     */
    _promptScenarioName(defaultName) {
        const modal = document.getElementById('scenario-name-modal');
        const input = document.getElementById('scenario-name-input');
        if (!modal || !input) return Promise.resolve(defaultName);
        input.value = defaultName || '';
        modal.classList.remove('hidden');
        requestAnimationFrame(() => input.focus());
        return new Promise(resolve => {
            this._state.scenarioNameResolver = resolve;
        });
    }

    /**
     * Closes string-prompt modal resolving the pending deferred promise.
     * @param {string|null} nameOrNull - String value confirming intent, null discarding.
     */
    _resolveScenarioName(nameOrNull) {
        const modal = document.getElementById('scenario-name-modal');
        if (modal) modal.classList.add('hidden');
        if (this._state.scenarioNameResolver) {
            this._state.scenarioNameResolver(nameOrNull);
            this._state.scenarioNameResolver = null;
        }
    }

    /**
     * Prints immediate contextual feedback texts relative to targeted input form sections.
     * @param {HTMLElement} el - Container element node tracking string output.
     * @param {string} text - Message sequence.
     * @param {string} type - 'error' or 'success' determining UI style.
     */
    _showMsg(el, text, type) {
        if (!el) return;
        el.textContent = text;
        el.className = `text-xs ${type === 'error' ? 'text-red-400' : 'text-teal-400'}`;
        el.classList.remove('hidden');
    }

    /**
     * Reports a non-blocking frontend failure without turning recurring background noise into toast spam.
     * @param {string} scope Short scope label for diagnostics.
     * @param {*} error Raw error object or message.
     * @param {Object} [options={}] Reporting options.
     * @returns {void}
     */
    _reportNonBlockingIssue(scope, error, options = {}) {
        const message = error?.response?.data?.detail || error?.message || String(error || 'Unexpected frontend error');
        const statusId = String(options.statusId || '');
        const statusMessage = String(options.statusMessage || '');
        const toastMessage = String(options.toastMessage || '');
        const toastType = String(options.toastType || 'warn');
        const dedupeMs = Number.isFinite(Number(options.dedupeMs)) ? Number(options.dedupeMs) : 15000;
        const dedupeKey = String(options.dedupeKey || `${scope}:${toastMessage || statusMessage || message}`);
        const now = Date.now();
        const last = Number(this._state.nonBlockingIssueTimestamps?.[dedupeKey] || 0);
        const shouldEmit = now - last >= dedupeMs;

        if (statusId) {
            const statusEl = document.getElementById(statusId);
            if (statusEl) {
                statusEl.textContent = statusMessage || message;
                statusEl.classList.remove('hidden');
            }
        }

        if (!shouldEmit) return;

        if (!this._state.nonBlockingIssueTimestamps) {
            this._state.nonBlockingIssueTimestamps = {};
        }
        this._state.nonBlockingIssueTimestamps[dedupeKey] = now;

        console.warn(`[frontend] ${scope}: ${message}`, error);

        if (toastMessage) {
            this._showToast(toastMessage, toastType);
        }
    }

    /**
     * Assays a designated string sequence against established strong criteria constraints (length, symbols).
     * @param {string} password - Raw string logic element.
     * @returns {Array<string>} List specifying which parameters were structurally omitted.
     */
    _passwordPolicyErrors(password = '') {
        const pwd = String(password || '');
        const errors = [];
        if (pwd.length < 10) errors.push('at least 10 characters');
        if (/\s/.test(pwd)) errors.push('no spaces');
        if (!/[a-z]/.test(pwd)) errors.push('one lowercase letter');
        if (!/[A-Z]/.test(pwd)) errors.push('one uppercase letter');
        if (!/\d/.test(pwd)) errors.push('one number');
        if (!/[^A-Za-z0-9\s]/.test(pwd)) errors.push('one special character');
        return errors;
    }

    /**
     * Retrieve standard readable feedback mapping full password criteria requirements.
     * @returns {string} Explanatory context.
     */
    _passwordPolicyText() {
        return 'Password policy: min 10 chars, uppercase, lowercase, number, special char, no spaces.';
    }

    /**
     * Inverts and updates visual sidebar size reduction formats.
     */
    _toggleSidebar() {
        this._state.sidebarCollapsed = !this._state.sidebarCollapsed;
        this._applySidebarState();
    }

    /**
     * Renders local representation of sidebar dimensions variables according to global state logic.
     */
    _applySidebarState() {
        const shell = document.querySelector('.app-shell');
        if (!shell) return;
        const collapsed = !!this._state.sidebarCollapsed;
        shell.classList.toggle('sidebar-collapsed', collapsed);
        const btn = document.getElementById('sidebar-toggle-btn');
        const icon = document.getElementById('sidebar-toggle-icon');
        if (btn) btn.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
        if (icon) icon.className = collapsed ? 'ph ph-caret-double-right' : 'ph ph-caret-double-left';
        try {
            localStorage.setItem('fl_sidebar_collapsed', collapsed ? '1' : '0');
        } catch { }
    }

    // ── Dashboard init ────────────────────────────────────────────────────────

    /**
     * Bootstraps logic specifically tailored for post-authentication states.
     * Activates data population, API long-polling configurations, components mappings, 
     * custom modules (charts, dataset upload tracking).
     * @returns {Promise<void>}
     */
    async _initDashboard() {
        await Promise.all([
            this._loadDefaultsIntoForm(),
            this._loadScenarioList(),
            this._refreshRunsList(),
        ]);
        this._toggleCustomModelInputs();
        this._applySetupMode();
        this._initSetupWizard();
        this._applyMethodConditionalFields();
        this._syncClientLearningRateInputs?.();
        this._updateGppWarning?.();
        this._toggleCustomDatasetInputs();
        this._initCodeEditors();
        this._startAlertPolling();
        this._initLogAutoScroll();

        let preferredTab = 0;
        try {
            const raw = localStorage.getItem('fl_dashboard_tab');
            const parsed = Number.parseInt(raw || '0', 10);
            const maxTab = this._state.isAdmin ? 6 : 4;
            if (Number.isFinite(parsed) && parsed >= 0 && parsed <= maxTab) preferredTab = parsed;
            if (!raw) preferredTab = 1;
        } catch { }
        this._switchTab(preferredTab);

        if (this._state.isAdmin) {
            const ownerOpt = document.getElementById('hist-search-owner-option');
            const ownerHeader = document.getElementById('hist-owner-col-header');
            const searchInput = document.getElementById('hist-search-input');
            const datasetsOwnerOpt = document.getElementById('datasets-search-owner-option');
            const datasetsOwnerSortOpt = document.getElementById('datasets-sort-owner-option');
            if (ownerOpt) {
                ownerOpt.disabled = false;
                ownerOpt.hidden = false;
            }
            if (datasetsOwnerOpt) {
                datasetsOwnerOpt.disabled = false;
                datasetsOwnerOpt.hidden = false;
            }
            if (datasetsOwnerSortOpt) {
                datasetsOwnerSortOpt.disabled = false;
                datasetsOwnerSortOpt.hidden = false;
            }
            if (ownerHeader) ownerHeader.hidden = false;
            if (searchInput) searchInput.placeholder = 'Search runs (name, owner, dataset, method...)';

            try {
                // Dynamically inject the admin module script and wait for it
                await new Promise((resolve, reject) => {
                    if (document.getElementById('admin-module-script')) return resolve();
                    if (window.AppAdminModule) return resolve(); // Already loaded
                    const script = document.createElement('script');
                    script.id = 'admin-module-script';
                    script.src = '/wrapper/js/modules/admin.module.js';
                    script.onload = () => {
                        if (window.AppAdminModule) Object.assign(this.constructor.prototype, window.AppAdminModule);
                        resolve();
                    };
                    script.onerror = reject;
                    document.head.appendChild(script);
                });

                this._bindAdminEvents();
                this._refreshAdminOverview();
                this._refreshAdminUsers();
            } catch (err) {
                console.error("Failed to load admin module:", err);
            }
        } else {
            const ownerOpt = document.getElementById('hist-search-owner-option');
            const ownerHeader = document.getElementById('hist-owner-col-header');
            const searchInput = document.getElementById('hist-search-input');
            const datasetsOwnerOpt = document.getElementById('datasets-search-owner-option');
            const datasetsOwnerSortOpt = document.getElementById('datasets-sort-owner-option');
            if (ownerOpt) {
                ownerOpt.disabled = true;
                ownerOpt.hidden = true;
            }
            if (datasetsOwnerOpt) {
                datasetsOwnerOpt.disabled = true;
                datasetsOwnerOpt.hidden = true;
            }
            if (datasetsOwnerSortOpt) {
                datasetsOwnerSortOpt.disabled = true;
                datasetsOwnerSortOpt.hidden = true;
            }
            if (ownerHeader) ownerHeader.hidden = true;
            if (searchInput) searchInput.placeholder = 'Search runs (name, dataset, method...)';
            const fieldSel = document.getElementById('hist-search-field');
            if (fieldSel && fieldSel.value === 'owner') fieldSel.value = 'all';
            this._state.histSearchField = String(fieldSel?.value || 'all');

            const datasetsFieldSel = document.getElementById('datasets-search-field');
            if (datasetsFieldSel && datasetsFieldSel.value === 'owner') datasetsFieldSel.value = 'all';
            this._state.datasetsSearchField = String(datasetsFieldSel?.value || 'all');

            const datasetsSortSel = document.getElementById('datasets-sort-field');
            if (datasetsSortSel && datasetsSortSel.value === 'owner_asc') datasetsSortSel.value = 'dataset_name_asc';
            this._state.datasetsSortField = String(datasetsSortSel?.value || 'dataset_name_asc');
        }

        // Upgrades every dropdown already in the DOM and keeps watching for the ones
        // rendered later (setup wizard, scenarios, admin panels).
        window.MobileSelect?.observe();

        // Restore any running session
        const runningPayload = await api.getRunsPage({
            search: 'running',
            searchField: 'status',
            allUsers: false,
            refreshDb: false,
            page: 1,
            pageSize: 20,
            sortKey: 'run_ts',
            sortDir: 'desc',
        }).catch(() => ({ runs: [] }));
        const running = (runningPayload.runs || []).find(r => r.status === 'running');
        if (running) {
            this._state.currentRunId = running.name;
            this._state.trainingStatus = 'running';
            this._setTrainingButtons(true);
            await this._hydrateMonitoringFromRun(running.name).catch(() => { });
            this._startPolling();
        }
    }

    /**
     * Executes GET retrieval onto platform configurations baseline values, feeding into forms limits logic.
     * @returns {Promise<void>}
     */
    async _loadDefaultsIntoForm() {
        try {
            const defaults = await api.getDefaults();
            this._populateFormDefaults(defaults);
        } catch { }
    }

    /**
     * Translates provided key-value dictionary onto identical ID layout parameter DOM elements variables bindings formats structure mappings formats constraints bindings.
     * @param {Object} cfg - Target configuration parameters dictionary properties.
     */
    _populateFormDefaults(cfg) {
        const normalizeGpTransferLearning = (value) => {
            if (value === true) return 'full_reuse';
            if (value === false || value == null) return 'disabled';
            const v = String(value).trim();
            if (!v) return 'disabled';
            if (v === 'true') return 'full_reuse';
            if (['false', 'none', 'disabled', 'off'].includes(v.toLowerCase())) return 'disabled';
            return v;
        };

        const normalizedCfg = { ...(cfg || {}) };
        if (normalizedCfg.server_warmup_epochs == null && normalizedCfg.global_model_epochs != null) {
            normalizedCfg.server_warmup_epochs = normalizedCfg.global_model_epochs;
        }

        normalizedCfg.gp_transfer_learning = normalizeGpTransferLearning(normalizedCfg.gp_transfer_learning);
        if (normalizedCfg.learning_rate == null) {
            if (Array.isArray(normalizedCfg.client_learning_rates) && normalizedCfg.client_learning_rates.length) {
                normalizedCfg.learning_rate = normalizedCfg.client_learning_rates[0];
            } else if (typeof normalizedCfg.client_learning_rates === 'string' && normalizedCfg.client_learning_rates.trim()) {
                const firstLr = parseFloat(String(normalizedCfg.client_learning_rates).split(',')[0]);
                if (Number.isFinite(firstLr)) normalizedCfg.learning_rate = firstLr;
            }
        }

        const set = (id, val) => {
            const el = document.getElementById(`cfg-${id}`);
            if (!el) return;
            if (id === 'custom_model_code' && (val === null || val === undefined || String(val).trim() === '')) {
                return;
            }
            if (id === 'round_client_allocation_percentages') {
                if (Array.isArray(val)) el.value = JSON.stringify(val);
                else if (val !== null && val !== undefined) el.value = String(val);
                return;
            }
            if (el.type === 'checkbox') el.checked = !!val;
            else if (Array.isArray(val)) el.value = val.join(',');
            else if (val !== null && val !== undefined) el.value = val;
        };

        const fields = [
            'method', 'dataset_name', 'num_clients', 'initial_eligible_clients', 'server_data_percentage', 'imbalance_rate',
            'train_val_split', 'batch_size', 'server_learning_rate', 'learning_rate', 'momentum', 'mu', 'fedprox_weighted', 'death_prob', 'new_client_prob', 'seed',
            'server_warmup_epochs', 'local_model_epochs', 'weights_sending_frequency',
            'evaluation_split_mode', 'round_train_schedule_mode', 'round_train_schedule_percentages',
            'round_client_allocation_mode', 'round_client_allocation_percentages',
            'run_names',
            'client_learning_rates',
            'individuals', 'generations', 'mutation_rate',
            'crossover_rate', 'elitism_size', 'gp_patience', 'max_tree_size',
            'mutation_subtree_maxsize', 'gp_fitness_metric',
            'gp_initialization', 'gp_transfer_learning', 'mutation_type', 'crossover_type',
            'selection_type', 'available_primitives', 'repeat', 'iid',
            'custom_model_mode', 'custom_model_code',
        ];
        fields.forEach(f => set(f, normalizedCfg[f]));
        this._hydrateRoundTrainScheduleState?.(normalizedCfg.round_train_schedule_percentages);
        this._hydrateRoundClientAllocationState?.(normalizedCfg.round_client_allocation_percentages);
        const mEl = document.getElementById('cfg-method');
        if (mEl) {
            const selectedMethods = Array.isArray(normalizedCfg.methods) && normalizedCfg.methods.length
                ? normalizedCfg.methods
                : [normalizedCfg.method || mEl.options?.[0]?.value].filter(Boolean);
            Array.from(mEl.options).forEach(opt => {
                opt.selected = selectedMethods.includes(opt.value);
            });
        }
        this._syncMethodChipPicker?.();
        const runNamesWrap = document.getElementById('run-name-inputs');
        if (runNamesWrap) runNamesWrap.innerHTML = '';
        this._syncRunNameInputs?.();

        if (normalizedCfg.min_tree_size !== false && normalizedCfg.min_tree_size != null) {
            document.getElementById('cfg-min_tree_size_enabled').checked = true;
            const inp = document.getElementById('cfg-min_tree_size');
            inp.value = normalizedCfg.min_tree_size;
            inp.disabled = false;
        }

        this._state.customDatasetRef = String(normalizedCfg.custom_dataset_ref || '');
        this._state.customDatasetFileName = String(normalizedCfg.custom_dataset_name || normalizedCfg.custom_dataset_ref || '');
        this._state.customDatasetFormat = this._normalizeCustomDatasetFormat(
            normalizedCfg.custom_dataset_format || '',
            normalizedCfg.dataset_name || '',
        );

        if (['custom_csv', 'custom_image_npz', 'custom_image_folder', 'custom'].includes(normalizedCfg.dataset_name)) {
            if (normalizedCfg.dataset_name !== 'custom') {
                this._state.customDatasetType = this._normalizeCustomDatasetType(
                    normalizedCfg.custom_dataset_format || '',
                    normalizedCfg.dataset_name || '',
                );
            }
            const dsSel = document.getElementById('cfg-dataset_name');
            if (dsSel) dsSel.value = 'custom';
        }

        this._toggleCustomModelInputs?.();
        this._toggleCustomDatasetInputs?.();
        this._toggleRoundTrainScheduleInputs?.();
        this._applySetupMode?.();

        // Push the loaded model code into CodeMirror only after the editor panel
        // has been revealed by _toggleCustomModelInputs above. Setting it while
        // the container was still hidden left the editor blank until the user
        // clicked into it (CodeMirror mis-measures in a display:none host).
        const incomingModelCode = String(normalizedCfg.custom_model_code || '');
        if (this._state.editors?.model && incomingModelCode.trim()) {
            this._state.editors.model.setValue(incomingModelCode);
            requestAnimationFrame(() => this._state.editors?.model?.refresh?.());
        }

        const dsNameEl = document.getElementById('custom-dataset-file-name');
        if (dsNameEl && this._state.customDatasetFileName) dsNameEl.textContent = this._state.customDatasetFileName;
        const dsMetaEl = document.getElementById('custom-dataset-meta');
        if (dsMetaEl && this._state.customDatasetRef) dsMetaEl.textContent = `Loaded reference: ${this._state.customDatasetRef}`;
    }

    /**
     * Checks validation logic generating model custom fields mappings targets.
     * @param {string} [fileName=''] - Overriding logic strings constraints targets formats fields.
     */
    _ensureLoadedModelOption(fileName = '') {
        const sel = document.getElementById('cfg-custom_model_mode');
        if (!sel) return;
        let opt = Array.from(sel.options).find(o => o.value === 'loaded_file');
        if (!opt) {
            opt = document.createElement('option');
            opt.value = 'loaded_file';
            sel.appendChild(opt);
        }
        opt.textContent = fileName ? `Loaded model (${fileName})` : 'Loaded model';
    }

    /**
     * Opens the authenticated change-password modal.
     * @returns {void}
     */
    _openChangePasswordModal() {
        document.getElementById('change-password-current').value = '';
        document.getElementById('change-password-new').value = '';
        document.getElementById('change-password-new-confirm').value = '';
        const msg = document.getElementById('change-password-msg');
        if (msg) msg.classList.add('hidden');
        document.getElementById('change-password-modal')?.classList.remove('hidden');
    }

    /**
     * Closes the authenticated change-password modal.
     * @returns {void}
     */
    _closeChangePasswordModal() {
        document.getElementById('change-password-modal')?.classList.add('hidden');
    }

    /**
     * Validates and submits the authenticated change-password form.
     * @returns {Promise<void>}
     */
    async _handleChangePassword() {
        const currentPwd = String(document.getElementById('change-password-current')?.value || '');
        const newPwd = String(document.getElementById('change-password-new')?.value || '');
        const newPwdConfirm = String(document.getElementById('change-password-new-confirm')?.value || '');
        const msg = document.getElementById('change-password-msg');
        if (!currentPwd || !newPwd || !newPwdConfirm) {
            this._showMsg(msg, 'Current password, new password and confirmation are required', 'error');
            return;
        }
        if (newPwd !== newPwdConfirm) {
            this._showMsg(msg, 'New password and confirmation do not match', 'error');
            return;
        }
        const pwdErrors = this._passwordPolicyErrors(newPwd);
        if (pwdErrors.length) {
            this._showMsg(msg, `${this._passwordPolicyText()} Missing: ${pwdErrors.join(', ')}.`, 'error');
            return;
        }
        try {
            await api.changePasswordAuthenticated(currentPwd, newPwd, newPwdConfirm);
            this._showMsg(msg, 'Password updated successfully', 'success');
            setTimeout(() => this._closeChangePasswordModal(), 700);
        } catch (e) {
            this._showMsg(msg, e.response?.data?.detail || 'Password update failed', 'error');
        }
    }

}

[
    window.AppSetupCustomDatasetModule,
    window.AppSetupModule,
    window.AppDatasetsModule,
    window.AppEventsModule,
    window.AppClientLearningRateModule,
    window.AppHistoryListModule,
    window.AppHistoryModule,
    window.AppRenderModule,
    window.AppJobsModule,
    window.AppTrainingScenariosModule,
    window.AppTrainingModule,
    window.AppRuntimeModule,
].filter(Boolean).forEach((moduleMixin) => {
    Object.assign(App.prototype, moduleMixin);
});

// ── Boot ──────────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
    window.app.boot();
});
