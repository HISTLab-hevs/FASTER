/**
 * events.module.js — App mixin for centralizing and orchestrating UI DOM events.
 *
 * Handles bindings for landing page navigation, dashboard toolbars, wizard
 * interaction states, and administrative modals.
 */

const CUSTOM_MODEL_FILE_MAX_MB = 5;
const CUSTOM_MODEL_FILE_MAX_BYTES = CUSTOM_MODEL_FILE_MAX_MB * 1024 * 1024;

window.AppEventsModule = {
    /**
     * Initializes unauthenticated public landing page interaction tracking.
     * Binds toggles for tooltips, login accordions, and mobile offcanvas menus.
     * @returns {void}
     */
    _attachLandingEventsModular() {
        this._initHelpTooltips();
        document.getElementById('login-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this._handleLogin();
        });

        const loginWrap = document.getElementById('login-card-wrap');
        const loginChevron = document.getElementById('login-accordion-chevron');

        const setLoginAccordion = (open) => {
            if (!loginWrap) return;
            loginWrap.classList.toggle('login-accordion-open', !!open);
            loginWrap.setAttribute('aria-hidden', open ? 'false' : 'true');
            if (loginChevron) loginChevron.classList.toggle('rotate-180', !!open);
        };

        const landingMenuBtn = document.getElementById('landing-mobile-menu-btn');
        const landingMenuPanel = document.getElementById('landing-mobile-menu-panel');
        landingMenuBtn?.addEventListener('click', () => {
            landingMenuPanel?.classList.toggle('hidden');
        });
        landingMenuPanel?.querySelectorAll('a[href]').forEach((a) => {
            a.addEventListener('click', () => landingMenuPanel.classList.add('hidden'));
        });

        document.querySelectorAll('a[href="#/signin"], a[href="#/signin/login-card"]').forEach((a) => {
            a.addEventListener('click', (e) => {
                if (this._publicRouteFromPath() !== 'signin') return;
                e.preventDefault();
                setLoginAccordion(false);
                landingMenuPanel?.classList.add('hidden');
            });
        });

        document.getElementById('login-accordion-toggle')?.addEventListener('click', () => {
            if (!loginWrap) return;
            const willOpen = !loginWrap.classList.contains('login-accordion-open');
            setLoginAccordion(willOpen);
        });

        setLoginAccordion(false);

        // Intercept landing page links for SPA routing with capitalized slugs
        document.querySelector('.landing-shell')?.addEventListener('click', (e) => {
            const a = e.target.closest('a[href^="/Faster/"]');
            if (!a || a.target === '_blank') return;

            const href = a.getAttribute('href') || '';
            const parts = href.split('/').filter(Boolean);
            if (parts.length >= 2 && parts[0].toLowerCase() === 'faster') {
                const slug = parts[1];
                const publicRoutes = ['docs', 'creators', 'support', 'Documentation', 'Creators', 'Contribute', 'signin'];
                
                if (publicRoutes.includes(slug) || slug.toLowerCase() === 'signin') {
                    e.preventDefault();
                    const displayMap = { docs: 'Documentation', creators: 'Creators', support: 'Contribute', signin: 'Signin' };
                    const displaySlug = displayMap[slug] || slug;
                    const rest = parts.slice(2).join('/');
                    const newPath = `/Faster/${displaySlug}${rest ? '/' + rest : ''}`;
                    
                    if (window.location.pathname !== newPath) {
                        history.pushState(null, '', newPath);
                    }
                    this._onPopState();
                }
            }
        });
    },

    /**
     * Distributes DOM bindings across the primary authenticated dashboard workspace.
     * Routes logic setup arrays for Wizard configuration panels, charts formats, and global overlays.
     * @returns {void}
     */
    _attachDashboardEventsModular() {
        this._initHelpTooltips();
        this._bindDashboardTopbarEvents();
        this._bindDatasetsEvents();
        this._bindWizardCoreEvents();
        this._bindScenarioAndModelEvents();
        this._bindHistoryEvents();
        this._bindModalAndConsoleEvents();

        // Prevents users traversing windows when custom uploads process
        if (!this._datasetUploadBeforeUnloadBound) {
            window.addEventListener('beforeunload', (e) => {
                if (!this._state.customDatasetUpload?.active) return;
                this._requestCancelDatasetUpload?.('Dataset upload interrupted by page refresh/close.');
                e.preventDefault();
                e.returnValue = '';
            });
            this._datasetUploadBeforeUnloadBound = true;
        }
    },

    /**
     * Binds the topbar, sidebar, and authenticated navigation controls.
     * @returns {void}
     */
    _bindDashboardTopbarEvents() {
        document.getElementById('logout-btn')?.addEventListener('click', () => this._handleLogout());
        document.getElementById('change-password-btn')?.addEventListener('click', () => this._openChangePasswordModal());

        document.querySelectorAll('.nav-tab').forEach((btn) => {
            btn.addEventListener('click', () => this._switchTab(parseInt(btn.dataset.tab, 10)));
        });

        document.getElementById('auth-link-docs')?.addEventListener('click', () => this._openAuthResource('Documentation'));
        document.getElementById('auth-link-creators')?.addEventListener('click', () => this._openAuthResource('Creators'));
        document.getElementById('auth-link-support')?.addEventListener('click', () => this._openAuthResource('Contribute'));

        const mobileMenuBtn = document.getElementById('mobile-app-menu-btn');
        const mobileMenuPanel = document.getElementById('mobile-app-menu-panel');
        mobileMenuBtn?.addEventListener('click', () => {
            mobileMenuPanel?.classList.toggle('hidden');
        });
        mobileMenuPanel?.querySelectorAll('.mobile-tab-item[data-tab]').forEach((btn) => {
            btn.addEventListener('click', () => {
                const n = parseInt(btn.getAttribute('data-tab') || '0', 10);
                this._switchTab(Number.isNaN(n) ? 0 : n);
                mobileMenuPanel.classList.add('hidden');
            });
        });

        document.getElementById('mobile-auth-link-docs')?.addEventListener('click', () => {
            this._openAuthResource('Documentation');
            mobileMenuPanel?.classList.add('hidden');
        });
        document.getElementById('mobile-auth-link-creators')?.addEventListener('click', () => {
            this._openAuthResource('Creators');
            mobileMenuPanel?.classList.add('hidden');
        });
        document.getElementById('mobile-auth-link-support')?.addEventListener('click', () => {
            this._openAuthResource('Contribute');
            mobileMenuPanel?.classList.add('hidden');
        });
        document.getElementById('mobile-logout-btn')?.addEventListener('click', () => this._handleLogout());

        const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
        sidebarToggleBtn?.addEventListener('click', () => this._toggleSidebar());
        this._applySidebarState();
    },

    /**
     * Binds controls for the Datasets management tab.
     * @returns {void}
     */
    _bindDatasetsEvents() {
        document.getElementById('datasets-upload-file-btn')?.addEventListener('click', () => {
            document.getElementById('datasets-upload-file-input')?.click();
        });
        document.getElementById('datasets-upload-file-input')?.addEventListener('change', (e) => {
            const file = e.target?.files?.[0];
            const out = document.getElementById('datasets-upload-file-name');
            if (out) out.textContent = file ? file.name : 'No dataset selected';
        });
        document.getElementById('datasets-upload-submit-btn')?.addEventListener('click', () => this._uploadDatasetFromDatasetsTab?.());
        document.getElementById('datasets-refresh-btn')?.addEventListener('click', () => this._refreshDatasetsManager?.());
        document.getElementById('datasets-search-field')?.addEventListener('change', (e) => this._setDatasetSearchField?.(e.target?.value || 'all'));
        document.getElementById('datasets-search-input')?.addEventListener('input', (e) => this._setDatasetSearchQuery?.(e.target?.value || ''));
        document.getElementById('datasets-sort-field')?.addEventListener('change', (e) => this._setDatasetSortField?.(e.target?.value || 'dataset_name_asc'));
        document.getElementById('datasets-custom-list')?.addEventListener('click', (e) => this._onDatasetsManagerClick?.(e));
    },

    /**
     * Binds the setup wizard navigation and core form controls.
     * @returns {void}
     */
    _bindWizardCoreEvents() {
        document.getElementById('start-btn')?.addEventListener('click', () => this._startTraining());
        document.getElementById('stop-btn')?.addEventListener('click', () => this._stopTraining());

        document.getElementById('setup-prev-btn')?.addEventListener('click', () => this._goSetupStep(-1));
        document.getElementById('setup-next-btn')?.addEventListener('click', () => this._goSetupStep(1));
        document.getElementById('cfg-method')?.addEventListener('change', () => this._applyMethodConditionalFields());
        document.getElementById('cfg-dataset_name')?.addEventListener('change', () => this._toggleCustomDatasetInputs());
        document.getElementById('cfg-num_clients')?.addEventListener('input', () => {
            this._syncClientLearningRateInputs?.();
            this._syncRoundClientAllocationInputs?.();
            this._updateGppWarning?.();
        });
        document.getElementById('cfg-available_primitives')?.addEventListener('input', () => this._updateGppWarning?.());
        document.getElementById('cfg-client_lr_mode')?.addEventListener('change', () => this._syncClientLearningRateInputs?.());
        document.getElementById('cfg-local_model_epochs')?.addEventListener('input', () => this._toggleRoundTrainScheduleInputs?.());
        document.getElementById('cfg-weights_sending_frequency')?.addEventListener('input', () => this._toggleRoundTrainScheduleInputs?.());
        document.getElementById('cfg-round_train_schedule_mode')?.addEventListener('change', () => this._toggleRoundTrainScheduleInputs?.());
        document.getElementById('cfg-round_client_allocation_mode')?.addEventListener('change', () => this._toggleRoundTrainScheduleInputs?.());
        document.getElementById('cfg-client_learning_rates')?.closest('.lr-compact-block')?.addEventListener('input', () => this._syncLearningRateBackingField?.());

        if (!this._state.dashboardResizeBound) {
            window.addEventListener('resize', () => this._syncClientLearningRateInputs?.());
            this._state.dashboardResizeBound = true;
        }

        document.getElementById('cfg-individuals')?.addEventListener('input', () => this._updateGppWarning?.());
        document.getElementById('cfg-generations')?.addEventListener('input', () => this._updateGppWarning?.());
        document.getElementById('cfg-max_tree_size')?.addEventListener('input', () => this._updateGppWarning?.());
        document.getElementById('setup-mode-simple')?.addEventListener('click', () => this._setSetupMode('simple'));
        document.getElementById('setup-mode-advanced')?.addEventListener('click', () => this._setSetupMode('advanced'));

        document.getElementById('save-defaults-btn')?.addEventListener('click', () => this._saveDefaults());
        document.getElementById('save-scenario-btn')?.addEventListener('click', () => this._saveScenario());
    },

    /**
     * Interconnects custom definitions settings contexts formatting validations validation models limits definitions formats algorithms limits sequences properties constraints dependencies validations limits constraints definitions context validations properties strings vectors models fields constraints schema logic implementations updates vectors parameters strings.
     */
    _bindScenarioAndModelEvents() {
        document.getElementById('scenario-dd')?.addEventListener('change', () => this._loadScenario());
        document.getElementById('scenario-delete-btn')?.addEventListener('click', () => this._deleteScenario?.());
        document.getElementById('scenario-file-choose-btn')?.addEventListener('click', () => {
            document.getElementById('scenario-file-input')?.click();
        });
        document.getElementById('scenario-file-input')?.addEventListener('change', (e) => {
            const f = e.target?.files?.[0];
            const nameEl = document.getElementById('scenario-file-name');
            if (nameEl) nameEl.textContent = f ? f.name : 'No file selected';
            if (f) this._loadScenarioFromFile();
        });

        document.getElementById('cfg-custom_model_mode')?.addEventListener('change', () => this._toggleCustomModelInputs());
        document.getElementById('custom-model-file-btn')?.addEventListener('click', () => {
            document.getElementById('custom-model-file-input')?.click();
        });
        document.getElementById('custom-model-file-input')?.addEventListener('change', async (e) => {
            const f = e.target?.files?.[0];
            const nameEl = document.getElementById('custom-model-file-name');
            if (nameEl) nameEl.textContent = f ? f.name : 'No file selected';
            this._state.customModelFileName = f?.name || '';
            if (!f) return;
            if (!/\.py$/i.test(String(f.name || ''))) {
                this._showToast('Model upload supports .py files only', 'warn');
                return;
            }
            if ((f.size || 0) > CUSTOM_MODEL_FILE_MAX_BYTES) {
                this._showToast(`Model file too large (max ${CUSTOM_MODEL_FILE_MAX_MB} MB)`, 'warn');
                return;
            }
            try {
                const txt = await f.text();
                const codeEl = document.getElementById('cfg-custom_model_code');
                if (codeEl) codeEl.value = txt;
                if (this._state.editors?.model) this._state.editors.model.setValue(txt);
                const modeEl = document.getElementById('cfg-custom_model_mode');
                if (modeEl) {
                    this._ensureLoadedModelOption(f.name);
                    modeEl.value = 'loaded_file';
                }
                this._toggleCustomModelInputs();
                this._showToast(`Loaded model file: ${f.name}`, 'success');
            } catch (e) {
                this._reportNonBlockingIssue?.('custom model file read', e, {
                    toastMessage: 'Unable to read model file',
                    toastType: 'error',
                    dedupeMs: 5000,
                });
            }
        });

        document.getElementById('custom-dataset-file-btn')?.addEventListener('click', () => {
            document.getElementById('custom-dataset-file-input')?.click();
        });
        document.getElementById('custom-dataset-file-input')?.addEventListener('change', async (e) => {
            const f = e.target?.files?.[0];
            const nameEl = document.getElementById('custom-dataset-file-name');
            if (nameEl) nameEl.textContent = f ? f.name : 'No dataset uploaded';
            if (f) await this._uploadCustomDatasetFromFile?.(f);
        });
        document.getElementById('custom-dataset-docs-link')?.addEventListener('click', async () => {
            await this._openAuthResource('Documentation');
            setTimeout(() => {
                const target = document.getElementById('chapter-datasets');
                target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 0);
        });
    },

    /**
     * Translates logic validation settings representations validations objects sequences updates mappings sequences limits templates logic values context matrices variables limits templates mappings values string implementations representation properties limits elements forms variables representations configurations settings limits properties sequences parameters lists frameworks formats vectors frameworks sequences.
     */
    _bindHistoryEvents() {
        document.getElementById('hist-refresh-btn')?.addEventListener('click', () => this._refreshRunsList(true));
        document.getElementById('hist-delete-btn')?.addEventListener('click', () => this._toggleDeleteMode());
        document.getElementById('hist-search-input')?.addEventListener('input', (e) => {
            this._state.histSearchQuery = String(e.target?.value || '').trim().toLowerCase();
            this._state.histTablePage = 1;
            this._scheduleHistoryListRefresh?.();
        });
        document.getElementById('hist-search-field')?.addEventListener('change', (e) => {
            this._state.histSearchField = String(e.target?.value || 'all');
            this._state.histTablePage = 1;
            this._scheduleHistoryListRefresh?.();
        });

        document.getElementById('hist-table-toggle')?.addEventListener('click', () => this._toggleHistoryTable());
        document.getElementById('hist-table-pagination')?.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-page-setter][data-page-value]');
            if (!btn || btn.disabled) return;
            const setterName = String(btn.getAttribute('data-page-setter') || '').trim();
            const page = Number(btn.getAttribute('data-page-value') || 1);
            const handler = setterName && typeof this[setterName] === 'function' ? this[setterName] : null;
            if (!handler) return;
            handler.call(this, page);
        });
        document.getElementById('runs-table-body')?.addEventListener('click', (e) => {
            const delBtn = e.target.closest('.run-del-x[data-run-delete]');
            if (delBtn) {
                e.stopPropagation();
                const runId = String(delBtn.getAttribute('data-run-delete') || '').trim();
                if (!runId) return;
                this._requestDeleteRun?.(runId);
                return;
            }

            const modelBtn = e.target.closest('[data-run-model-code]');
            if (modelBtn) {
                e.stopPropagation();
                const runKey = String(modelBtn.getAttribute('data-run-model-code') || '').trim();
                if (runKey) this._openModelCodeForRun?.(runKey);
                return;
            }

            const compareBtn = e.target.closest('[data-run-compare]');
            if (compareBtn) {
                e.stopPropagation();
                const runKey = String(compareBtn.getAttribute('data-run-compare') || '').trim();
                if (runKey) this._toggleRunSelection?.(runKey);
                return;
            }

            const row = e.target.closest('[data-run-row]');
            if (!row) return;
            const runKey = String(row.getAttribute('data-run-row') || '').trim();
            if (runKey) this._onRunRowClicked?.(runKey);
        });
        document.querySelectorAll('[data-hist-sort]').forEach((th) => {
            th.addEventListener('click', () => this._setHistoryTableSort?.(th.getAttribute('data-hist-sort') || 'name'));
        });

        document.getElementById('hist-run-detail-close')?.addEventListener('click', () => this._closeRunDetailModal());
        document.getElementById('hist-run-detail-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'hist-run-detail-modal') this._closeRunDetailModal();
        });
        document.getElementById('hist-run-detail-name')?.addEventListener('click', () => this._openRenameRunDialog?.());
        document.getElementById('hist-run-detail-system-btn')?.addEventListener('click', () => this._openRunSystemInfoModal?.());
        document.getElementById('hist-run-system-close')?.addEventListener('click', () => this._closeRunSystemInfoModal?.());
        document.getElementById('hist-run-system-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'hist-run-system-modal') this._closeRunSystemInfoModal?.();
        });

        document.getElementById('hist-run-detail-config-section')?.addEventListener('toggle', (e) => {
            if (e.target?.open) this._state.editors?.runCfg?.refresh();
        });
        document.getElementById('hist-run-detail-model-section')?.addEventListener('toggle', (e) => {
            if (e.target?.open) this._state.editors?.runModel?.refresh();
        });

        document.getElementById('hist-model-code-close')?.addEventListener('click', () => this._closeModelCodeModal());
        document.getElementById('hist-model-code-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'hist-model-code-modal') this._closeModelCodeModal();
        });

        document.getElementById('scenario-yaml-help')?.addEventListener('toggle', (e) => {
            if (e.target?.open) this._state.editors?.scenarioSchema?.refresh();
        });
    },

    /**
     * Binds internal events mappings schemas parameters values context structures vectors formats validations variables representations variables updates implementations parameters logic constraints vectors representations fields mappings matrices variables implementations metrics rules context algorithms implementations models vectors formats structures settings updates bindings targets architectures settings forms templates formatting logic schemas attributes templates parameters representations references variables vectors targets schemas structures elements logic forms targets bindings formats string lists models strings templates.
     */
    _bindModalAndConsoleEvents() {
        document.getElementById('confirm-modal-cancel')?.addEventListener('click', () => this._resolveConfirm(false));
        document.getElementById('confirm-modal-ok')?.addEventListener('click', () => this._resolveConfirm(true));
        document.getElementById('confirm-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'confirm-modal') this._resolveConfirm(false);
        });

        document.getElementById('scenario-name-cancel')?.addEventListener('click', () => this._resolveScenarioName(null));
        document.getElementById('scenario-name-save')?.addEventListener('click', () => {
            const name = document.getElementById('scenario-name-input')?.value?.trim() || '';
            this._resolveScenarioName(name || null);
        });
        document.getElementById('scenario-name-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'scenario-name-modal') this._resolveScenarioName(null);
        });

        document.getElementById('change-password-cancel')?.addEventListener('click', () => this._closeChangePasswordModal());
        document.getElementById('change-password-save')?.addEventListener('click', () => this._handleChangePassword());
        document.getElementById('change-password-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'change-password-modal') this._closeChangePasswordModal();
        });

        document.getElementById('cfg-min_tree_size_enabled')?.addEventListener('change', (e) => {
            const target = document.getElementById('cfg-min_tree_size');
            if (target) target.disabled = !e.target.checked;
        });

        document.getElementById('log-toggle-btn')?.addEventListener('click', () => this._toggleLogConsole());
        document.getElementById('log-console')?.addEventListener('mouseup', () => {
            const main = document.querySelector('.app-shell main');
            const wrap = document.getElementById('log-console-wrap');
            if (main && wrap) {
                const y = wrap.offsetTop + wrap.offsetHeight + 32;
                main.scrollTo({ top: y, behavior: 'smooth' });
            }
        });

        document.getElementById('myjob-modal-close')?.addEventListener('click', () => this._closeMyJobModal());
        document.getElementById('myjob-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'myjob-modal') this._closeMyJobModal();
        });
        document.getElementById('jobs-queue-list')?.addEventListener('click', (e) => this._onJobsPanelClick(e));
        document.getElementById('jobs-pagination')?.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-jobs-page]');
            if (!btn || btn.disabled) return;
            this._setJobsPage?.(btn.getAttribute('data-jobs-page'));
        });
        document.getElementById('dataset-upload-cancel')?.addEventListener('click', () => {
            this._requestCancelDatasetUpload?.('Dataset upload canceled by user.');
        });
    }
};
