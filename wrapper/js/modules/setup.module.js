window.AppSetupModule = {
    /**
     * Truncates long labels for compact UI slots while preserving full text in tooltip.
     * @param {string} value Raw label.
     * @param {number} maxLen Maximum visible length.
     * @returns {string} Truncated label.
     */
    _truncateLabel(value, maxLen = 56) {
        const txt = String(value || '').trim();
        if (!txt) return '';
        if (txt.length <= maxLen) return txt;
        return `${txt.slice(0, Math.max(1, maxLen - 1)).trimEnd()}…`;
    },

    /**
     * Normalizes backend dataset metadata into the frontend execution family.
     * @param {string} rawFormat Precise backend dataset format marker.
     * @param {string} [rawType=''] Backend dataset type marker.
     * @returns {'csv'|'image_npz'|'image_folder'} Frontend dataset type.
     */
    _normalizeCustomDatasetType(rawFormat, rawType = '') {
        const fmt = this._normalizeCustomDatasetFormat(rawFormat, rawType);
        if (fmt === 'npz') return 'image_npz';
        if (fmt === 'zip') return 'image_folder';
        return 'csv';
    },

    /**
     * Resets Experimental Setup to a clean initial state.
     * @param {{keepYamlFilename?: boolean}} [opts] Optional reset controls.
     * @returns {Promise<void>}
     */
    async _resetExperimentalSetupState(opts = {}) {
        const keepYamlFilename = !!opts.keepYamlFilename;
        const scenarioNameEl = document.getElementById('scenario-file-name');
        const scenarioInputEl = document.getElementById('scenario-file-input');
        const keptScenarioName = keepYamlFilename ? String(scenarioNameEl?.textContent || '') : '';

        this._clearSetupWarnings?.();
        this._clearSetupFieldErrors?.();

        this._state.customDatasetRef = '';
        this._state.customDatasetFileName = '';
        this._state.customDatasetFormat = 'csv';
        this._state.customModelFileName = '';
        this._state.roundClientAllocationManual = [];

        const runNamesHiddenEl = document.getElementById('cfg-run_names');
        const runNamesWrapEl = document.getElementById('run-name-inputs');
        const runNamesErrEl = document.getElementById('run-name-inputs-error');
        if (runNamesHiddenEl) runNamesHiddenEl.value = '';
        if (runNamesWrapEl) runNamesWrapEl.innerHTML = '';
        if (runNamesErrEl) {
            runNamesErrEl.textContent = '';
            runNamesErrEl.classList.add('hidden');
        }

        const dsNameEl = document.getElementById('custom-dataset-file-name');
        const dsMetaEl = document.getElementById('custom-dataset-meta');
        const dsStatusEl = document.getElementById('custom-dataset-status');
        const dsWarnEl = document.getElementById('custom-dataset-simple-warning');
        if (dsNameEl) dsNameEl.textContent = 'No file';
        if (dsMetaEl) dsMetaEl.textContent = '';
        if (dsStatusEl) {
            dsStatusEl.textContent = '';
            dsStatusEl.classList.add('hidden');
        }
        if (dsWarnEl) {
            dsWarnEl.textContent = '';
            dsWarnEl.classList.add('hidden');
        }

        const modelNameEl = document.getElementById('custom-model-file-name');
        if (modelNameEl) modelNameEl.textContent = 'No file selected';
        if (!keepYamlFilename && scenarioNameEl) scenarioNameEl.textContent = 'No file selected';
        if (!keepYamlFilename && scenarioInputEl) scenarioInputEl.value = '';

        await this._loadDefaultsIntoForm?.();
        this._toggleCustomDatasetInputs?.();
        this._toggleCustomModelInputs?.();

        if (keepYamlFilename && scenarioNameEl) {
            scenarioNameEl.textContent = keptScenarioName || 'No file selected';
        }

        this._state.setupStep = 1;
        this._renderSetupWizard?.();
        this._applyMethodConditionalFields?.();
        this._toggleCustomModelInputs?.();
        await this._toggleCustomDatasetInputs?.();

        const scenarioDd = document.getElementById('scenario-dd');
        if (scenarioDd) scenarioDd.value = '';
    },
    /**
     * Opens the dataset upload progress modal and initializes its UI.
     * @param {string} fileName The name of the file being uploaded.
     * @returns {void}
     */
    _openDatasetUploadModal(fileName) {
        const modal = document.getElementById('dataset-upload-modal');
        const title = document.getElementById('dataset-upload-title');
        const stage = document.getElementById('dataset-upload-stage');
        const fill = document.getElementById('dataset-upload-progress-fill');
        const text = document.getElementById('dataset-upload-progress-text');
        if (title) title.textContent = `Uploading ${fileName}`;
        if (stage) stage.textContent = 'Preparing file...';
        if (fill) fill.style.width = '0%';
        if (text) text.textContent = '0%';
        modal?.classList.remove('hidden');
    },

    /**
     * Updates the dataset upload progress modal with current progress percentage and stage text.
     * @param {number|string} progressPct The current progress percentage (0-100).
     * @param {string} stageText A description of the current upload stage.
     * @returns {void}
     */
    _updateDatasetUploadModal(progressPct, stageText) {
        const pct = Math.max(0, Math.min(100, Number(progressPct) || 0));
        const stage = document.getElementById('dataset-upload-stage');
        const fill = document.getElementById('dataset-upload-progress-fill');
        const text = document.getElementById('dataset-upload-progress-text');
        if (stage && stageText) stage.textContent = stageText;
        if (fill) fill.style.width = `${pct.toFixed(1)}%`;
        if (text) text.textContent = `${pct.toFixed(1)}%`;
    },

    /**
     * Closes the dataset upload progress modal.
     * @returns {void}
     */
    _closeDatasetUploadModal() {
        document.getElementById('dataset-upload-modal')?.classList.add('hidden');
    },

    /**
     * Requests cancellation of an ongoing custom dataset upload.
     * @param {string} [reason='Dataset upload canceled.'] The reason for cancellation.
     * @returns {boolean} True if an active upload was signaled to cancel, false otherwise.
     */
    _requestCancelDatasetUpload(reason = 'Dataset upload canceled.') {
        const task = this._state.customDatasetUpload;
        if (!task?.active) return false;
        task.cancelReason = reason;
        task.cancelRequested = true;
        this._updateDatasetUploadModal(task.progressPct || 0, 'Canceling upload...');
        try { task.controller?.abort?.(); } catch { }
        return true;
    },

    /**
     * Initializes setup wizard state and renders the first step.
     * @returns {void}
     */
    _initSetupWizard() {
        this._state.setupStep = 1;
        this._state.setupSteps = [
            { id: 1, title: 'Mode' },
            { id: 2, title: 'Scenario + Methods' },
            { id: 3, title: 'Training & Data' },
            { id: 4, title: 'Model & Advanced' },
            { id: 5, title: 'Review & Submit' },
        ];
        this._initMethodChipPicker();
        this._syncRunNameInputs();
        this._syncClientLearningRateInputs();
        this._updateGppWarning();
        this._bindSetupValidationObservers?.();
        this._updateClientChurnWarnings?.();
        this._renderSetupWizard();
    },

    /**
     * Extracts selected method options from the method dropdown along with their display labels.
     * @returns {Array<{value: string, label: string}>} Array of mapped method objects.
     */
    _selectedMethodsWithLabels() {
        const sel = document.getElementById('cfg-method');
        if (!sel) return [];
        return Array.from(sel.selectedOptions || []).map((opt) => ({
            value: String(opt.value || '').trim(),
            label: String(opt.textContent || opt.value || '').trim(),
        })).filter((item) => item.value);
    },

    /**
     * Combines dynamically generated run name input values and writes them to the hidden form field.
     * @returns {void}
     */
    _syncRunNamesHiddenFromInputs() {
        const hidden = document.getElementById('cfg-run_names');
        const wrap = document.getElementById('run-name-inputs');
        if (!hidden || !wrap) return;
        const values = Array.from(wrap.querySelectorAll('input[data-run-name]'))
            .map((inp) => String(inp.value || '').trim());
        hidden.value = values.some(Boolean) ? values.join(',') : '';
    },

    /**
     * Validates dynamic run name inputs to ensure there are no duplicates within the same launch attempt.
     * @returns {boolean} True if all run names are valid (unique or empty), false otherwise.
     */
    _validateRunNameInputs() {
        const wrap = document.getElementById('run-name-inputs');
        const errEl = document.getElementById('run-name-inputs-error');
        if (!wrap || !errEl) return true;

        const values = Array.from(wrap.querySelectorAll('input[data-run-name]'))
            .map((inp) => String(inp.value || '').trim())
            .filter(Boolean);
        const seen = new Set();
        const duplicates = new Set();
        values.forEach((name) => {
            const key = name.toLowerCase();
            if (seen.has(key)) duplicates.add(name);
            seen.add(key);
        });

        if (duplicates.size) {
            errEl.textContent = `Duplicate run names are not allowed in the same launch: ${Array.from(duplicates).join(', ')}`;
            errEl.classList.remove('hidden');
            return false;
        }

        errEl.classList.add('hidden');
        errEl.textContent = '';
        return true;
    },

    /**
     * Generates individual run name input fields based on currently selected aggregation methods.
     * Preserves previously entered values seamlessly.
     * @returns {void}
     */
    _syncRunNameInputs() {
        const wrap = document.getElementById('run-name-inputs');
        const hidden = document.getElementById('cfg-run_names');
        const hint = document.getElementById('run-name-inputs-hint');
        if (!wrap || !hidden) return;

        const selected = this._selectedMethodsWithLabels();
        const previousByMethod = {};
        wrap.querySelectorAll('input[data-method]').forEach((inp) => {
            const method = String(inp.getAttribute('data-method') || '');
            if (method) previousByMethod[method] = String(inp.value || '').trim();
        });
        const hiddenValues = String(hidden.value || '').split(',').map((v) => String(v || '').trim());

        if (!selected.length) {
            wrap.innerHTML = '<p class="text-[11px] text-slate-500">Select at least one method to define run names.</p>';
            hidden.value = '';
            this._validateRunNameInputs();
            return;
        }

        wrap.innerHTML = selected.map((item, idx) => {
            const value = previousByMethod[item.value] ?? hiddenValues[idx] ?? '';
            return `
                <div>
                    <label class="block text-[11px] text-slate-400 mb-1">Run name for ${this._escapeHtml(item.label)}</label>
                    <input
                        id="cfg-run_name_${idx}"
                        type="text"
                        class="fl-input"
                        data-run-name="1"
                        data-method="${this._escapeHtml(item.value)}"
                        maxlength="80"
                        placeholder="e.g. baseline_${idx + 1}"
                        value="${this._escapeHtml(value)}"
                    >
                </div>
            `;
        }).join('');

        if (hint) {
            hint.textContent = selected.length > 1
                ? `One input per method (${selected.length} selected). Names must be unique inside this launch.`
                : 'Provide a run name to save and monitor exactly that name.';
        }

        wrap.querySelectorAll('input[data-run-name]').forEach((inp) => {
            inp.addEventListener('input', () => {
                this._syncRunNamesHiddenFromInputs();
                this._validateRunNameInputs();
            });
        });

        this._syncRunNamesHiddenFromInputs();
        this._validateRunNameInputs();
    },

    /**
     * Computes and renders a dynamic warning about GP search-space complexity.
     * @returns {void}
     */
    _updateGppWarning() {
        const box = document.getElementById('gp-population-warning');
        if (!box) return;

        if (this._state.setupMode !== 'advanced') {
            box.classList.add('hidden');
            return;
        }

        const methodsEl = document.getElementById('cfg-method');
        const hasGp = methodsEl
            ? Array.from(methodsEl.selectedOptions || []).some(o => String(o.value || '').toLowerCase().includes('gp'))
            : false;
        if (!hasGp) {
            box.classList.add('hidden');
            return;
        }

        const individuals = parseInt(document.getElementById('cfg-individuals')?.value || '0', 10) || 0;
        const generations = parseInt(document.getElementById('cfg-generations')?.value || '0', 10) || 0;
        const maxTree = parseFloat(document.getElementById('cfg-max_tree_size')?.value || '0') || 0;
        const complexity = Math.max(0, individuals) * Math.max(0, generations) * Math.max(1, maxTree);

        let tone = 'border-emerald-200 bg-emerald-50 text-emerald-900';
        let headline = 'Current GP load is moderate';
        let detail = 'Configuration is balanced; response time and resource usage should remain stable.';
        if (complexity >= 20000) {
            tone = 'border-red-200 bg-red-50 text-red-900';
            headline = 'Warning: very high GP complexity';
            detail = 'Search-space complexity grows drastically. Expect longer response time and higher CPU/RAM usage.';
        } else if (complexity >= 6000) {
            tone = 'border-amber-200 bg-amber-50 text-amber-900';
            headline = 'Attention: GP complexity is rising';
            detail = 'Search space is growing significantly; latency and resource usage may increase.';
        }

        // torch_median is only registered with arity >= 3, so with < 3 clients it is silently
        // dropped (median of 2 values = their mean). Warn the user before they launch.
        const numClients = parseInt(document.getElementById('cfg-num_clients')?.value || '0', 10) || 0;
        const primitives = String(document.getElementById('cfg-available_primitives')?.value || '');
        const medianIgnored = primitives.includes('torch_median') && numClients < 3;

        box.className = `setup-field setup-field-wide rounded-lg border px-2.5 py-1.5 text-[11px] leading-tight ${tone}`;
        box.innerHTML = `
            <p class="font-semibold">${headline}</p>
            <p class="mt-0.5">GPP estimate: individuals (${individuals}) x generations (${generations}) x max_tree_size (${maxTree}) = <strong>${complexity.toLocaleString()}</strong></p>
            <p class="mt-0.5">${detail}</p>
            ${medianIgnored ? '<p class="mt-0.5 font-semibold">Note: torch_median needs at least 3 clients (median of 2 values equals their mean); it will be ignored with the current client count.</p>' : ''}
        `;
    },

    /**
     * Return the fixed GP preset used when FedGP is selected in Simple mode.
     * @returns {Record<string, string|number|boolean>} Simple-mode FedGP preset.
     */
    _getSimpleModeFedGpPreset() {
        return {
            individuals: 50,
            generations: 20,
            mutation_rate: 0.4,
            crossover_rate: 0.6,
            elitism_size: 1,
            gp_patience: 4,
            max_tree_size: 5,
            min_tree_size: false,
            mutation_subtree_maxsize: 2,
            gp_fitness_metric: 'accuracy',
            gp_initialization: 'genHalfAndHalf',
            gp_transfer_learning: 'full_reuse',
            mutation_type: 'mutUniform',
            crossover_type: 'cxOnePoint',
            selection_type: 'selTournament',
            available_primitives: 'torch.add,torch.sub,torch.mul,torch_protected_div,torch_mean,torch_median,torch.abs,torch_protected_sqrt',
        };
    },

    /**
     * Return whether the active method selection currently includes FedGP.
     * @returns {boolean} Whether FedGP is selected.
     */
    _setupHasFedGpSelected() {
        const methodsEl = document.getElementById('cfg-method');
        const selected = methodsEl
            ? Array.from(methodsEl.selectedOptions || []).map((opt) => String(opt.value || '').toLowerCase())
            : [];
        return selected.some((method) => method === 'fed_gp');
    },

    /**
     * Force the Simple-mode FedGP preset onto hidden GP controls when needed.
     * @returns {void}
     */
    _applySimpleModeFedGpPreset() {
        if (this._state.setupMode !== 'simple' || !this._setupHasFedGpSelected()) return;
        const preset = this._getSimpleModeFedGpPreset();
        Object.entries(preset).forEach(([key, value]) => {
            if (key === 'min_tree_size') {
                const enabled = document.getElementById('cfg-min_tree_size_enabled');
                const input = document.getElementById('cfg-min_tree_size');
                if (enabled) enabled.checked = value !== false;
                if (input) {
                    input.disabled = value === false;
                    if (value !== false) input.value = value;
                }
                return;
            }
            const el = document.getElementById(`cfg-${key}`);
            if (!el) return;
            el.value = value;
        });
    },

    /**
     * Updates the inline note that explains FedGP behavior in Simple mode.
     * @returns {void}
     */
    _updateSimpleModeFedGpNotice() {
        const box = document.getElementById('simple-gp-mode-note');
        if (!box) return;
        if (this._state.setupMode !== 'simple' || !this._setupHasFedGpSelected()) {
            box.classList.add('hidden');
            box.textContent = '';
            return;
        }
        const preset = this._getSimpleModeFedGpPreset();
        box.textContent = `Simple mode uses the default FedGP preset: ${preset.individuals} individuals, ${preset.generations} generations, mutation ${preset.mutation_rate}, crossover ${preset.crossover_rate}, max tree size ${preset.max_tree_size}, transfer learning ${preset.gp_transfer_learning}. Switch to Advanced mode to customize GP search parameters.`;
        box.classList.remove('hidden');
    },

    /**
     * Resolve the most appropriate setup mode for one loaded scenario/config payload.
     * @param {Record<string, *>} cfg Loaded configuration object.
     * @returns {'simple'|'advanced'} Target setup mode.
     */
    _inferSetupModeFromConfig(cfg) {
        const explicitMode = String(cfg?.ui_mode || '').trim().toLowerCase();
        if (explicitMode === 'simple' || explicitMode === 'advanced') {
            return explicitMode;
        }

        const preset = this._getSimpleModeFedGpPreset();
        const methods = (Array.isArray(cfg?.methods) && cfg.methods.length)
            ? cfg.methods
            : [cfg?.method].filter(Boolean);
        const hasGp = methods.some((method) => String(method || '').toLowerCase() === 'fed_gp');
        const matches = (actual, expected) => {
            if (typeof expected === 'boolean') return Boolean(actual) === expected;
            if (typeof expected === 'number') return Number(actual) === expected;
            return String(actual ?? '').trim() === String(expected);
        };
        const differs = (value, expected) => value !== undefined && value !== null && !matches(value, expected);

        if (
            differs(cfg?.repeat, 1)
            || differs(cfg?.initial_eligible_clients, cfg?.num_clients)
            || differs(cfg?.seed, 0)
            || differs(cfg?.server_data_percentage, 0.1)
            || differs(cfg?.iid, false)
            || differs(cfg?.imbalance_rate, 0.5)
            || differs(cfg?.train_val_split, 0.9)
            || differs(cfg?.momentum, 0.9)
            || differs(cfg?.death_prob, 0.0)
            || differs(cfg?.new_client_prob, 0.0)
            || differs(cfg?.weights_sending_frequency, 3)
            || String(cfg?.round_train_schedule_mode || 'auto').trim() !== 'auto'
            || (Array.isArray(cfg?.round_train_schedule_percentages) && cfg.round_train_schedule_percentages.length > 0)
            || (typeof cfg?.round_train_schedule_percentages === 'string' && String(cfg.round_train_schedule_percentages || '').trim() !== '')
            || String(cfg?.round_client_allocation_mode || 'auto').trim() !== 'auto'
            || (Array.isArray(cfg?.round_client_allocation_percentages) && cfg.round_client_allocation_percentages.length > 0)
            || (typeof cfg?.round_client_allocation_percentages === 'string' && String(cfg.round_client_allocation_percentages || '').trim() !== '')
            || String(cfg?.custom_model_mode || 'default').trim() !== 'default'
            || String(cfg?.custom_model_code || '').trim()
        ) {
            return 'advanced';
        }

        if (hasGp) {
            const gpFields = Object.keys(preset);
            const customized = gpFields.some((field) => differs(cfg?.[field], preset[field]));
            if (customized) return 'advanced';
        }

        return 'simple';
    },

    /**
     * Updates inline warnings for valid but surprising churn/client-pool settings.
     * @returns {void}
     */
    _updateClientChurnWarnings() {
        if (this._state.setupMode !== 'advanced') {
            const poolBox = document.getElementById('client-pool-warning');
            const churnBox = document.getElementById('client-churn-warning');
            if (poolBox) {
                poolBox.classList.add('hidden');
                poolBox.textContent = '';
            }
            if (churnBox) {
                churnBox.classList.add('hidden');
                churnBox.textContent = '';
            }
            return;
        }

        const getValue = (id) => {
            const raw = document.getElementById(`cfg-${id}`)?.value;
            const value = Number(raw);
            return Number.isFinite(value) ? value : null;
        };
        const cfg = {
            num_clients: getValue('num_clients'),
            initial_eligible_clients: getValue('initial_eligible_clients'),
            death_prob: getValue('death_prob'),
            new_client_prob: getValue('new_client_prob'),
        };
        const warnings = this._getClientChurnWarnings?.(cfg) || [];

        const poolBox = document.getElementById('client-pool-warning');
        const hasAllInitiallyEligible = (
            Number.isInteger(cfg.num_clients)
            && cfg.num_clients >= 1
            && Number.isInteger(cfg.initial_eligible_clients)
            && cfg.initial_eligible_clients === cfg.num_clients
        );
        if (poolBox) {
            if (hasAllInitiallyEligible) {
                poolBox.className = 'setup-field setup-field-wide rounded-lg border px-2.5 py-1.5 text-[11px] leading-tight border-amber-200 bg-amber-50 text-amber-900';
                poolBox.textContent = 'All clients start eligible. If you later enable new_client_prob, it will have no effect until some clients first become inactive.';
                poolBox.classList.remove('hidden');
            } else {
                poolBox.classList.add('hidden');
                poolBox.textContent = '';
            }
        }

        const churnBox = document.getElementById('client-churn-warning');
        if (!churnBox) return;
        if (!warnings.length) {
            churnBox.classList.add('hidden');
            churnBox.textContent = '';
            return;
        }
        churnBox.className = 'setup-field setup-field-wide rounded-lg border px-2.5 py-1.5 text-[11px] leading-tight border-amber-200 bg-amber-50 text-amber-900';
        churnBox.textContent = warnings[0];
        churnBox.classList.remove('hidden');
    },

    /**
     * Bind one-time listeners for churn warnings and live field cleanup.
     * @returns {void}
     */
    _bindSetupValidationObservers() {
        if (this._state.setupValidationObserversBound) return;
        this._state.setupValidationObserversBound = true;

        ['num_clients', 'initial_eligible_clients', 'death_prob', 'new_client_prob'].forEach((fieldId) => {
            const el = document.getElementById(`cfg-${fieldId}`);
            if (!el) return;
            const handler = () => {
                el.classList.remove('setup-field-invalid');
                this._updateClientChurnWarnings?.();
            };
            el.addEventListener('input', handler);
            el.addEventListener('change', handler);
        });
    },

    /**
     * Initializes the method chip picker and synchronizes it with the hidden select.
     * @returns {void}
     */
    _initMethodChipPicker() {
        const sel = document.getElementById('cfg-method');
        const picker = document.getElementById('cfg-method-picker');
        if (!sel || !picker) return;

        if (!Array.from(sel.options).some(o => o.selected) && sel.options.length) {
            sel.options[0].selected = true;
        }

        picker.addEventListener('click', (e) => {
            const chip = e.target.closest('.method-chip[data-value]');
            if (!chip) return;
            const value = chip.getAttribute('data-value');
            const opt = Array.from(sel.options).find(o => o.value === value);
            if (!opt) return;
            opt.selected = !opt.selected;
            if (!Array.from(sel.options).some(o => o.selected) && sel.options.length) {
                opt.selected = true;
            }
            this._syncMethodChipPicker();
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });

        sel.addEventListener('change', () => this._syncMethodChipPicker());
        this._syncMethodChipPicker();
    },

    /**
     * Synchronizes chip active states with currently selected method values.
     * @returns {void}
     */
    _syncMethodChipPicker() {
        const sel = document.getElementById('cfg-method');
        const picker = document.getElementById('cfg-method-picker');
        if (!sel || !picker) return;
        const selected = new Set(Array.from(sel.selectedOptions).map(o => o.value));
        picker.querySelectorAll('.method-chip[data-value]').forEach(chip => {
            chip.classList.toggle('method-chip-selected', selected.has(chip.getAttribute('data-value')));
        });
    },

    /**
     * Returns the wizard steps visible in the current setup mode.
     * @returns {Array<{id: number, title: string}>} Visible setup steps.
     */
    _getVisibleSetupSteps() {
        const steps = this._state.setupSteps || [];
        if (this._state.setupMode === 'advanced') return steps;
        return steps.filter((step) => step.id !== 4);
    },

    /**
     * Applies setup-mode visibility rules to method selection and normalizes invalid selections.
     * @param {boolean} isSimple Whether Simple mode is active.
     * @returns {void}
     */
    _applyMethodModeVisibility(isSimple) {
        const sel = document.getElementById('cfg-method');
        const picker = document.getElementById('cfg-method-picker');
        if (!sel || !picker) return;

        Array.from(sel.options).forEach((opt) => {
            const isAdvanced = opt.getAttribute('data-advanced-method') === 'true';
            opt.hidden = isSimple && isAdvanced;
            opt.disabled = isSimple && isAdvanced;
            if (isSimple && isAdvanced) opt.selected = false;
        });

        const selected = Array.from(sel.selectedOptions || []);
        if (isSimple && !selected.length) {
            const fallback = Array.from(sel.options).find((opt) => !opt.disabled);
            if (fallback) fallback.selected = true;
        }

        this._syncMethodChipPicker();
    },

    /**
     * Sets setup mode and refreshes mode-dependent sections.
     * @param {string} mode Requested setup mode.
     * @returns {void}
     */
    _setSetupMode(mode) {
        const wasAdvanced = this._state.setupMode === 'advanced';
        this._state.setupMode = mode === 'advanced' ? 'advanced' : 'simple';
        if (wasAdvanced && this._state.setupMode === 'simple' && this._state.setupStep === 4) {
            this._state.setupStep = 3;
        }
        this._applySetupMode();
    },

    /**
     * Navigates setup wizard steps while keeping bounds valid.
     * @param {number} delta Step increment or decrement.
     * @returns {void}
     */
    _goSetupStep(delta) {
        const visibleSteps = this._getVisibleSetupSteps();
        if (!visibleSteps.length) return;
        const currentStep = this._state.setupStep || visibleSteps[0].id;
        const currentIndex = Math.max(0, visibleSteps.findIndex((step) => step.id === currentStep));

        if (delta > 0) {
            const stepCheck = this._validateSetupStep(currentStep);
            if (!stepCheck.ok) {
                this._showSetupWarnings?.(stepCheck.issues || ['Please complete the required fields before continuing.']);
                return;
            }
            this._clearSetupWarnings?.();
        }

        const nextIndex = Math.max(0, Math.min(visibleSteps.length - 1, currentIndex + delta));
        const nextStep = visibleSteps[nextIndex]?.id ?? currentStep;
        if (delta > 0 && nextStep === 5) {
            try {
                const cfg = this._collectConfig?.();
                const check = this._validateLaunchConfig?.(cfg);
                if (check && !check.ok) {
                    this._showSetupWarnings?.(check.issues || ['Invalid setup parameters.']);
                    return;
                }
                this._clearSetupWarnings?.();
            } catch {
                this._showSetupWarnings?.(['Failed to validate setup parameters before progressing.']);
                return;
            }
        }
        this._state.setupStep = nextStep;
        this._renderSetupWizard();
    },

    /**
     * Validates fields required in the current wizard step and highlights invalid inputs.
     * @param {number} step Current wizard step.
     * @returns {{ok: boolean, issues: Array<string>}} Step validation result.
     */
    _validateSetupStep(step) {
        this._clearSetupFieldErrors?.();

        const issues = [];
        const invalidIds = new Set();
        const get = (id) => document.getElementById(`cfg-${id}`);
        const resolveInvalidTarget = (id) => {
            if (id === 'round_train_schedule_percentages') return 'round-train-schedule-manual-wrap';
            if (id === 'round_client_allocation_percentages') return 'round-client-allocation-manual-wrap';
            if (id === 'client_learning_rates') return 'client-learning-rate-manual-wrap';
            return `cfg-${id}`;
        };
        const addIssue = (id, message) => {
            issues.push(message);
            if (id) invalidIds.add(resolveInvalidTarget(id));
        };

        const requireText = (id, label) => {
            const el = get(id);
            if (!el) return;
            if (String(el.value || '').trim() === '') addIssue(id, `${label} is required.`);
        };

        const requireNumber = (id, label, opts = {}) => {
            const el = get(id);
            if (!el) return;
            // Disabled fields are not applicable to the current dataset/config; skip them.
            if (el.disabled) return;
            const raw = String(el.value || '').trim();
            if (!raw) {
                addIssue(id, `${label} is required.`);
                return;
            }
            const val = Number(raw);
            if (!Number.isFinite(val)) {
                addIssue(id, `${label} must be a valid number.`);
                return;
            }
            if (opts.integer && !Number.isInteger(val)) {
                addIssue(id, `${label} must be an integer.`);
                return;
            }
            if (opts.min !== undefined && val < opts.min) {
                addIssue(id, `${label} must be >= ${opts.min}.`);
            }
            if (opts.max !== undefined && val > opts.max) {
                addIssue(id, `${label} must be <= ${opts.max}.`);
            }
        };
        let cfg;
        try {
            cfg = this._collectConfig?.() || null;
        } catch {
            cfg = null;
        }

        if (step >= 2) {
            const methodsEl = document.getElementById('cfg-method');
            const selectedMethods = methodsEl ? Array.from(methodsEl.selectedOptions || []).map(o => o.value) : [];
            const hasGp = selectedMethods.some(m => m.includes('fed_gp'));

            if (methodsEl && !selectedMethods.length) {
                issues.push('Select at least one aggregation method.');
            }
            requireText('dataset_name', 'Dataset');
            requireText('evaluation_split_mode', 'Evaluation split');
            requireNumber('num_clients', 'Clients', { integer: true, min: 1 });
            requireNumber('initial_eligible_clients', 'Initial eligible', { integer: true, min: 1 });
            requireNumber('seed', 'Seed', { integer: true, min: 0 });
            requireNumber('repeat', 'Repeat', { integer: true, min: 1 });
            if (!this._validateRunNameInputs()) {
                issues.push('Fix duplicate run names before continuing.');
            }

            const datasetName = String(get('dataset_name')?.value || '').trim();
            if (datasetName === 'custom') {
                if (!String(this._state.customDatasetRef || '').trim()) {
                    issues.push('Upload or select a custom dataset before continuing.');
                }
            }
            if (cfg) {
                (this._validateClientPoolRules?.(cfg) || []).forEach((issue) => addIssue(issue.field, issue.message));
            }
        }

        if (step >= 3) {
            requireNumber('batch_size', 'Batch size', { integer: true, min: 1 });
            requireNumber('server_learning_rate', 'Server learning rate', { min: 0 });
            requireNumber('momentum', 'Momentum', { min: 0 });
            requireNumber('server_data_percentage', 'Server data percentage', { min: 0 });
            requireNumber('imbalance_rate', 'Imbalance rate', { min: 0 });
            requireNumber('train_val_split', 'Train/Val split', { min: 0 });
            requireNumber('death_prob', 'Death probability', { min: 0, max: 0.9 });
            requireNumber('new_client_prob', 'New client probability', { min: 0, max: 0.9 });
            requireNumber('server_warmup_epochs', 'Server warm-up epochs', { integer: true, min: 1 });
            requireNumber('local_model_epochs', 'Local epochs', { integer: true, min: 1 });
            requireNumber('weights_sending_frequency', 'Weights frequency', { integer: true, min: 1 });
            if (cfg && (!Number.isFinite(Number(cfg.learning_rate)) || Number(cfg.learning_rate) <= 0)) {
                addIssue('client_learning_rates', 'learning_rate must be > 0.');
            }
            if (cfg && (!Number.isFinite(Number(cfg.server_learning_rate)) || Number(cfg.server_learning_rate) <= 0)) {
                addIssue('server_learning_rate', 'server_learning_rate must be > 0.');
            }
            if (cfg) {
                (this._validateChurnProbabilityRules?.(cfg) || []).forEach((issue) => addIssue(issue.field, issue.message));
                (this._validateClientLearningRateRules?.(cfg) || []).forEach((issue) => addIssue(issue.field, issue.message));
                (this._validateEpochFrequencyRule?.(cfg) || []).forEach((issue) => addIssue(issue.field, issue.message));
                (this._validateRoundTrainScheduleRules?.(cfg) || []).forEach((issue) => addIssue(issue.field, issue.message));
                (this._validateRoundClientAllocationRules?.(cfg) || []).forEach((issue) => addIssue(issue.field, issue.message));
                (this._validateGpConfigRules?.(cfg) || []).forEach((issue) => addIssue(issue.field, issue.message));
            }
        }

        if (step >= 4) {
            const mode = String(get('custom_model_mode')?.value || 'default');
            if (mode !== 'default') {
                const code = this._state.editors?.model
                    ? this._state.editors.model.getValue()
                    : (get('custom_model_code')?.value || '');
                if (!String(code || '').trim()) {
                    issues.push('Model code is required when custom model source is enabled.');
                    invalidIds.add('cfg-custom_model_code');
                }
            }
        }

        invalidIds.forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.classList.add('setup-field-invalid');
        });

        return { ok: issues.length === 0, issues };
    },

    /**
     * Removes temporary setup input error highlighting.
     * @returns {void}
     */
    _clearSetupFieldErrors() {
        document.querySelectorAll('.setup-field-invalid').forEach((el) => el.classList.remove('setup-field-invalid'));
    },

    /**
     * Renders wizard step visibility, progress chips, and navigation button states.
     * @returns {void}
     */
    _renderSetupWizard() {
        const steps = this._getVisibleSetupSteps();
        if (!steps.length) return;
        let step = this._state.setupStep || steps[0].id;
        if (!steps.some((item) => item.id === step)) {
            step = steps[Math.min(steps.length - 1, 0)].id;
            this._state.setupStep = step;
        }
        const visibleIndex = Math.max(0, steps.findIndex((item) => item.id === step));
        const isSimple = this._state.setupMode === 'simple';
        const methodsEl = document.getElementById('cfg-method');
        const selected = methodsEl
            ? Array.from(methodsEl.selectedOptions || []).map(o => String(o.value || '').toLowerCase())
            : [];
        const hasGp = selected.some(m => m.includes('gp'));
        document.querySelectorAll('[data-wizard-step]').forEach(el => {
            const stepOk = Number(el.getAttribute('data-wizard-step')) === step;
            const isAdvancedOnly = el.getAttribute('data-advanced') === 'true';
            const isGpOnly = el.getAttribute('data-gp-only') === 'true';
            const advancedAllowed = !isSimple || !isAdvancedOnly;
            const visible = stepOk && advancedAllowed && (!isGpOnly || hasGp);
            el.classList.toggle('hidden', !visible);
        });

        const label = document.getElementById('setup-step-label');
        if (label) label.textContent = `Step ${visibleIndex + 1} of ${steps.length}`;

        const progress = document.getElementById('setup-step-progress');
        if (progress) {
            progress.innerHTML = steps.map((s, index) => {
                const active = s.id === step;
                const done = s.id < step;
                const cls = active
                    ? 'setup-step-chip setup-step-chip-active'
                    : (done ? 'setup-step-chip setup-step-chip-done' : 'setup-step-chip setup-step-chip-pending');
                return `<div class="${cls}" title="${s.title}">${index + 1}. ${s.title}</div>`;
            }).join('');
        }

        const prev = document.getElementById('setup-prev-btn');
        const next = document.getElementById('setup-next-btn');
        if (prev) prev.disabled = visibleIndex <= 0;
        if (next) {
            next.disabled = visibleIndex >= steps.length - 1;
            next.textContent = visibleIndex >= steps.length - 1 ? 'Ready' : 'Next';
        }

        const simpleBtn = document.getElementById('setup-mode-simple');
        const advBtn = document.getElementById('setup-mode-advanced');
        const isAdv = this._state.setupMode === 'advanced';
        if (simpleBtn) simpleBtn.classList.toggle('setup-mode-card-active', !isAdv);
        if (advBtn) advBtn.classList.toggle('setup-mode-card-active', isAdv);

        if (step === 5) this._renderSetupSummary();

        // CodeMirror mis-measures when its host is display:none. Whenever the
        // wizard reveals the step holding the model editor, refresh it once so
        // loaded code shows immediately instead of only after a click.
        const modelEditor = this._state?.editors?.model;
        const modelHost = document.getElementById('cfg-custom_model_code_editor');
        if (modelEditor && modelHost && modelHost.offsetParent !== null) {
            requestAnimationFrame(() => modelEditor.refresh?.());
        }
    },

    /**
     * Builds a final review summary from the current form configuration.
     * @returns {void}
     */
    _renderSetupSummary() {
        const box = document.getElementById('setup-summary-body');
        if (!box) return;
        const cfg = this._collectConfig();
        const methods = Array.isArray(cfg.methods) ? cfg.methods : [cfg.method].filter(Boolean);
        const hasGp = methods.some(m => String(m).includes('fed_gp'));
        const hasProx = methods.some(m => String(m).includes('prox'));

        const gpFields = [
            'individuals', 'generations', 'mutation_rate', 'crossover_rate', 'elitism_size',
            'gp_patience', 'max_tree_size', 'min_tree_size', 'mutation_subtree_maxsize', 'gp_fitness_metric', 'gp_initialization',
            'gp_transfer_learning', 'mutation_type', 'crossover_type', 'selection_type',
            'available_primitives'
        ];

        const entries = Object.entries(cfg).filter(([k]) => {
            if (k === 'custom_model_code' || k === 'custom_dataset_path') return false;
            // Hide GP fields if GP is not used
            if (gpFields.includes(k) && !hasGp) return false;
            // Hide 'mu' and the FedProx aggregation choice if Prox is not used
            if ((k === 'mu' || k === 'fedprox_weighted') && !hasProx) return false;
            // Hide methods list to avoid redundancy with 'method'
            if (k === 'methods') return false;
            // Hide custom model mode if default
            if (k === 'custom_model_mode' && cfg[k] === 'default') return false;
            // If it's a built-in dataset, hide all custom_* fields
            if (!['custom', 'custom_csv', 'custom_image_npz', 'custom_image_folder'].includes(cfg.dataset_name)) {
                if (k.startsWith('custom_')) return false;
            } else {
                // If it's custom, hide the internal 'dataset_name' (custom_csv/etc) to show custom_dataset_name instead
                if (k === 'dataset_name') return false;
            }
            // Hide internal fields
            if (k === 'id' || k === 'owner' || k === 'created_at' || k === 'updated_at') return false;

            return true;
        });

        const rows = entries.map(([k, v]) => {
            // When multiple methods are selected, show the full list under the
            // 'method' row (the raw 'methods' key stays hidden above). One run
            // is launched per method, so showing only the first is misleading.
            if (k === 'method' && methods.length > 1) {
                const list = methods.join(', ');
                return `<div class="text-xs"><span class="font-semibold text-slate-500">methods:</span> <span class="text-slate-800">${list}</span> <span class="text-slate-400">(${methods.length} runs, one per method)</span></div>`;
            }
            const value = Array.isArray(v)
                ? (v.length > 5 ? `${v.length} values [${v.slice(0, 3).join(', ')} ...]` : (v.length ? v.join(', ') : '[]'))
                : (v === '' || v == null ? '-' : String(v));
            return `<div class="text-xs"><span class="font-semibold text-slate-500">${k}:</span> <span class="text-slate-800">${value}</span></div>`;
        }).join('');

        const code = String(cfg.custom_model_code || '').trim();
        const modelBlock = code
            ? `<div class="mt-3"><p class="text-xs font-semibold text-slate-500 mb-1">Custom Model Code</p><pre class="modal-pre" style="max-height:180px; overflow:auto;">${this._escapeHtml(code)}</pre></div>`
            : '<div class="mt-3 text-xs text-slate-600">Custom model code: Default model in use</div>';

        const allocationMode = String(cfg.round_client_allocation_mode || 'auto').trim().toLowerCase();
        const allocationRows = Array.isArray(cfg.round_client_allocation_percentages) ? cfg.round_client_allocation_percentages : [];
        const hasTinyShares = allocationRows.some((row) => (
            Array.isArray(row)
            && row.some((value) => {
                const numeric = Number(value);
                return Number.isFinite(numeric) && numeric > 0 && numeric < 5;
            })
        ));
        const runtimeNote = `<div class="mt-3 rounded-lg border ${hasTinyShares ? 'border-amber-200 bg-amber-50 text-amber-900' : 'border-slate-200 bg-slate-50 text-slate-700'} px-3 py-2 text-xs leading-relaxed">
                <p class="font-semibold">Round client allocation runtime note</p>
                <p class="mt-1">${allocationMode === 'manual'
                    ? 'This manual allocation is a percentage plan. Actual per-client sample counts are resolved after the round subset is realized, so very small shares can still round down to 0 samples and be skipped from training and aggregation for that round.'
                    : 'Automatic client allocation reuses the prepared client split for each round. Actual per-client sample counts are still resolved discretely after the round subset is realized, so clients that land on 0 samples are skipped from training and aggregation for that round.'
                }</p>
            </div>`;

        box.innerHTML = `<div class="grid grid-cols-1 md:grid-cols-2 gap-2">${rows}</div>${modelBlock}${runtimeNote}`;
    },

    /**
     * Shows or hides an input wrapper by field id.
     * @param {string} fieldId Field suffix used in cfg-* ids.
     * @param {boolean} visible Visibility flag.
     * @returns {void}
     */
    _toggleFieldVisibility(fieldId, visible) {
        const el = document.getElementById(`cfg-${fieldId}`);
        const wrap = el?.closest('div');
        if (!wrap) return;
        wrap.style.display = visible ? '' : 'none';
    },

    /**
     * Applies method-specific visibility rules for conditional fields.
     * @returns {void}
     */
    _applyMethodConditionalFields() {
        const methodsEl = document.getElementById('cfg-method');
        const selected = methodsEl
            ? Array.from(methodsEl.selectedOptions || []).map(o => String(o.value || '').toLowerCase())
            : [];
        const isProx = selected.some(m => m.includes('prox'));
        const hasGp = selected.some(m => m.includes('gp'));
        this._toggleFieldVisibility('mu', isProx);
        this._toggleFieldVisibility('fedprox_weighted', isProx);
        const gpSection = document.getElementById('gp-settings-section');
        if (gpSection) gpSection.classList.toggle('hidden', !hasGp || this._state.setupMode !== 'advanced' || this._state.setupStep !== 3);
        this._applySimpleModeFedGpPreset?.();
        this._syncMethodChipPicker();
        this._syncRunNameInputs();
        this._updateGppWarning();
        this._updateSimpleModeFedGpNotice?.();
        this._toggleRoundTrainScheduleInputs?.();
        this._updateClientChurnWarnings?.();
        this._renderSetupWizard();
    },

    /**
     * Toggles custom model editor visibility based on selected model mode.
     * @returns {void}
     */
    _toggleCustomModelInputs() {
        const mode = document.getElementById('cfg-custom_model_mode')?.value || 'default';
        const codeWrap = document.getElementById('custom-model-code-wrap');
        const showEditor = mode !== 'default';
        if (codeWrap) codeWrap.classList.toggle('hidden', !showEditor);
        if (showEditor) {
            const ta = document.getElementById('cfg-custom_model_code');
            const current = this._state?.editors?.model?.getValue?.() ?? ta?.value ?? '';
            if (String(current).trim() === '' && ta && String(ta.defaultValue || '').trim()) {
                const seed = String(ta.defaultValue);
                ta.value = seed;
                if (this._state?.editors?.model) this._state.editors.model.setValue(seed);
            }
            // CodeMirror initialized in a hidden container can mis-measure gutter width.
            // Refresh when shown so typing starts to the right of line numbers.
            requestAnimationFrame(() => {
                this._state?.editors?.model?.refresh?.();
            });
        }
    },

    async _toggleCustomDatasetInputs() {
        const wrap = document.getElementById('custom-dataset-wrap');
        const selectWrap = document.getElementById('custom-dataset-select-wrap');
        const dsName = document.getElementById('cfg-dataset_name')?.value;
        const isCustom = dsName === 'custom';

        if (wrap) wrap.classList.toggle('hidden', !isCustom);
        if (selectWrap) selectWrap.classList.toggle('hidden', !isCustom);

        if (isCustom) {
            await this._loadCustomDatasetOptions?.();
        }
        this._toggleTrainValSplitInput(dsName);
        this._updateCustomDatasetSimpleModeWarning?.();
    },

    /**
     * Enable Train/Val Split only for built-in datasets that lack a native validation
     * split (FashionMNIST/CIFAR). MedMNIST and custom datasets carry explicit
     * train/val/test splits, so the ratio is ignored and the field is disabled.
     * @param {string} dsName Selected dataset_name value.
     * @returns {void}
     */
    _toggleTrainValSplitInput(dsName) {
        const SPLIT_DATASETS = new Set(['fashionmnist', 'cifar10', 'cifar100']);
        const applies = SPLIT_DATASETS.has(String(dsName || '').trim().toLowerCase());
        const input = document.getElementById('cfg-train_val_split');
        if (!input) return;
        const field = input.closest('.setup-field');
        input.disabled = !applies;
        if (field) {
            field.classList.toggle('setup-field-disabled', !applies);
            const help = field.querySelector('.help-dot');
            const desc = applies
                ? 'Train/Validation split ratio (e.g. 0.9). Applies only to FashionMNIST and CIFAR.'
                : 'Train/val/test splits are already defined by this dataset, so this value is ignored.';
            if (help) {
                help.setAttribute('aria-label', desc);
                help.setAttribute('data-tooltip', desc);
            }
        }
    },

    /**
     * Toggle round-level training schedule inputs based on mode and manual selection.
     * @returns {void}
     */
    _toggleRoundTrainScheduleInputs() {
        const isSimple = this._state.setupMode === 'simple';
        const simpleNote = document.getElementById('round-train-schedule-simple-note');
        const modeWrap = document.getElementById('cfg-round_train_schedule_mode')?.closest('.setup-field');
        const mode = String(document.getElementById('cfg-round_train_schedule_mode')?.value || 'auto').trim().toLowerCase();
        const manualWrap = document.getElementById('round-train-schedule-manual-wrap');
        const editingSection = document.getElementById('round-editing-section');
        const clientModeWrap = document.getElementById('cfg-round_client_allocation_mode')?.closest('.setup-field');
        const clientMode = String(document.getElementById('cfg-round_client_allocation_mode')?.value || 'auto').trim().toLowerCase();
        const clientAutoNote = document.getElementById('round-client-allocation-runtime-note');
        const clientManualWrap = document.getElementById('round-client-allocation-manual-wrap');
        const showRoundEditing = !isSimple && (mode === 'manual' || clientMode === 'manual');

        if (simpleNote) simpleNote.classList.toggle('hidden', !isSimple);
        if (modeWrap) modeWrap.classList.toggle('hidden', isSimple);
        if (manualWrap) manualWrap.classList.toggle('hidden', isSimple || mode !== 'manual');
        if (clientModeWrap) clientModeWrap.classList.toggle('hidden', isSimple);
        if (clientAutoNote) clientAutoNote.classList.toggle('hidden', isSimple || clientMode !== 'auto');
        if (clientManualWrap) clientManualWrap.classList.toggle('hidden', isSimple || clientMode !== 'manual');
        if (editingSection) editingSection.classList.toggle('hidden', !showRoundEditing);
        this._syncRoundTrainScheduleInputs?.();
        this._syncRoundClientAllocationInputs?.();
    },

    /**
     * Return the aggregation round count implied by the current setup form.
     * @returns {number} Aggregation round count or zero when unavailable.
     */
    _getSetupAggregationRoundCount() {
        const localEpochs = Number(document.getElementById('cfg-local_model_epochs')?.value || 0);
        const frequency = Number(document.getElementById('cfg-weights_sending_frequency')?.value || 0);
        if (!Number.isInteger(localEpochs) || !Number.isInteger(frequency) || frequency <= 0 || localEpochs < 1) {
            return 0;
        }
        if (localEpochs % frequency !== 0) return 0;
        return localEpochs / frequency;
    },

    /**
     * Create one weighted round-schedule preset.
     * @param {number} roundCount Number of aggregation rounds.
     * @param {'even'|'front'|'back'} kind Preset shape.
     * @returns {Array<number>} Rounded percentages summing to 100.
     */
    _buildRoundTrainSchedulePreset(roundCount, kind = 'even') {
        const totalRounds = Number(roundCount);
        if (!Number.isInteger(totalRounds) || totalRounds < 1) return [];
        if (totalRounds === 1) return [100];

        const weights = Array.from({ length: totalRounds }, (_, idx) => {
            if (kind === 'front') return totalRounds - idx;
            if (kind === 'back') return idx + 1;
            return 1;
        });
        return this._buildRoundedPercentageDistribution?.(weights, 2) || [];
    },

    /**
     * Build one rounded percentage distribution that always totals 100.
     * @param {Array<number>} weights Relative weights for each bucket.
     * @param {number} precision Decimal places to preserve.
     * @returns {Array<number>} Rounded percentages summing to 100.
     */
    _buildRoundedPercentageDistribution(weights = [], precision = 2) {
        const bucketCount = Array.isArray(weights) ? weights.length : 0;
        if (bucketCount < 1) return [];

        const scale = 10 ** Math.max(0, Math.min(6, Math.floor(Number(precision) || 0)));
        const numericWeights = weights.map((value) => Math.max(0, Number(value) || 0));
        const weightSum = numericWeights.reduce((acc, value) => acc + value, 0);
        if (weightSum <= 0) {
            const baseUnits = Math.floor((100 * scale) / bucketCount);
            const units = Array.from({ length: bucketCount }, () => baseUnits);
            let remainder = 100 * scale - baseUnits * bucketCount;
            for (let idx = 0; idx < bucketCount && remainder > 0; idx += 1, remainder -= 1) {
                units[idx] += 1;
            }
            return units.map((value) => value / scale);
        }

        const exactUnits = numericWeights.map((weight) => (weight / weightSum) * 100 * scale);
        const flooredUnits = exactUnits.map((value) => Math.floor(value));
        let remainder = Math.round(100 * scale - flooredUnits.reduce((acc, value) => acc + value, 0));
        const ranked = exactUnits
            .map((value, idx) => ({
                idx,
                fraction: value - flooredUnits[idx],
            }))
            .sort((left, right) => (
                right.fraction - left.fraction
                || left.idx - right.idx
            ));

        for (const item of ranked) {
            if (remainder <= 0) break;
            flooredUnits[item.idx] += 1;
            remainder -= 1;
        }
        for (let idx = 0; remainder > 0 && idx < flooredUnits.length; idx += 1, remainder -= 1) {
            flooredUnits[idx] += 1;
        }

        return flooredUnits.map((value) => value / scale);
    },

    /**
     * Hydrate manual round-schedule state from loaded config.
     * @param {*} value Raw loaded value.
     * @returns {void}
     */
    _hydrateRoundTrainScheduleState(value) {
        if (!Array.isArray(value)) {
            this._state.roundTrainScheduleManual = [];
            return;
        }
        this._state.roundTrainScheduleManual = value.map((item) => {
            const parsed = Number(item);
            return Number.isFinite(parsed) ? Number(parsed.toFixed(2)) : NaN;
        });
        this._syncRoundTrainScheduleInputs?.();
    },

    /**
     * Collect the current manual round-schedule percentages.
     * @returns {Array<number>} Numeric manual round schedule.
     */
    _collectRoundTrainScheduleValues() {
        const roundCount = this._getSetupAggregationRoundCount?.() || 0;
        const existing = Array.isArray(this._state.roundTrainScheduleManual)
            ? this._state.roundTrainScheduleManual
            : [];
        const values = [];
        for (let roundIdx = 0; roundIdx < roundCount; roundIdx += 1) {
            const input = document.getElementById(`cfg-round_train_schedule_${roundIdx}`);
            const raw = input ? parseFloat(input.value) : Number(existing?.[roundIdx]);
            values.push(Number.isFinite(raw) ? Number(raw.toFixed(2)) : NaN);
        }
        this._state.roundTrainScheduleManual = values;
        const hidden = document.getElementById('cfg-round_train_schedule_percentages');
        if (hidden) {
            hidden.value = values.map((value) => (Number.isFinite(value) ? Number(value.toFixed(2)) : '')).join(',');
        }
        return values;
    },

    /**
     * Apply one schedule preset to the manual round editor.
     * @param {'even'|'front'|'back'} kind Preset identifier.
     * @returns {void}
     */
    _applyRoundTrainSchedulePreset(kind = 'even') {
        const roundCount = this._getSetupAggregationRoundCount?.() || 0;
        this._state.roundTrainScheduleManual = this._buildRoundTrainSchedulePreset?.(roundCount, kind) || [];
        this._syncRoundTrainScheduleInputs?.();
    },

    /**
     * Update round-schedule summary chips.
     * @param {Array<number>} values Schedule values.
     * @returns {void}
     */
    _updateRoundTrainScheduleSummary(values = [], mode = 'manual') {
        const roundCount = this._getSetupAggregationRoundCount?.() || 0;
        const normMode = String(mode || 'manual').trim().toLowerCase();
        const isManual = normMode === 'manual';
        const isNoSplit = normMode === 'no_split';
        const autoStatusText = isNoSplit
            ? 'Full data reused every round'
            : 'Automatic coverage across all rounds';
        const autoTotalText = isNoSplit ? 'No split' : 'Automatic';
        const topRow = document.getElementById('round-train-schedule-summary-top');
        const editorRow = document.getElementById('round-train-schedule-summary-editor');
        const roundCountTopEl = document.getElementById('round-train-schedule-round-count-top');
        const totalTopEl = document.getElementById('round-train-schedule-total-top');
        const statusTopEl = document.getElementById('round-train-schedule-status-top');
        const roundCountEditorEl = document.getElementById('round-train-schedule-round-count-editor');
        const totalEditorEl = document.getElementById('round-train-schedule-total-editor');
        const statusEditorEl = document.getElementById('round-train-schedule-status-editor');
        const total = values.reduce((sum, value) => sum + (Number.isFinite(Number(value)) ? Number(value) : 0), 0);
        const missingCount = values.filter((value) => !Number.isFinite(Number(value))).length;

        if (topRow) topRow.classList.toggle('hidden', isManual || roundCount < 1);
        if (editorRow) editorRow.classList.toggle('hidden', !isManual || roundCount < 1);

        const topTotal = totalTopEl || totalEditorEl;
        const topStatus = statusTopEl || statusEditorEl;

        if (roundCountTopEl) roundCountTopEl.textContent = `${roundCount} round${roundCount === 1 ? '' : 's'}`;
        if (roundCountEditorEl) roundCountEditorEl.textContent = `${roundCount} round${roundCount === 1 ? '' : 's'}`;
        if (topTotal) {
            if (isManual) {
                topTotal.textContent = `Total: ${total.toFixed(2)}%`;
                topTotal.className = `rounded-full border px-2 py-1 ${Math.abs(total - 100) <= 1e-6 ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-amber-200 bg-amber-50 text-amber-700'}`;
            } else {
                topTotal.textContent = autoTotalText;
                topTotal.className = 'rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600';
            }
        }
        if (totalEditorEl) {
            if (isManual) {
                totalEditorEl.textContent = `Total: ${total.toFixed(2)}%`;
                totalEditorEl.className = `rounded-full border px-2 py-1 ${Math.abs(total - 100) <= 1e-6 ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-amber-200 bg-amber-50 text-amber-700'}`;
            } else {
                totalEditorEl.textContent = autoTotalText;
                totalEditorEl.className = 'rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600';
            }
        }
        if (topStatus) {
            if (roundCount < 1) {
                topStatus.textContent = 'Waiting for a valid aggregation-round count';
                topStatus.className = 'rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600';
            } else if (!isManual) {
                topStatus.textContent = autoStatusText;
                topStatus.className = 'rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600';
            } else if (missingCount > 0) {
                topStatus.textContent = `${missingCount} round${missingCount === 1 ? '' : 's'} still need a percentage`;
                topStatus.className = 'rounded-full border border-amber-200 bg-amber-50 px-2 py-1 text-amber-700';
            } else if (Math.abs(total - 100) > 1e-6) {
                topStatus.textContent = 'Adjust the entries so the full plan totals 100%';
                topStatus.className = 'rounded-full border border-amber-200 bg-amber-50 px-2 py-1 text-amber-700';
            } else {
                topStatus.textContent = 'Every round is configured and the plan totals 100%';
                topStatus.className = 'rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 text-emerald-700';
            }
        }
        if (statusEditorEl) {
            if (roundCount < 1) {
                statusEditorEl.textContent = 'Waiting for a valid aggregation-round count';
                statusEditorEl.className = 'rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600';
            } else if (!isManual) {
                statusEditorEl.textContent = autoStatusText;
                statusEditorEl.className = 'rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600';
            } else if (missingCount > 0) {
                statusEditorEl.textContent = `${missingCount} round${missingCount === 1 ? '' : 's'} still need a percentage`;
                statusEditorEl.className = 'rounded-full border border-amber-200 bg-amber-50 px-2 py-1 text-amber-700';
            } else if (Math.abs(total - 100) > 1e-6) {
                statusEditorEl.textContent = 'Adjust the entries so the full plan totals 100%';
                statusEditorEl.className = 'rounded-full border border-amber-200 bg-amber-50 px-2 py-1 text-amber-700';
            } else {
                statusEditorEl.textContent = 'Every round is configured and the plan totals 100%';
                statusEditorEl.className = 'rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 text-emerald-700';
            }
        }
        this._updateRoundClientAllocationRuntimeNote?.(
            Array.isArray(this._state.roundClientAllocationManual) ? this._state.roundClientAllocationManual : [],
            String(document.getElementById('cfg-round_client_allocation_mode')?.value || 'auto').trim().toLowerCase(),
        );
    },

    /**
     * Render manual round-schedule inputs for the current aggregation shape.
     * @returns {void}
     */
    _syncRoundTrainScheduleInputs() {
        const wrap = document.getElementById('round-train-schedule-grid');
        const hidden = document.getElementById('cfg-round_train_schedule_percentages');
        const mode = String(document.getElementById('cfg-round_train_schedule_mode')?.value || 'auto').trim().toLowerCase();
        const isSimple = this._state.setupMode === 'simple';
        const roundCount = this._getSetupAggregationRoundCount?.() || 0;
        if (!wrap) return;

        if (isSimple || mode !== 'manual') {
            wrap.innerHTML = '';
            this._state.roundTrainScheduleManual = [];
            if (hidden) hidden.value = '';
            this._updateRoundTrainScheduleSummary?.([], mode);
            return;
        }

        const existing = Array.isArray(this._state.roundTrainScheduleManual)
            ? this._state.roundTrainScheduleManual
            : [];
        const defaults = this._buildRoundTrainSchedulePreset?.(roundCount, 'even') || [];
        const values = Array.from({ length: roundCount }, (_, roundIdx) => {
            const prev = Number(existing?.[roundIdx]);
            if (Number.isFinite(prev)) return Number(prev.toFixed(2));
            return defaults?.[roundIdx] ?? NaN;
        });
        this._state.roundTrainScheduleManual = values;

        if (roundCount < 1) {
            wrap.innerHTML = `<div class="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-3 py-3 text-[11px] text-slate-500">Set Local epochs and Weights sending frequency to a valid divisible combination before editing the round coverage plan.</div>`;
            if (hidden) hidden.value = '';
            this._updateRoundTrainScheduleSummary?.([], mode);
            return;
        }

        wrap.innerHTML = values.map((value, roundIdx) => `
            <label class="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                <span class="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Round ${roundIdx + 1}</span>
                <div class="flex items-center gap-2">
                    <input id="cfg-round_train_schedule_${roundIdx}" type="number" min="0" step="0.01" class="fl-input w-28 text-right" value="${Number.isFinite(Number(value)) ? Number(value).toFixed(2) : ''}">
                    <span class="text-[11px] text-slate-500">%</span>
                </div>
            </label>
        `).join('');

        const refreshSummary = () => {
            const current = this._collectRoundTrainScheduleValues?.() || [];
            this._updateRoundTrainScheduleSummary?.(current);
        };
        wrap.querySelectorAll('input').forEach((input) => {
            input.addEventListener('input', refreshSummary);
            input.addEventListener('change', () => {
                const parsed = Number(input.value);
                if (Number.isFinite(parsed)) {
                    input.value = parsed.toFixed(2);
                }
                refreshSummary();
            });
        });

        const evenBtn = document.getElementById('round-train-schedule-even-btn');
        const frontBtn = document.getElementById('round-train-schedule-front-btn');
        const backBtn = document.getElementById('round-train-schedule-back-btn');
        if (evenBtn) evenBtn.onclick = () => this._applyRoundTrainSchedulePreset?.('even');
        if (frontBtn) frontBtn.onclick = () => this._applyRoundTrainSchedulePreset?.('front');
        if (backBtn) backBtn.onclick = () => this._applyRoundTrainSchedulePreset?.('back');

        if (hidden) hidden.value = values.join(',');
        this._updateRoundTrainScheduleSummary?.(values, mode);
    },

    /**
     * Hydrate manual per-round client allocation state from loaded config.
     * @param {*} value Raw loaded value.
     * @returns {void}
     */
    _hydrateRoundClientAllocationState(value) {
        if (!Array.isArray(value)) {
            this._state.roundClientAllocationManual = [];
            this._state.roundClientAllocationActiveRound = 0;
            return;
        }
        this._state.roundClientAllocationManual = value.map((row) => (
            Array.isArray(row)
                ? row.map((item) => {
                    const parsed = Number(item);
                    return Number.isFinite(parsed) ? Number(parsed.toFixed(2)) : NaN;
                })
                : []
        ));
        this._state.roundClientAllocationActiveRound = 0;
        this._syncRoundClientAllocationInputs?.();
    },

    /**
     * Collect the current manual per-round client allocation matrix.
     * @returns {Array<Array<number>>} Numeric round/client allocation matrix.
     */
    _collectRoundClientAllocationMatrix() {
        const roundCount = this._getSetupAggregationRoundCount?.() || 0;
        const clientCount = Math.max(1, Number(document.getElementById('cfg-num_clients')?.value || 1));
        const existing = Array.isArray(this._state.roundClientAllocationManual)
            ? this._state.roundClientAllocationManual
            : [];
        const matrix = Array.from({ length: roundCount }, (_, roundIdx) => (
            Array.from({ length: clientCount }, (_, clientIdx) => {
                const fallback = Number(existing?.[roundIdx]?.[clientIdx]);
                const raw = roundIdx === Number(this._state.roundClientAllocationActiveRound || 0)
                    ? parseFloat(document.getElementById(`cfg-round_client_allocation_${roundIdx}_${clientIdx}`)?.value || '')
                    : fallback;
                return Number.isFinite(raw) ? Number(raw.toFixed(2)) : NaN;
            })
        ));
        this._state.roundClientAllocationManual = matrix;
        const hidden = document.getElementById('cfg-round_client_allocation_percentages');
        if (hidden) hidden.value = JSON.stringify(matrix);
        return matrix;
    },

    /**
     * Return the total percentage for one client-allocation row.
     * @param {Array<number>} row Row values.
     * @returns {number} Rounded row total.
     */
    _roundClientAllocationTotal(row = []) {
        return row.reduce((sum, value) => sum + (Number.isFinite(Number(value)) ? Number(value) : 0), 0);
    },

    /**
     * Update summary labels for the current client-allocation editor state.
     * @param {Array<Array<number>>} matrix Allocation matrix.
     * @returns {void}
     */
    _updateRoundClientAllocationSummary(matrix = [], mode = 'manual') {
        const roundCount = this._getSetupAggregationRoundCount?.() || 0;
        const clientCount = Math.max(1, Number(document.getElementById('cfg-num_clients')?.value || 1));
        const activeRound = Math.max(0, Math.min(roundCount - 1, Number(this._state.roundClientAllocationActiveRound || 0)));
        const isManual = String(mode || 'manual').trim().toLowerCase() === 'manual';
        const topRow = document.getElementById('round-client-allocation-summary-top');
        const editorRow = document.getElementById('round-client-allocation-summary-editor');
        const roundCountEls = document.querySelectorAll(
            '#round-client-allocation-round-count-top, #round-client-allocation-round-count-editor',
        );
        const clientCountEls = document.querySelectorAll(
            '#round-client-allocation-client-count-top, #round-client-allocation-editor-client-count',
        );
        const activeEls = document.querySelectorAll(
            '#round-client-allocation-active-label-top, #round-client-allocation-editor-active-label',
        );
        if (topRow) topRow.classList.toggle('hidden', isManual || roundCount < 1);
        if (editorRow) editorRow.classList.toggle('hidden', !isManual || roundCount < 1);
        roundCountEls.forEach((el) => {
            el.textContent = `${roundCount} round${roundCount === 1 ? '' : 's'}`;
        });
        clientCountEls.forEach((el) => {
            el.textContent = `${clientCount} client${clientCount === 1 ? '' : 's'}`;
        });
        activeEls.forEach((activeEl) => {
            if (roundCount < 1) {
                activeEl.textContent = 'Waiting for aggregation rounds';
                activeEl.className = 'rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600';
            } else if (!isManual) {
                activeEl.textContent = 'Automatic client allocation';
                activeEl.className = 'rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600';
            } else {
                const total = this._roundClientAllocationTotal?.(matrix?.[activeRound] || []) || 0;
                activeEl.textContent = `Editing round ${activeRound + 1} of ${roundCount} (${total.toFixed(2)}%)`;
                activeEl.className = `rounded-full border px-2 py-1 ${Math.abs(total - 100) <= 1e-6 ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-amber-200 bg-amber-50 text-amber-700'}`;
            }
        });
        this._updateRoundClientAllocationRuntimeNote?.(matrix, mode);
    },

    /**
     * Update the round-client allocation runtime note with a small heuristic warning.
     * @param {Array<Array<number>>} matrix Allocation matrix.
     * @param {string} mode Allocation mode.
     * @returns {void}
     */
    _updateRoundClientAllocationRuntimeNote(matrix = [], mode = 'manual') {
        const box = document.getElementById('round-client-allocation-runtime-note');
        if (!box) return;

        if (this._state.setupMode !== 'advanced') {
            box.classList.add('hidden');
            box.textContent = '';
            return;
        }

        const isManual = String(mode || 'manual').trim().toLowerCase() === 'manual';
        const activeRound = Math.max(0, Math.min(
            (Array.isArray(matrix) ? matrix.length : 0) - 1,
            Number(this._state.roundClientAllocationActiveRound || 0),
        ));
        const currentRow = isManual && Array.isArray(matrix?.[activeRound]) ? matrix[activeRound] : [];
        const positiveShares = currentRow
            .map((value) => Number(value))
            .filter((value) => Number.isFinite(value) && value > 0);
        const smallestPositive = positiveShares.length ? Math.min(...positiveShares) : null;
        const hasSmallShares = smallestPositive != null && smallestPositive < 5;
        const hasMultiplePositive = positiveShares.length >= 2;
        const roundMode = String(document.getElementById('cfg-round_train_schedule_mode')?.value || 'auto').trim().toLowerCase();
        const roundCount = this._getSetupAggregationRoundCount?.() || 0;
        const roundPlan = roundMode === 'manual'
            ? (this._collectRoundTrainScheduleValues?.() || [])
            : (this._buildRoundTrainSchedulePreset?.(roundCount, 'even') || []);
        const currentRoundCoverage = Number(roundPlan?.[activeRound]);
        const hasTinyRoundCoverage = Number.isFinite(currentRoundCoverage) && currentRoundCoverage > 0 && currentRoundCoverage < 5;
        const hasRisk = hasTinyRoundCoverage || hasSmallShares;

        const lead = isManual
            ? 'Manual client allocation is a percentage plan. Actual per-client sample counts are resolved after the round subset is realized.'
            : 'Automatic client allocation reuses the prepared client split for each round. Actual per-client sample counts are still resolved after the round subset is realized.';
        const details = [];
        if (hasTinyRoundCoverage) {
            details.push(`This round currently covers only ${currentRoundCoverage.toFixed(2)}% of the federated training pool.`);
        }
        if (hasSmallShares) {
            details.push('One or more client shares in the current plan are below 5%, so very small shares can still round down to 0 samples.');
        } else if (hasTinyRoundCoverage && hasMultiplePositive) {
            details.push('With multiple positive client shares, a very small realized round can still round one or more clients down to 0 samples.');
        }
        details.push('Clients that land on 0 samples are skipped from training and aggregation for that round.');

        box.textContent = `${hasRisk ? 'Soft warning: ' : ''}${lead} ${details.join(' ')}`;
        box.classList.remove('hidden');
        box.className = `setup-distribution-note rounded-lg border px-2.5 py-1.5 text-[11px] leading-tight ${
            hasRisk
                ? 'border-amber-200 bg-amber-50 text-amber-900'
                : 'border-slate-200 bg-slate-50 text-slate-700'
        }`;
    },

    /**
     * Apply a client-allocation helper action.
     * @param {'even'|'copy'} kind Action identifier.
     * @returns {void}
     */
    _applyRoundClientAllocationHelper(kind = 'even') {
        const roundCount = this._getSetupAggregationRoundCount?.() || 0;
        const clientCount = Math.max(1, Number(document.getElementById('cfg-num_clients')?.value || 1));
        const activeRound = Math.max(0, Math.min(roundCount - 1, Number(this._state.roundClientAllocationActiveRound || 0)));
        const matrix = this._collectRoundClientAllocationMatrix?.() || [];
        const evenRow = this._buildRoundTrainSchedulePreset?.(clientCount, 'even') || [];

        if (kind === 'copy') {
            const source = Array.isArray(matrix?.[activeRound]) ? matrix[activeRound] : evenRow;
            this._state.roundClientAllocationManual = Array.from({ length: roundCount }, () => [...source]);
        } else {
            this._state.roundClientAllocationManual = matrix.map((row, roundIdx) => (
                roundIdx === activeRound ? [...evenRow] : row
            ));
        }
        this._syncRoundClientAllocationInputs?.();
    },

    /**
     * Render manual per-round client allocation inputs for the current round/client shape.
     * @returns {void}
     */
    _syncRoundClientAllocationInputs() {
        const wrap = document.getElementById('round-client-allocation-grid');
        const hidden = document.getElementById('cfg-round_client_allocation_percentages');
        const mode = String(document.getElementById('cfg-round_client_allocation_mode')?.value || 'auto').trim().toLowerCase();
        const isSimple = this._state.setupMode === 'simple';
        const roundCount = this._getSetupAggregationRoundCount?.() || 0;
        const clientCount = Math.max(1, Number(document.getElementById('cfg-num_clients')?.value || 1));
        if (!wrap) return;
        if (isSimple || mode !== 'manual') {
            wrap.innerHTML = '';
            this._state.roundClientAllocationManual = [];
            this._state.roundClientAllocationActiveRound = 0;
            if (hidden) hidden.value = '[]';
            this._updateRoundClientAllocationSummary?.([], mode);
            return;
        }

        const existing = Array.isArray(this._state.roundClientAllocationManual)
            ? this._state.roundClientAllocationManual
            : [];
        const defaultRow = this._buildRoundedPercentageDistribution(
            Array.from({ length: clientCount }, () => 1),
            2,
        );
        const matrix = Array.from({ length: roundCount }, (_, roundIdx) => (
            Array.from({ length: clientCount }, (_, clientIdx) => {
                const prev = existing?.[roundIdx]?.[clientIdx];
                if (Number.isFinite(Number(prev))) return Number(Number(prev).toFixed(2));
                return roundCount > 0 && clientCount > 0
                    ? Number(defaultRow?.[clientIdx] ?? 0)
                    : 0;
            })
        ));
        this._state.roundClientAllocationManual = matrix;
        const activeRound = Math.max(0, Math.min(roundCount - 1, Number(this._state.roundClientAllocationActiveRound || 0)));
        this._state.roundClientAllocationActiveRound = activeRound;

        if (roundCount < 1) {
            wrap.innerHTML = `<div class="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-3 py-3 text-[11px] text-slate-500">Set Local epochs and Weights sending frequency to a valid divisible combination before editing the round client plan.</div>`;
            if (hidden) hidden.value = JSON.stringify(matrix);
            this._updateRoundClientAllocationSummary?.(matrix, mode);
            return;
        }

        const overview = matrix.map((row, roundIdx) => {
            const total = this._roundClientAllocationTotal?.(row) || 0;
            const isActive = roundIdx === activeRound;
            const tone = Math.abs(total - 100) <= 1e-6
                ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                : 'border-amber-200 bg-amber-50 text-amber-700';
            return `
                <button type="button" data-round-allocation-chip="${roundIdx}" class="rounded-full border px-2 py-1 text-[11px] ${isActive ? 'border-blue-300 bg-blue-50 text-blue-700' : tone}">
                    Round ${roundIdx + 1}: ${total.toFixed(2)}%
                </button>
            `;
        }).join('');
        const currentRow = matrix?.[activeRound] || [];
        const currentTotal = this._roundClientAllocationTotal?.(currentRow) || 0;
        const fields = currentRow.map((value, clientIdx) => `
            <label class="flex flex-col gap-1 rounded-lg border border-slate-200 bg-slate-50 px-2 py-2 text-[11px] text-slate-500">
                <span>Client ${clientIdx + 1}</span>
                <input id="cfg-round_client_allocation_${activeRound}_${clientIdx}" type="number" min="0" step="0.01" class="fl-input" value="${Number.isFinite(Number(value)) ? Number(value).toFixed(2) : ''}">
            </label>
        `).join('');

        wrap.innerHTML = `
            <div class="round-client-allocation-layout">
                <div class="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                    <div class="mb-2 flex flex-wrap items-center justify-between gap-2">
                        <div>
                            <p class="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Round Overview</p>
                            <p class="text-[11px] text-slate-500">Use the chips to jump between rounds. This stays compact even when the run has many rounds.</p>
                        </div>
                        <div class="flex flex-wrap gap-2">${overview}</div>
                    </div>
                </div>
                <div class="rounded-lg border border-slate-200 bg-white px-3 py-3">
                    <div class="mb-3 flex items-center justify-between gap-3">
                        <div>
                            <p class="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Editing Round ${activeRound + 1}</p>
                            <p class="text-[11px] text-slate-500">These percentages are defined for the full planned client pool before churn is applied.</p>
                        </div>
                        <span id="round-client-allocation-total-active" class="text-[11px] ${Math.abs(currentTotal - 100) <= 1e-6 ? 'text-emerald-600' : 'text-amber-600'}">Total: ${currentTotal.toFixed(2)}%</span>
                    </div>
                    <div class="round-client-allocation-fields">${fields}</div>
                </div>
            </div>
        `;

        wrap.querySelectorAll('[data-round-allocation-chip]').forEach((btn) => {
            btn.addEventListener('click', () => {
                this._collectRoundClientAllocationMatrix?.();
                this._state.roundClientAllocationActiveRound = Number(btn.getAttribute('data-round-allocation-chip') || 0);
                this._syncRoundClientAllocationInputs?.();
            });
        });
        wrap.querySelectorAll('input').forEach((input) => {
            input.addEventListener('input', () => {
                const current = this._collectRoundClientAllocationMatrix?.() || [];
                const total = this._roundClientAllocationTotal?.(current?.[activeRound] || []) || 0;
                const totalEl = document.getElementById('round-client-allocation-total-active');
                if (totalEl) {
                    totalEl.textContent = `Total: ${total.toFixed(2)}%`;
                    totalEl.classList.toggle('text-emerald-600', Math.abs(total - 100) <= 1e-6);
                    totalEl.classList.toggle('text-amber-600', Math.abs(total - 100) > 1e-6);
                }
                this._updateRoundClientAllocationSummary?.(current);
                wrap.querySelectorAll('[data-round-allocation-chip]').forEach((chip, idx) => {
                    if (idx !== activeRound) return;
                    chip.textContent = `Round ${activeRound + 1}: ${total.toFixed(2)}%`;
                    chip.className = `rounded-full border px-2 py-1 text-[11px] ${Math.abs(total - 100) <= 1e-6 ? 'border-blue-300 bg-blue-50 text-blue-700' : 'border-blue-300 bg-blue-50 text-blue-700'}`;
                });
            });
            input.addEventListener('change', () => {
                const parsed = Number(input.value);
                if (Number.isFinite(parsed)) {
                    input.value = parsed.toFixed(2);
                }
                const current = this._collectRoundClientAllocationMatrix?.() || [];
                this._updateRoundClientAllocationSummary?.(current, mode);
            });
        });
        const allocationEvenBtn = document.getElementById('round-client-allocation-even-btn');
        const allocationCopyBtn = document.getElementById('round-client-allocation-copy-btn');
        if (allocationEvenBtn) allocationEvenBtn.onclick = () => this._applyRoundClientAllocationHelper?.('even');
        if (allocationCopyBtn) allocationCopyBtn.onclick = () => this._applyRoundClientAllocationHelper?.('copy');
        if (hidden) hidden.value = JSON.stringify(matrix);
        this._updateRoundClientAllocationSummary?.(matrix, mode);
    },

    /**
     * Shows a concise warning for ZIP image-folder datasets in Simple mode.
     * @returns {void}
     */
    _updateCustomDatasetSimpleModeWarning() {
        const box = document.getElementById('custom-dataset-simple-warning');
        if (!box) return;

        const isSimple = this._state.setupMode === 'simple';
        const isImageFolder = this._state.customDatasetType === 'image_folder';
        if (!isSimple || !isImageFolder) {
            box.textContent = '';
            box.classList.add('hidden');
            return;
        }

        box.textContent = 'ZIP image-folder datasets may not match the default image model assumptions in Simple mode. If your images have a different shape or preprocessing needs, switch to Advanced mode and provide a custom model that fits your data.';
        box.classList.remove('hidden');
    },

    /**
     * Uploads a custom dataset file to backend and stores resulting reference.
     * @param {File} file Uploaded file.
     * @returns {Promise<void>}
     */
    async _uploadCustomDatasetFromFile(file, opts = {}) {
        if (!file) return;
        const description = String(opts?.description || '').trim();
        const isZipDataset = /\.zip$/i.test(String(file.name || ''));
        const extOk = /\.(csv|tsv|npz|zip)$/i.test(file.name || '');
        if (!extOk) {
            this._showToast('Only .csv, .tsv, .npz, and .zip files are supported', 'warn');
            return;
        }
        if (this._state.customDatasetUpload?.active) {
            this._showToast('Another dataset upload is already in progress.', 'warn');
            return;
        }

        const controller = new AbortController();
        this._state.customDatasetUpload = {
            active: true,
            controller,
            cancelRequested: false,
            cancelReason: '',
            progressPct: 0,
            fileName: String(file.name || 'dataset file'),
        };
        this._openDatasetUploadModal(this._state.customDatasetUpload.fileName);

        const setProgress = (pct, stage) => {
            if (!this._state.customDatasetUpload?.active) return;
            this._state.customDatasetUpload.progressPct = pct;
            this._updateDatasetUploadModal(pct, stage);
        };
        let uploadIdleTimer = null;
        let lastUploadProgressAt = Date.now();
        const startUploadIdleHint = () => {
            if (uploadIdleTimer) return;
            uploadIdleTimer = window.setInterval(() => {
                const task = this._state.customDatasetUpload;
                if (!task?.active) return;
                if ((Date.now() - lastUploadProgressAt) < 4000) return;
                const current = Number(task.progressPct || 0);
                if (current >= 96) return;
                const slowText = isZipDataset
                    ? 'Uploading large ZIP archive to server... this can pause briefly on slower disks or networks.'
                    : 'Uploading large dataset to server... this can take a while.';
                this._updateDatasetUploadModal(current, slowText);
            }, 1000);
        };
        const stopUploadIdleHint = () => {
            if (!uploadIdleTimer) return;
            window.clearInterval(uploadIdleTimer);
            uploadIdleTimer = null;
        };
        let zipProcessingTimer = null;
        let zipPulseDirection = 1;
        const startZipProcessingPulse = () => {
            if (!isZipDataset || zipProcessingTimer) return;
            zipProcessingTimer = window.setInterval(() => {
                const task = this._state.customDatasetUpload;
                if (!task?.active) return;
                const current = Number(task.progressPct || 96);
                let next = current + (zipPulseDirection * 0.8);
                if (next >= 99) {
                    next = 99;
                    zipPulseDirection = -1;
                } else if (next <= 96) {
                    next = 96;
                    zipPulseDirection = 1;
                }
                setProgress(next, 'Upload complete. Processing ZIP archive on server. Large archives can take a while...');
            }, 700);
        };
        const stopZipProcessingPulse = () => {
            if (!zipProcessingTimer) return;
            window.clearInterval(zipProcessingTimer);
            zipProcessingTimer = null;
        };

        try {
            setProgress(5, 'Preparing upload...');
            startUploadIdleHint();
            const out = await api.uploadCustomDatasetFileWithProgress(file, {
                signal: controller.signal,
                onProgress: (ratio) => {
                    lastUploadProgressAt = Date.now();
                    const normalized = Math.max(0, Math.min(1, Number(ratio) || 0));
                    if (isZipDataset && normalized >= 0.999) {
                        stopUploadIdleHint();
                        setProgress(96, 'Upload complete. Processing ZIP archive on server...');
                        startZipProcessingPulse();
                        return;
                    }
                    setProgress(8 + normalized * 88, 'Uploading to server...');
                },
                onUploadComplete: () => {
                    lastUploadProgressAt = Date.now();
                    stopUploadIdleHint();
                    if (!isZipDataset) return;
                    setProgress(96, 'Upload complete. Processing ZIP archive on server...');
                    startZipProcessingPulse();
                },
                description,
            });

            stopUploadIdleHint();
            stopZipProcessingPulse();
            setProgress(100, isZipDataset ? 'ZIP dataset ready.' : 'Validating and finalizing...');
            this._state.customDatasetRef = String(out.dataset_ref || '');
            const rawName = String(out.dataset_name || file.name);
            this._state.customDatasetFileName = rawName.replace(/\.(csv|tsv|npz|zip)$/i, '');
            this._state.customDatasetFormat = this._normalizeCustomDatasetFormat(out.dataset_format, out.dataset_type);
            this._state.customDatasetType = this._normalizeCustomDatasetType(out.dataset_format, out.dataset_type);
            const dsSel = document.getElementById('cfg-dataset_name');
        if (dsSel) dsSel.value = 'custom';
        const nameEl = document.getElementById('custom-dataset-file-name');
        if (nameEl) nameEl.textContent = this._state.customDatasetFileName || 'Custom dataset';
        const metaEl = document.getElementById('custom-dataset-meta');
        if (metaEl) {
                const formatLabel = this._customDatasetFormatLabel(out.dataset_format, out.dataset_type);
                metaEl.textContent = `${formatLabel} | ${out.rows || 0} samples, ${out.features || 0} features, ${out.classes || 0} classes`;
            }
            const statusEl = document.getElementById('custom-dataset-status');
            if (statusEl) {
                const formatLabel = this._customDatasetFormatLabel(out.dataset_format, out.dataset_type);
                statusEl.textContent = `Validated ${formatLabel}: ${out.rows || 0} samples, ${out.features || 0} features, ${out.classes || 0} classes.`;
                statusEl.classList.remove('hidden');
                clearTimeout(this._state.customDatasetStatusTimer);
                this._state.customDatasetStatusTimer = setTimeout(() => {
                    statusEl.classList.add('hidden');
                }, 3200);
            }
            this._updateCustomDatasetSimpleModeWarning?.();
            this._showToast('Custom dataset uploaded successfully', 'success');
            // Refresh the customized dataset dropdown with the newly uploaded dataset
            await this._toggleCustomDatasetInputs?.();
            await this._refreshDatasetsManager?.();
        } catch (e) {
            stopUploadIdleHint();
            stopZipProcessingPulse();
            const canceled = e?.name === 'AbortError' || /aborted|canceled|cancelled/i.test(String(e?.message || ''));
            if (canceled) {
                const reason = String(this._state.customDatasetUpload?.cancelReason || 'Dataset upload canceled.');
                this._showToast(reason, 'warn');
            } else {
                this._showToast(e.response?.data?.detail || 'Custom dataset upload failed', 'error');
            }
        } finally {
            stopUploadIdleHint();
            stopZipProcessingPulse();
            this._closeDatasetUploadModal();
            this._state.customDatasetUpload = null;
        }
    },

    /**
     * Ensures the uploaded-model option exists and updates its label.
     * @param {string} [fileName=''] Uploaded file name.
     * @returns {void}
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
    },

    /**
     * Toggles between simple and advanced setup modes.
     * @returns {void}
     */
    _toggleSetupMode() {
        this._state.setupMode = this._state.setupMode === 'simple' ? 'advanced' : 'simple';
        this._applySetupMode();
    },

    /**
     * Applies current setup mode to advanced sections and refreshes the wizard.
     * @returns {void}
     */
    _applySetupMode() {
        const isSimple = this._state.setupMode === 'simple';
        this._applyMethodModeVisibility(isSimple);
        this._applySimpleModeFedGpPreset?.();
        document.querySelectorAll('details[data-advanced="true"]').forEach(el => {
            const hideAdvanced = isSimple;
            el.classList.toggle('hidden', hideAdvanced);
        });
        document.querySelectorAll('[data-advanced-field="true"]').forEach((el) => {
            el.classList.toggle('hidden', isSimple);
        });
        const lbl = document.getElementById('setup-mode-label');
        if (lbl) lbl.textContent = isSimple
            ? 'Mode: Simple - baseline FL runs with fixed advanced defaults'
            : 'Mode: Advanced - full experiment control';
        if (isSimple && this._state.setupStep === 4) this._state.setupStep = 3;
        this._updateSimpleModeFedGpNotice?.();
        this._updateClientChurnWarnings?.();
        this._updateCustomDatasetSimpleModeWarning?.();
        this._toggleRoundTrainScheduleInputs?.();
        this._renderSetupWizard?.();
    },
};
