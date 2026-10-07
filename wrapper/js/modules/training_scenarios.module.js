window.AppTrainingScenariosModule = {
    /**
     * Updates the scenario delete button state for the current selection.
     * @returns {void}
     */
    _updateScenarioDeleteButton() {
        const dd = document.getElementById('scenario-dd');
        const btn = document.getElementById('scenario-delete-btn');
        if (!btn) return;
        const selected = String(dd?.value || '').trim();
        const deletable = selected.startsWith('my/');
        btn.disabled = !deletable;
        btn.setAttribute('aria-disabled', deletable ? 'false' : 'true');
        btn.setAttribute(
            'aria-label',
            deletable ? `Delete selected scenario ${selected}` : 'Delete selected user scenario'
        );
        btn.classList.toggle('hidden', false);
        btn.title = deletable
            ? `Delete ${selected}`
            : (selected ? 'Only user scenarios can be deleted' : 'Select one of your saved scenarios to delete it');
    },

    /**
     * Apply one loaded scenario/config to the setup wizard with mode-aware behavior.
     * @param {Record<string, *>} cfg Parsed scenario/config payload.
     * @param {{sourceLabel?: string, keepYamlFilename?: boolean}} [opts] Apply options.
     * @returns {Promise<void>}
     */
    async _applyLoadedScenarioConfig(cfg, opts = {}) {
        const sourceLabel = String(opts.sourceLabel || 'Scenario');
        const keepYamlFilename = !!opts.keepYamlFilename;
        const nextMode = this._inferSetupModeFromConfig?.(cfg) || 'simple';
        const previousMode = this._state.setupMode === 'advanced' ? 'advanced' : 'simple';

        if (keepYamlFilename) {
            await this._resetExperimentalSetupState?.({ keepYamlFilename: true });
        }

        this._state.setupMode = nextMode;
        this._populateFormDefaults(cfg);
        this._applyMethodConditionalFields?.();
        this._state.setupStep = 2;
        this._renderSetupWizard?.();

        if (previousMode !== nextMode) {
            const targetLabel = nextMode === 'advanced' ? 'Advanced' : 'Simple';
            this._showToast(`${sourceLabel} loaded. Switched to ${targetLabel} mode to match the configuration.`, 'info');
            return;
        }
        this._showToast(`${sourceLabel} applied`, 'success');
    },

    /**
     * Fetches available scenario names and fills scenario selector.
     * @returns {Promise<void>}
     */
    async _loadScenarioList() {
        const dd = document.getElementById('scenario-dd');
        try {
            const scenarios = await api.getScenarios();
            if (!dd) return;
            dd.innerHTML = '<option value="">— select scenario —</option>' +
                scenarios.map((s) => `<option value="${s}">${s}</option>`).join('');
            this._updateScenarioDeleteButton?.();
        } catch (e) {
            if (dd) {
                dd.innerHTML = '<option value="">— scenarios unavailable —</option>';
            }
            this._updateScenarioDeleteButton?.();
            this._reportNonBlockingIssue?.('scenario list load', e, {
                dedupeMs: 15000,
            });
        }
    },

    /**
     * Loads a YAML file from disk, parses it server-side, and applies values.
     * @returns {Promise<void>}
     */
    async _loadScenarioFromFile() {
        const inp = document.getElementById('scenario-file-input');
        const file = inp?.files?.[0];
        if (!file) { this._showToast('Choose a config.yaml file first', 'warn'); return; }
        const name = String(file.name || '');
        if (!/\.(ya?ml)$/i.test(name)) {
            this._showToast('Scenario/config file must be .yaml or .yml', 'warn');
            return;
        }
        const maxBytes = 2 * 1024 * 1024;
        if ((file.size || 0) > maxBytes) {
            this._showToast('Scenario file too large (max 2 MB)', 'warn');
            return;
        }
        try {
            const text = await file.text();
            if (!text || !text.trim()) {
                this._showToast('Config file is empty', 'error');
                return;
            }
            let cfg;
            try {
                cfg = await api.parseScenarioYaml(text);
            } catch (parseErr) {
                const detail = parseErr.response?.data?.detail || parseErr.message || 'YAML parse error';
                this._showToast(`Failed to parse YAML: ${detail}`, 'error');
                return;
            }
            if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) {
                this._showToast('Invalid config.yaml: root must be a YAML object', 'error');
                return;
            }
            const check = this._validateScenarioConfig(cfg);
            if (!check.ok) {
                this._showToast(check.message, 'error');
                return;
            }
            await this._applyLoadedScenarioConfig?.(cfg, {
                sourceLabel: `Config file "${file.name}"`,
                keepYamlFilename: true,
            });
        } catch (e) {
            const msg = e.response?.data?.detail || e.message || 'Failed to load scenario file';
            this._showToast(msg, 'error');
        } finally {
            // Reset the input value so re-selecting the same file (or any file
            // after _resetExperimentalSetupState keeps the value) still fires
            // the `change` event and reloads the config. The visible filename
            // label lives in #scenario-file-name and is preserved separately.
            if (inp) inp.value = '';
        }
    },

    /**
     * Loads selected stored scenario and applies values to setup form.
     * @returns {Promise<void>}
     */
    async _loadScenario() {
        const name = document.getElementById('scenario-dd').value;
        this._updateScenarioDeleteButton?.();
        if (!name) { this._showToast('Select a scenario first', 'warn'); return; }
        try {
            const cfg = await api.getScenario(name);
            const check = this._validateScenarioConfig(cfg);
            if (!check.ok) {
                this._showToast(check.message, 'error');
                return;
            }
            await this._applyLoadedScenarioConfig?.(cfg, {
                sourceLabel: `Scenario "${name}"`,
            });
        } catch (e) {
            const msg = e.response?.data?.detail || e.message || 'Failed to load scenario';
            this._showToast(msg, 'error');
        }
    },

    /**
     * Loads selected scenario and starts training immediately.
     * @returns {Promise<void>}
     */
    async _runScenario() {
        const name = document.getElementById('scenario-dd').value;
        if (!name) { this._showToast('Select a scenario first', 'warn'); return; }
        try {
            const cfg = await api.getScenario(name);
            const check = this._validateScenarioConfig(cfg);
            if (!check.ok) {
                this._showToast(check.message, 'error');
                return;
            }
            await this._applyLoadedScenarioConfig?.(cfg, {
                sourceLabel: `Scenario "${name}"`,
            });
        } catch (e) {
            this._reportNonBlockingIssue?.('scenario launch preparation', e, {
                toastMessage: e.response?.data?.detail || e.message || 'Failed to load scenario',
                toastType: 'error',
                dedupeMs: 5000,
            });
            return;
        }
        await this._startTraining();
    },

    /**
     * Deletes the currently selected user-owned scenario.
     * @returns {Promise<void>}
     */
    async _deleteScenario() {
        const dd = document.getElementById('scenario-dd');
        const name = String(dd?.value || '').trim();
        if (!name.startsWith('my/')) {
            this._showToast('Only your saved scenarios can be deleted', 'warn');
            this._updateScenarioDeleteButton?.();
            return;
        }

        const ok = await this._confirmDialog(`Delete scenario "${name}"? This cannot be undone.`);
        if (!ok) return;

        try {
            await api.deleteScenario(name);
            if (dd) dd.value = '';
            this._updateScenarioDeleteButton?.();
            await this._loadScenarioList();
            this._showToast(`Deleted scenario "${name}"`, 'success');
        } catch (e) {
            this._showToast(e.response?.data?.detail || e.message || 'Failed to delete scenario', 'error');
        }
    },

    /**
     * Validates minimal structural requirements for scenario configuration.
     * @param {Record<string, *>} cfg Candidate config object.
     * @returns {{ok: boolean, message: string}} Validation result.
     */
    _validateScenarioConfig(cfg) {
        if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) {
            return { ok: false, message: 'Invalid config.yaml: root must be a YAML object.' };
        }
        const required = [
            'method', 'dataset_name', 'num_clients', 'batch_size',
            'local_model_epochs', 'weights_sending_frequency', 'server_learning_rate',
        ];
        const missing = required.filter((k) => cfg[k] === undefined || cfg[k] === null || cfg[k] === '');
        if (missing.length) {
            return { ok: false, message: `Invalid config.yaml: missing required key(s): ${missing.join(', ')}` };
        }
        const hasClientRates = Array.isArray(cfg.client_learning_rates)
            ? cfg.client_learning_rates.length > 0
            : String(cfg.client_learning_rates || '').trim() !== '';
        if (!hasClientRates && (cfg.learning_rate === undefined || cfg.learning_rate === null || cfg.learning_rate === '')) {
            return {
                ok: false,
                message: 'Invalid config.yaml: define client_learning_rates (preferred) or a positive scalar learning_rate for clients.',
            };
        }
        return { ok: true, message: '' };
    },

    /**
     * Saves current setup as a named scenario.
     * @returns {Promise<void>}
     */
    async _saveScenario() {
        try {
            const config = {
                ...this._collectConfig(),
                ui_mode: this._state.setupMode === 'advanced' ? 'advanced' : 'simple',
            };
            const suggested = `scenario_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '_')}`;
            const name = await this._promptScenarioName(suggested);
            if (!name) {
                this._showToast('Save cancelled', 'info');
                return;
            }
            const res = await api.saveScenario(config, name);
            this._showToast(res.message || 'Scenario saved', 'success');
            await this._loadScenarioList();
        } catch (e) {
            const msg = e.response?.data?.detail || e.message || 'Failed to save scenario';
            this._reportNonBlockingIssue?.('scenario save', e, {
                toastMessage: msg,
                toastType: 'error',
                dedupeMs: 5000,
            });
        }
    },
};
