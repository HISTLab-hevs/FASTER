window.AppTrainingModule = {
    /**
     * Returns the normalized selected method set for a config payload.
     * @param {Record<string, *>} cfg Run config.
     * @returns {Array<string>} Selected methods in launch order.
     */
    _getLaunchMethods(cfg) {
        return (Array.isArray(cfg?.methods) && cfg.methods.length)
            ? cfg.methods
            : [cfg?.method].filter(Boolean);
    },

    /**
     * Shows a blocking overlay while launch request is in progress.
     * @returns {void}
     */
    _showLaunchOverlay() {
        const overlay = document.getElementById('launching-overlay');
        if (!overlay) return;
        overlay.classList.remove('hidden');
        overlay.setAttribute('aria-hidden', 'false');
    },

    /**
     * Hides launch blocking overlay.
     * @returns {void}
     */
    _hideLaunchOverlay() {
        const overlay = document.getElementById('launching-overlay');
        if (!overlay) return;
        overlay.classList.add('hidden');
        overlay.setAttribute('aria-hidden', 'true');
    },

    // ── Config collection ─────────────────────────────────────────────────────

    /**
     * Collects current form values and converts them to API payload shape.
     * @returns {Record<string, *>} Normalized configuration payload.
     */
    _collectConfig() {
        this._syncRunNamesHiddenFromInputs?.();
        const g   = id  => document.getElementById(`cfg-${id}`);
        const num = id  => parseFloat(g(id).value);
        const int = id  => parseInt(g(id).value, 10);
        const bol = id  => g(id).checked;
        const sanitizeText = (value, maxLen = 240) => String(value ?? '')
            .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim()
            .slice(0, maxLen);
        const str = id  => sanitizeText(g(id).value);

        const buildClientLearningRates = () => {
            const nClients = Math.max(1, int('num_clients') || 1);
            const seededLr = Number.isFinite(parseFloat(document.getElementById('cfg-learning_rate')?.value || ''))
                ? parseFloat(document.getElementById('cfg-learning_rate').value)
                : 0.001;
            const baseLr = seededLr > 0 ? seededLr : 0.001;
            const mode = String(document.getElementById('cfg-client_lr_mode')?.value || 'uniform');

            if (mode === 'manual') {
                return Array.from({ length: nClients }, (_, i) => {
                    const raw = parseFloat(document.getElementById(`cfg-client_lr_manual_${i}`)?.value || '');
                    return Number.isFinite(raw) ? raw : baseLr;
                });
            }

            if (mode === 'random') {
                return Array.from({ length: nClients }, (_, i) => {
                    const minRaw = parseFloat(document.getElementById(`cfg-client_lr_random_min_${i}`)?.value || '');
                    const maxRaw = parseFloat(document.getElementById(`cfg-client_lr_random_max_${i}`)?.value || '');
                    const minVal = Number.isFinite(minRaw) ? minRaw : baseLr * 0.5;
                    const maxVal = Number.isFinite(maxRaw) ? maxRaw : baseLr * 1.5;
                    const lo = Math.min(minVal, maxVal);
                    const hi = Math.max(minVal, maxVal);
                    const r = Math.random();
                    return lo + r * (hi - lo);
                });
            }

            const uniformRaw = parseFloat(document.getElementById('cfg-client_lr_uniform')?.value || '');
            const lr = Number.isFinite(uniformRaw) ? uniformRaw : baseLr;
            return Array.from({ length: nClients }, () => lr);
        };

        const methodsEl = g('method');
        const selectedMethods = methodsEl
            ? Array.from(methodsEl.selectedOptions || []).map(o => o.value).filter(Boolean)
            : [];

        const customModeRaw = str('custom_model_mode') || 'default';
        const customMode = (customModeRaw === 'loaded_file' || customModeRaw === 'upload_file')
            ? 'code'
            : customModeRaw;
        const gpTlRaw = String(str('gp_transfer_learning') || 'disabled').trim();
        const gpTransferLearning = gpTlRaw === 'disabled' ? false : gpTlRaw;
        const liveCode = this._state.editors?.model ? this._state.editors.model.getValue() : (str('custom_model_code') || '');
        const customCode = customMode === 'default' ? '' : (liveCode || '');

        const clientLearningRates = buildClientLearningRates();
        const scalarLearningRate = this._syncLearningRateBackingField?.()
            ?? clientLearningRates.find((value) => Number.isFinite(Number(value)) && Number(value) > 0)
            ?? 0.001;
        const scheduleMode = this._state.setupMode === 'simple'
            ? 'auto'
            : String(str('round_train_schedule_mode') || 'auto').trim().toLowerCase();
        const roundTrainSchedulePercentages = scheduleMode === 'manual'
            ? (this._collectRoundTrainScheduleValues?.() || [])
            : [];
        const clientAllocationMode = this._state.setupMode === 'simple'
            ? 'auto'
            : String(str('round_client_allocation_mode') || 'auto').trim().toLowerCase();
        const roundClientAllocationPercentages = clientAllocationMode === 'manual'
            ? this._collectRoundClientAllocationMatrix?.()
            : [];

        return {
            method:                    selectedMethods[0] || str('method'),
            methods:                   selectedMethods,
            run_names:                 str('run_names'),
            dataset_name:              str('dataset_name') === 'custom'
                ? (this._state.customDatasetType === 'image_npz'
                    ? 'custom_image_npz'
                    : (this._state.customDatasetType === 'image_folder' ? 'custom_image_folder' : 'custom_csv'))
                : str('dataset_name'),
            custom_dataset_ref:        ['custom', 'custom_csv', 'custom_image_npz', 'custom_image_folder'].includes(str('dataset_name')) ? String(this._state.customDatasetRef || '') : '',
            custom_dataset_name:       ['custom', 'custom_csv', 'custom_image_npz', 'custom_image_folder'].includes(str('dataset_name')) ? String(this._state.customDatasetFileName || '') : '',
            custom_dataset_format:     ['custom', 'custom_csv', 'custom_image_npz', 'custom_image_folder'].includes(str('dataset_name')) ? String(this._state.customDatasetFormat || '') : '',
            evaluation_split_mode:     str('evaluation_split_mode'),
            round_train_schedule_mode: scheduleMode,
            round_train_schedule_percentages: roundTrainSchedulePercentages,
            round_client_allocation_mode: clientAllocationMode,
            round_client_allocation_percentages: roundClientAllocationPercentages,
            num_clients:               int('num_clients'),
            initial_eligible_clients:  int('initial_eligible_clients'),
            server_data_percentage:    num('server_data_percentage'),
            iid:                       bol('iid'),
            imbalance_rate:            num('imbalance_rate'),
            train_val_split:           num('train_val_split'),
            batch_size:                int('batch_size'),
            server_learning_rate:      num('server_learning_rate'),
            learning_rate:             Number(scalarLearningRate),
            momentum:                  num('momentum'),
            mu:                        num('mu'),
            fedprox_weighted:          bol('fedprox_weighted'),
            death_prob:                num('death_prob'),
            new_client_prob:           num('new_client_prob'),
            seed:                      int('seed'),
            server_warmup_epochs:      int('server_warmup_epochs'),
            local_model_epochs:        int('local_model_epochs'),
            weights_sending_frequency: int('weights_sending_frequency'),
            client_learning_rates:     clientLearningRates,
            individuals:               int('individuals'),
            generations:               int('generations'),
            mutation_rate:             num('mutation_rate'),
            crossover_rate:            num('crossover_rate'),
            elitism_size:              int('elitism_size'),
            gp_patience:               int('gp_patience'),
            max_tree_size:             num('max_tree_size'),
            min_tree_size:             g('min_tree_size_enabled').checked ? num('min_tree_size') : false,
            mutation_subtree_maxsize:  int('mutation_subtree_maxsize'),
            gp_fitness_metric:         str('gp_fitness_metric'),
            gp_initialization:         str('gp_initialization'),
            gp_transfer_learning:      gpTransferLearning,
            mutation_type:             str('mutation_type'),
            crossover_type:            str('crossover_type'),
            selection_type:            str('selection_type'),
            available_primitives:      str('available_primitives'),
            repeat:                    int('repeat'),
            custom_model_mode:         customMode,
            custom_model_code:         String(customCode || '').replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, ' '),
        };
    },

    /**
     * Return the aggregation round count implied by the current config.
     * @param {Record<string, *>} cfg Run config.
     * @returns {number|null} Aggregation round count when valid.
     */
    _getAggregationRoundCount(cfg) {
        const localEpochs = Number(cfg?.local_model_epochs);
        const frequency = Number(cfg?.weights_sending_frequency);
        if (!Number.isInteger(localEpochs) || !Number.isInteger(frequency) || frequency <= 0) {
            return null;
        }
        if (localEpochs < 1 || localEpochs % frequency !== 0) {
            return null;
        }
        return localEpochs / frequency;
    },

    /**
     * Validate client-pool sizing rules shared by wizard and launch validation.
     * @param {Record<string, *>} cfg Run config.
     * @returns {Array<{field: string, message: string}>} Validation issues.
     */
    _validateClientPoolRules(cfg) {
        const issues = [];
        const totalClients = Number(cfg?.num_clients);
        const initialEligible = Number(cfg?.initial_eligible_clients);
        const seed = Number(cfg?.seed);

        if (!Number.isInteger(totalClients) || totalClients < 1) {
            issues.push({ field: 'num_clients', message: 'num_clients must be an integer >= 1.' });
        }
        if (!Number.isInteger(initialEligible) || initialEligible < 1) {
            issues.push({ field: 'initial_eligible_clients', message: 'initial_eligible_clients must be an integer >= 1.' });
        } else if (Number.isInteger(totalClients) && totalClients >= 1 && initialEligible > totalClients) {
            issues.push({ field: 'initial_eligible_clients', message: 'initial_eligible_clients cannot exceed num_clients.' });
        }
        if (!Number.isInteger(seed) || seed < 0) {
            issues.push({ field: 'seed', message: 'seed must be an integer >= 0.' });
        }

        return issues;
    },

    /**
     * Validate churn probability rules shared by wizard and launch validation.
     * @param {Record<string, *>} cfg Run config.
     * @returns {Array<{field: string, message: string}>} Validation issues.
     */
    _validateChurnProbabilityRules(cfg) {
        const issues = [];
        ['death_prob', 'new_client_prob'].forEach((field) => {
            const value = Number(cfg?.[field]);
            if (!Number.isFinite(value) || value < 0 || value > 0.9) {
                issues.push({ field, message: `${field} must be between 0.0 and 0.9.` });
            }
        });
        return issues;
    },

    /**
     * Validate client learning-rate configuration.
     * @param {Record<string, *>} cfg Run config.
     * @returns {Array<{field: string, message: string}>} Validation issues.
     */
    _validateClientLearningRateRules(cfg) {
        const issues = [];
        if (!Array.isArray(cfg?.client_learning_rates) || cfg.client_learning_rates.length !== Number(cfg.num_clients)) {
            issues.push({ field: 'client_learning_rates', message: 'client_learning_rates must provide exactly one value per client.' });
            return issues;
        }
        cfg.client_learning_rates.forEach((v, idx) => {
            if (!Number.isFinite(Number(v)) || Number(v) <= 0) {
                issues.push({ field: 'client_learning_rates', message: `client_learning_rates[${idx + 1}] must be > 0.` });
            }
        });
        return issues;
    },

    /**
     * Validate the local-epoch/synchronization relationship.
     * @param {Record<string, *>} cfg Run config.
     * @returns {Array<{field: string, message: string}>} Validation issues.
     */
    _validateEpochFrequencyRule(cfg) {
        if (Number(cfg?.local_model_epochs) % Number(cfg?.weights_sending_frequency) !== 0) {
            return [{ field: 'weights_sending_frequency', message: 'local_model_epochs must be divisible by weights_sending_frequency.' }];
        }
        return [];
    },

    /**
     * Validate the round-level training schedule controls.
     * @param {Record<string, *>} cfg Run config.
     * @returns {Array<{field: string, message: string}>} Validation issues.
     */
    _validateRoundTrainScheduleRules(cfg) {
        const issues = [];
        const mode = String(cfg?.round_train_schedule_mode || 'auto').trim().toLowerCase();
        if (!['auto', 'manual', 'no_split'].includes(mode)) {
            issues.push({ field: 'round_train_schedule_mode', message: 'Choose Automatic, Manual, or No split for the round schedule.' });
            return issues;
        }

        if (mode !== 'manual') return issues;

        const roundCount = this._getAggregationRoundCount(cfg);
        if (!Number.isInteger(roundCount) || roundCount < 1) {
            issues.push({ field: 'round_train_schedule_percentages', message: 'Set Local epochs and Weights sending frequency to a valid aggregation-round combination before editing the round schedule.' });
            return issues;
        }

        if (!Array.isArray(cfg?.round_train_schedule_percentages)) {
            issues.push({ field: 'round_train_schedule_percentages', message: 'Enter the round schedule with the round editor instead of raw text.' });
            return issues;
        }

        if (cfg.round_train_schedule_percentages.length !== roundCount) {
            issues.push({ field: 'round_train_schedule_percentages', message: `Enter one training percentage for each aggregation round. This run has ${roundCount} rounds.` });
        }

        const numericValues = cfg.round_train_schedule_percentages.map((value) => Number(value));
        numericValues.forEach((value, idx) => {
            if (!Number.isFinite(value)) {
                issues.push({ field: 'round_train_schedule_percentages', message: `Round ${idx + 1} is missing a valid percentage.` });
            } else if (value <= 0) {
                issues.push({ field: 'round_train_schedule_percentages', message: `Round ${idx + 1} must be greater than 0%.` });
            }
        });

        const total = numericValues.reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0);
        if (Math.abs(total - 100) > 1e-6) {
            issues.push({ field: 'round_train_schedule_percentages', message: 'The round schedule must total 100% across all aggregation rounds.' });
        }

        return issues;
    },

    /**
     * Validate the per-round client allocation controls.
     * @param {Record<string, *>} cfg Run config.
     * @returns {Array<{field: string, message: string}>} Validation issues.
     */
    _validateRoundClientAllocationRules(cfg) {
        const issues = [];
        const mode = String(cfg?.round_client_allocation_mode || 'auto').trim().toLowerCase();
        if (!['auto', 'manual'].includes(mode)) {
            issues.push({ field: 'round_client_allocation_mode', message: 'Choose Automatic or Manual for round-level client allocation.' });
            return issues;
        }
        if (mode !== 'manual') return issues;

        const roundCount = this._getAggregationRoundCount(cfg);
        const clientCount = Number(cfg?.num_clients);
        if (!Number.isInteger(roundCount) || roundCount < 1) {
            issues.push({ field: 'round_client_allocation_percentages', message: 'Set Local epochs and Weights sending frequency to a valid aggregation-round combination before editing client allocation.' });
            return issues;
        }
        if (!Number.isInteger(clientCount) || clientCount < 1) {
            issues.push({ field: 'round_client_allocation_percentages', message: 'Choose a valid client count before editing client allocation.' });
            return issues;
        }
        if (!Array.isArray(cfg?.round_client_allocation_percentages)) {
            issues.push({ field: 'round_client_allocation_percentages', message: 'Enter client allocation with the round editor instead of raw text.' });
            return issues;
        }
        if (cfg.round_client_allocation_percentages.length !== roundCount) {
            issues.push({ field: 'round_client_allocation_percentages', message: `Add one client-allocation row for each aggregation round. This run has ${roundCount} rounds.` });
        }

        cfg.round_client_allocation_percentages.forEach((row, roundIdx) => {
            if (!Array.isArray(row)) {
                issues.push({ field: 'round_client_allocation_percentages', message: `Round ${roundIdx + 1} is missing its client allocation.` });
                return;
            }
            if (row.length !== clientCount) {
                issues.push({ field: 'round_client_allocation_percentages', message: `Round ${roundIdx + 1} needs one percentage for each client. This run has ${clientCount} clients.` });
            }
            const numericValues = row.map((value) => Number(value));
            numericValues.forEach((value, clientIdx) => {
                if (!Number.isFinite(value)) {
                    issues.push({ field: 'round_client_allocation_percentages', message: `Round ${roundIdx + 1}, Client ${clientIdx + 1} is missing a valid percentage.` });
                } else if (value < 0) {
                    issues.push({ field: 'round_client_allocation_percentages', message: `Round ${roundIdx + 1}, Client ${clientIdx + 1} cannot be negative.` });
                }
            });
            const total = numericValues.reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0);
            if (Math.abs(total - 100) > 1e-6) {
                issues.push({ field: 'round_client_allocation_percentages', message: `Round ${roundIdx + 1} client allocation must total 100%.` });
            }
        });

        return issues;
    },

    /**
     * Return whether the current config includes FedGP.
     * @param {Record<string, *>} cfg Run config.
     * @returns {boolean} Whether FedGP is selected.
     */
    _hasFedGpSelected(cfg) {
        return this._getLaunchMethods(cfg).some((method) => String(method || '').toLowerCase() === 'fed_gp');
    },

    /**
     * Validate GP-only settings shared by wizard and launch validation.
     * @param {Record<string, *>} cfg Run config.
     * @returns {Array<{field: string, message: string}>} Validation issues.
     */
    _validateGpConfigRules(cfg) {
        const issues = [];
        if (!this._hasFedGpSelected(cfg)) return issues;

        const gpNumeric = ['individuals', 'generations', 'mutation_rate', 'crossover_rate', 'elitism_size', 'gp_patience', 'max_tree_size', 'mutation_subtree_maxsize'];
        gpNumeric.forEach((field) => {
            if (cfg[field] === undefined || cfg[field] === null || !Number.isFinite(Number(cfg[field]))) {
                issues.push({ field, message: `${field} is required for fed_gp and must be numeric.` });
            }
        });
        if (Number(cfg.mutation_rate) + Number(cfg.crossover_rate) <= 0) {
            issues.push({ field: 'mutation_rate', message: 'mutation_rate + crossover_rate must be greater than 0 for fed_gp.' });
        }
        if (!String(cfg.gp_initialization || '').trim()) {
            issues.push({ field: 'gp_initialization', message: 'gp_initialization is required for fed_gp.' });
        }
        if (!String(cfg.mutation_type || '').trim()) {
            issues.push({ field: 'mutation_type', message: 'mutation_type is required for fed_gp.' });
        }
        if (!String(cfg.selection_type || '').trim()) {
            issues.push({ field: 'selection_type', message: 'selection_type is required for fed_gp.' });
        }
        if (!String(cfg.crossover_type || '').trim()) {
            issues.push({ field: 'crossover_type', message: 'crossover_type is required for fed_gp.' });
        }
        return issues;
    },

    /**
     * Return non-blocking churn guidance for valid but surprising setups.
     * @param {Record<string, *>} cfg Run config.
     * @returns {Array<string>} Warning text.
     */
    _getClientChurnWarnings(cfg) {
        const warnings = [];
        const totalClients = Number(cfg?.num_clients);
        const initialEligible = Number(cfg?.initial_eligible_clients);
        const deathProb = Number(cfg?.death_prob);
        const newClientProb = Number(cfg?.new_client_prob);

        if (Number.isInteger(totalClients) && totalClients >= 1 && Number.isInteger(initialEligible) && initialEligible === totalClients && newClientProb > 0) {
            warnings.push('All clients start eligible, so new_client_prob has no inactive clients to reactivate until some clients first become inactive.');
        }
        if (Number.isInteger(initialEligible) && initialEligible === 1 && deathProb > 0) {
            warnings.push('Only one client starts eligible. The runtime safeguard will keep one client eligible each round, so death_prob may have limited effect until inactive clients return.');
        }
        return warnings;
    },

    /**
     * Clears setup warning panel.
     * @returns {void}
     */
    _clearSetupWarnings() {
        const box = document.getElementById('setup-warning-box');
        if (!box) return;
        box.classList.add('hidden');
        box.innerHTML = '';
    },

    /**
     * Shows setup warning panel with a detailed list of validation issues.
     * @param {Array<string>} issues Validation issue list.
     * @returns {void}
     */
    _showSetupWarnings(issues) {
        const box = document.getElementById('setup-warning-box');
        if (!box) return;
        const safe = (v) => this._escapeHtml(String(v || ''));
        box.classList.remove('hidden');
        box.innerHTML = `<p class="font-semibold mb-1">Please fix the following before continuing:</p><ul class="list-disc pl-4 space-y-0.5">${issues.map(i => `<li>${safe(this._translateSetupMessage?.(i) || i)}</li>`).join('')}</ul>`;
    },

    /**
     * Translate raw setup/config validation text into user-facing language.
     * @param {string} message Raw validation message.
     * @returns {string} User-facing validation copy.
     */
    _translateSetupMessage(message) {
        const raw = String(message || '').trim();
        if (!raw) return 'Please review the setup values and try again.';

        let match = raw.match(/^round_train_schedule_percentages count must match aggregation rounds$/);
        if (match) {
            const roundCount = this._getSetupAggregationRoundCount?.() || 0;
            return `Enter one training percentage for each aggregation round. This run has ${roundCount} rounds.`;
        }
        match = raw.match(/^round_train_schedule_percentages must sum to 100$/);
        if (match) return 'The round schedule must total 100% across all aggregation rounds.';
        match = raw.match(/^round_train_schedule_percentages must contain only positive values$/);
        if (match) return 'Every round in the manual schedule must be greater than 0%.';
        match = raw.match(/^round_train_schedule_percentages must contain valid numbers$/);
        if (match) return 'Every round in the manual schedule needs a valid percentage.';
        match = raw.match(/^round_train_schedule_percentages must contain finite numbers$/);
        if (match) return 'Every round in the manual schedule needs a regular numeric percentage.';
        match = raw.match(/^round_client_allocation_percentages count must match aggregation rounds$/);
        if (match) {
            const roundCount = this._getSetupAggregationRoundCount?.() || 0;
            return `Add one client-allocation row for each aggregation round. This run has ${roundCount} rounds.`;
        }
        match = raw.match(/^round_client_allocation_percentages\[(\d+)\] count must match num_clients$/);
        if (match) {
            const clientCount = Math.max(1, Number(document.getElementById('cfg-num_clients')?.value || 1));
            return `Round ${match[1]} needs one percentage for each client. This run has ${clientCount} clients.`;
        }
        match = raw.match(/^round_client_allocation_percentages\[(\d+)\] must sum to 100$/);
        if (match) return `Round ${match[1]} client allocation must total 100%.`;
        match = raw.match(/^round_client_allocation_percentages\[(\d+)\] must contain only non-negative values$/);
        if (match) return `Round ${match[1]} cannot contain negative client percentages.`;
        match = raw.match(/^round_client_allocation_percentages must be a list of per-round client percentage lists$/);
        if (match) return 'Enter client allocation with the round editor instead of raw text.';
        match = raw.match(/^round_client_allocation_percentages\[(\d+)\] must contain valid numbers$/);
        if (match) return `Round ${match[1]} needs valid client percentages.`;
        match = raw.match(/^round_client_allocation_percentages\[(\d+)\] must contain finite numbers$/);
        if (match) return `Round ${match[1]} needs regular numeric client percentages.`;
        match = raw.match(/^round_train_schedule_mode must be 'auto', 'manual', or 'no_split'$/);
        if (match) return 'Choose Automatic, Manual, or No split for the round schedule.';
        match = raw.match(/^round_client_allocation_mode must be 'auto' or 'manual'$/);
        if (match) return 'Choose Automatic or Manual for round-level client allocation.';
        return raw;
    },

    /**
     * Validates run configuration before launch.
     * @param {Record<string, *>} cfg Run config.
     * @returns {{ok: boolean, issues: Array<string>}} Validation result.
     */
    _validateLaunchConfig(cfg) {
        const issues = [];
        const requiredNumeric = [
            'num_clients', 'initial_eligible_clients', 'batch_size', 'server_learning_rate', 'learning_rate', 'momentum', 'death_prob',
            'new_client_prob', 'seed', 'server_warmup_epochs',
            'local_model_epochs', 'weights_sending_frequency', 'server_data_percentage',
            'imbalance_rate', 'train_val_split', 'repeat'
        ];
        const requiredText = ['method', 'dataset_name', 'evaluation_split_mode'];

        requiredText.forEach((k) => {
            if (cfg[k] === undefined || cfg[k] === null || String(cfg[k]).trim() === '') {
                issues.push(`${k} is required.`);
            }
        });

        const selectedMethods = this._getLaunchMethods(cfg);
        // Only non-empty names count. Leaving every run-name field empty is valid (names
        // are auto-generated); a partial set must provide one name per selected method.
        const runNames = String(cfg.run_names || '')
            .split(',')
            .map(v => v.trim())
            .filter(Boolean);
        if (runNames.length && runNames.length !== selectedMethods.length) {
            issues.push('Provide one run name for each selected method, or leave all run-name fields empty.');
        }
        const runNameSet = new Set();
        runNames.forEach((name, idx) => {
            const key = name.toLowerCase();
            if (runNameSet.has(key)) {
                issues.push(`run_names[${idx + 1}] duplicates another run name.`);
            }
            runNameSet.add(key);
            if (name.length > 80) issues.push(`run_names[${idx + 1}] is too long (max 80 chars).`);
            if (!/^[A-Za-z0-9_.\- ]+$/.test(name)) {
                issues.push(`run_names[${idx + 1}] contains invalid characters.`);
            }
        });

        requiredNumeric.forEach((k) => {
            const v = cfg[k];
            if (v === undefined || v === null || !Number.isFinite(Number(v))) {
                issues.push(`${k} must be a valid number.`);
            }
        });
        if (Number(cfg.learning_rate) <= 0) {
            issues.push('learning_rate must be > 0.');
        }
        if (Number(cfg.server_learning_rate) <= 0) {
            issues.push('server_learning_rate must be > 0.');
        }

        this._validateClientLearningRateRules(cfg).forEach((issue) => issues.push(issue.message));
        this._validateEpochFrequencyRule(cfg).forEach((issue) => issues.push(issue.message));
        this._validateRoundTrainScheduleRules(cfg).forEach((issue) => issues.push(issue.message));
        this._validateRoundClientAllocationRules(cfg).forEach((issue) => issues.push(issue.message));
        this._validateClientPoolRules(cfg).forEach((issue) => issues.push(issue.message));
        this._validateChurnProbabilityRules(cfg).forEach((issue) => issues.push(issue.message));

        if (['custom', 'custom_csv', 'custom_image_npz', 'custom_image_folder'].includes(cfg.dataset_name)) {
            if (!cfg.custom_dataset_ref || !String(cfg.custom_dataset_ref).trim()) {
                issues.push('custom_dataset_ref is required when using a custom dataset.');
            }
        }

        if (cfg.custom_model_mode !== 'default' && !String(cfg.custom_model_code || '').trim()) {
            issues.push('custom_model_code is required when custom_model_mode is not default.');
        }

        this._validateGpConfigRules(cfg).forEach((issue) => issues.push(issue.message));

        return { ok: issues.length === 0, issues };
    },

    // ── Training ──────────────────────────────────────────────────────────────

    /**
     * Starts one or more jobs from current configuration and initializes monitoring.
     * @returns {Promise<void>}
     */
    async _startTraining() {
        let config;
        let launchUiLocked = false;
        const startBtn = document.getElementById('start-btn');
        let shouldOpenTrainingTab = false;
        this._state.monitorStopGuardRunId = null;
        try {
            config = this._collectConfig();
        } catch {
            this._showSetupWarnings(['Failed to parse setup form values. Please review all parameters.']);
            return;
        }

        const check = this._validateLaunchConfig(config);
        if (!check.ok) {
            this._showSetupWarnings(check.issues);
            return;
        }
        this._clearSetupWarnings();

        const statusEl = document.getElementById('experiment-status');
        if (statusEl) statusEl.textContent = 'Launching…';
        this._showLaunchOverlay();
        if (startBtn) {
            startBtn.disabled = true;
            launchUiLocked = true;
        }
        try {
            if (['custom', 'custom_csv', 'custom_image_npz', 'custom_image_folder'].includes(config.dataset_name) && !config.custom_dataset_ref) {
                this._showToast('Upload a custom dataset (.csv/.tsv/.npz/.zip) before launching', 'warn');
                if (statusEl) statusEl.textContent = '';
                return;
            }

            const methods = this._getLaunchMethods(config);
            if (!methods.length) {
                this._showToast('Select at least one aggregation method', 'warn');
                if (statusEl) statusEl.textContent = '';
                return;
            }

            let startedCount = 0;
            let queuedCount = 0;
            let firstStarted = null;
            const runNamesRaw = String(config.run_names || '')
                .split(',')
                .map(v => v.trim());
            const runNames = runNamesRaw.filter(Boolean);

            if (runNames.length) {
                const conflicts = await api.validateRunNames(runNames);
                if (conflicts.length) {
                    this._showSetupWarnings([`Run name already exists in your account: ${conflicts.join(', ')}. Choose a different name before launch.`]);
                    if (statusEl) statusEl.textContent = '';
                    return;
                }
            }

            for (let methodIdx = 0; methodIdx < methods.length; methodIdx += 1) {
                const method = methods[methodIdx];
                const payload = { ...config, method };
                payload.methods = methods.slice();
                payload.run_name = runNamesRaw[methodIdx] || '';
                const out = await api.queueRun(payload);
                if (out.started) {
                    startedCount += 1;
                    if (!firstStarted) firstStarted = { runId: out.run_id, cfg: payload };
                } else {
                    queuedCount += 1;
                }
            }

            if (statusEl) statusEl.textContent = '';
            this._showToast(`Launch summary: started ${startedCount}, queued ${queuedCount}`, 'success');

            if ((startedCount + queuedCount) > 0) {
                await this._resetExperimentalSetupState?.();
            }

            if (firstStarted) {
                this._resetMonitoringView(firstStarted.cfg);
                this._updateClientEpochProgressFromLogs('');
                this._state.currentRunId   = firstStarted.runId;
                this._state.currentConfig  = firstStarted.cfg;
                this._state.trainingStatus = 'running';
                this._setTrainingButtons(true);
                this._startPolling();
                shouldOpenTrainingTab = true;
            } else {
                this._pollSystem();
            }
        } catch (e) {
            const rawMsg = e.response?.data?.detail || 'Failed to launch job';
            const msg = this._translateSetupMessage?.(rawMsg) || rawMsg;
            if (statusEl) statusEl.textContent = msg;
            this._showToast(msg, 'error');
        } finally {
            this._hideLaunchOverlay();
            if (startBtn && launchUiLocked) startBtn.disabled = false;
            if (shouldOpenTrainingTab) {
                this._switchTab(2);
            }
        }
    },

    /**
     * Resets live monitor widgets and progress state before/after run transitions.
     * @param {Record<string, *>|null} [config=null] Current run config.
     * @returns {void}
     */
    _resetMonitoringView(config = null) {
        this._state.currentConfig = config;
        this._state.lastLiveMetrics = null;
        this._state.lastClientProgressLogs = null;

        const ids = [
            'mon-metric-val-acc', 'mon-metric-val-loss',
            'mon-metric-val-precision', 'mon-metric-val-recall',
            'mon-metric-val-f1', 'mon-metric-val-roc_auc',
            'mon-metric-test-acc', 'mon-metric-test-loss',
            'mon-metric-test-precision', 'mon-metric-test-recall',
            'mon-metric-test-f1', 'mon-metric-test-roc_auc',
        ];
        ids.forEach(id => this._setMetricValue(id, null, 4, id.includes('roc_auc') ? 'N/A' : '-'));

        const logEl = document.getElementById('log-console');
        if (logEl) logEl.textContent = 'Waiting for training output...';

        const dlRow = document.getElementById('mon-download-row');
        if (dlRow) dlRow.classList.add('hidden');

        this._updateProgressBar({ aggregated_val: { accuracy: [] } });
        this._updateClientEpochProgressFromLogs('');
        this._applyMonitoringModeVisibility(config?.evaluation_split_mode || 'train_val_test', { showTest: false });

        // Start a fresh visual timeline for the next run.
        if (this._state.liveChartsInit) {
            metricsCharts.initLive();
        }
    },

    /**
     * Requests stop for current run and updates UI state.
     * @returns {Promise<void>}
     */
    async _stopTraining() {
        const runId = String(this._state.currentRunId || '').trim();
        if (runId) {
            this._state.monitorStopGuardRunId = runId;
            this._state.trainingStatus = 'stopping';
        }
        try {
            await api.stopRun(runId || '');
            this._stopPolling();
            this._state.trainingStatus = 'idle';
            this._setTrainingButtons(false);
            this._showToast('Training stopped', 'warn');
        } catch (e) {
            this._state.monitorStopGuardRunId = null;
            if (runId) this._state.trainingStatus = 'running';
            this._showToast('Failed to stop: ' + (e.response?.data?.detail || e.message), 'error');
        }
    },

    /**
     * Toggles start/stop button enablement based on running state.
     * @returns {void}
     */
    _setTrainingButtons() {
        const startBtn = document.getElementById('start-btn');
        if (!startBtn) return;
        // Launch remains available to allow queueing additional jobs.
        startBtn.disabled = false;
        startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    },

    /**
     * Hydrates monitor state for an existing run (used after browser refresh).
     * @param {string} runId Run identifier.
     * @param {Record<string, *>|null} [cfg=null] Optional run config.
     * @returns {Promise<void>}
     */
    async _hydrateMonitoringFromRun(runId, cfg = null) {
        if (!runId) return;
        try {
            const runCfg = cfg || await api.getRunConfig(runId).catch(() => ({}));
            if (runCfg && Object.keys(runCfg).length) {
                this._state.currentConfig = runCfg;
                this._applyMonitoringModeVisibility(runCfg.evaluation_split_mode || 'train_val_test', { showTest: false });
            }

            if (!this._state.liveChartsInit) {
                metricsCharts.initLive();
                this._state.liveChartsInit = true;
            }

            const metrics = await api.getMetrics(runId).catch(() => ({}));
            if (metrics && Object.keys(metrics).length) {
                this._state.lastLiveMetrics = metrics;
                metricsCharts.updateLive(metrics);
                this._updateProgressBar(metrics);
                this._updateMonitorLatestMetrics(metrics);
            }

            const logOut = await api.getLogs(runId, 160).catch(() => ({ logs: '' }));
            const logs = logOut?.logs || '';
            this._setLogConsoleText(logs);
            this._updateClientEpochProgressFromLogs(logs, this._state.lastLiveMetrics || null);

            const monStatus = document.getElementById('mon-status');
            if (monStatus) monStatus.textContent = `Run: ${runId}`;
        } catch (e) {
            this._reportNonBlockingIssue?.('run hydration', e, {
                statusId: 'mon-status',
                statusMessage: `Run: ${runId} (live data temporarily unavailable)`,
                dedupeMs: 15000,
            });
        }
    },

    // ── Polling ───────────────────────────────────────────────────────────────

    /**
     * Starts log and metrics polling loops if not already active.
     * @returns {void}
     */
    _startPolling() {
        if (!this._state.logTimer) {
            this._pollLogs();
            this._state.logTimer = setInterval(() => this._pollLogs(), 2000);
        }
        if (!this._state.metricsTimer) {
            this._pollMetrics();
            // 2s (was 5s): the metrics payload carries live_state (phase, round,
            // aggregation) and the latest aggregated metric cards, so polling at
            // the same cadence as logs keeps the monitor responsive in realtime.
            this._state.metricsTimer = setInterval(() => this._pollMetrics(), 2000);
        }
    },

    /**
     * Stops log and metrics polling loops.
     * @returns {void}
     */
    _stopPolling() {
        clearInterval(this._state.logTimer);
        clearInterval(this._state.metricsTimer);
        this._state.logTimer = this._state.metricsTimer = null;
    },

    /**
     * Stops all polling loops and associated observers.
     * @returns {void}
     */
    _stopAllPolling() {
        this._stopPolling();
        clearInterval(this._state.systemTimer);
        clearInterval(this._state.alertTimer);
        clearInterval(this._state.historyAutoLoadTimer);
        clearInterval(this._state.adminOverviewTimer);
        clearTimeout(this._state.adminSearchDebounceTimer);
        this._state.systemTimer = null;
        this._state.alertTimer = null;
        this._state.historyAutoLoadTimer = null;
        this._state.adminOverviewTimer = null;
        this._state.adminSearchDebounceTimer = null;
        this._state.monitorStopGuardRunId = null;
        this._teardownLogAutoScroll();
    },

    /**
     * Toggles monitor log console collapse state.
     * @returns {void}
     */
    _toggleLogConsole() {
        this._state.logCollapsed = !this._state.logCollapsed;
        const wrap = document.getElementById('log-console-wrap');
        const btn = document.getElementById('log-toggle-btn');
        if (wrap) wrap.classList.toggle('hidden', this._state.logCollapsed);
        if (btn) btn.textContent = this._state.logCollapsed ? 'Expand' : 'Collapse';
    },

    /**
     * Updates the log console without stealing scroll when the user is reading older lines.
     * @param {string} logs Text payload for the log console.
     * @returns {void}
     */
    _setLogConsoleText(logs) {
        const el = document.getElementById('log-console');
        if (!el || !logs) return;
        const previousScrollTop = el.scrollTop;
        const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        const shouldFollowTail = distanceFromBottom <= 24;

        el.textContent = logs;
        if (shouldFollowTail) {
            el.scrollTop = el.scrollHeight;
        } else {
            el.scrollTop = Math.min(previousScrollTop, Math.max(0, el.scrollHeight - el.clientHeight));
        }
    },

    /**
     * Clears any legacy page-level log auto-scroll observer.
     * @returns {void}
     */
    _initLogAutoScroll() {
        this._teardownLogAutoScroll();
    },

    /**
     * Tears down automatic log scroll observer.
     * @returns {void}
     */
    _teardownLogAutoScroll() {
        try {
            this._state.logResizeObserver?.disconnect();
        } catch {}
        this._state.logResizeObserver = null;
    },

    /**
     * Initializes CodeMirror editors used in setup and history modals.
     * @returns {void}
     */
    _initCodeEditors() {
        if (typeof window.CodeMirror === 'undefined') return;
        const editors = this._state.editors || (this._state.editors = {});

        if (!editors.model) {
            const ta = document.getElementById('cfg-custom_model_code');
            const host = document.getElementById('cfg-custom_model_code_editor');
            if (ta && host) {
                editors.model = CodeMirror(host, {
                    value: ta.value || '',
                    mode: 'python',
                    lineNumbers: true,
                    lineWrapping: true,
                    tabSize: 4,
                    indentUnit: 4,
                    viewportMargin: 16,
                });
                editors.model.on('change', () => {
                    ta.value = editors.model.getValue();
                });
            }
        }

        const ensureReadOnly = (key, hostId, mode) => {
            if (editors[key]) return;
            const host = document.getElementById(hostId);
            if (!host) return;
            editors[key] = CodeMirror(host, {
                value: '',
                mode,
                lineNumbers: true,
                lineWrapping: true,
                readOnly: true,
                viewportMargin: 12,
            });
        };

        ensureReadOnly('runCfg', 'hist-run-detail-config', 'yaml');
        ensureReadOnly('runModel', 'hist-run-detail-model-code', 'python');
        ensureReadOnly('modelPreview', 'hist-model-code-pre', 'python');
        if (!editors.scenarioSchema) {
            const host = document.getElementById('scenario-yaml-template-editor');
            const src = document.getElementById('scenario-yaml-template-source');
            if (host) {
                editors.scenarioSchema = CodeMirror(host, {
                    value: src?.value || '',
                    mode: 'yaml',
                    lineNumbers: true,
                    lineWrapping: true,
                    readOnly: true,
                    viewportMargin: 12,
                });
            }
        }
    },

    /**
     * Sets value for a named editor if available.
     * @param {string} key Editor registry key.
     * @param {string} [text=''] New editor content.
     * @returns {boolean} True when editor exists and was updated.
     */
    _setEditorValue(key, text = '') {
        const ed = this._state.editors?.[key];
        if (!ed) return false;
        ed.setValue(String(text || ''));
        ed.refresh();
        return true;
    },

    /**
     * Opens the current-job configuration modal.
     * @returns {void}
     */
    _openMyJobModal() {
        if (!this._state.myJobConfig) return;
        const modal = document.getElementById('myjob-modal');
        const pre = document.getElementById('myjob-config-json');
        if (!modal || !pre) return;
        pre.textContent = JSON.stringify(this._state.myJobConfig, null, 2);
        modal.classList.remove('hidden');
    },

    /**
     * Closes the current-job configuration modal.
     * @returns {void}
     */
    _closeMyJobModal() {
        const modal = document.getElementById('myjob-modal');
        if (modal) modal.classList.add('hidden');
    },

    /**
     * Polls latest server logs for the active run.
     * @returns {Promise<void>}
     */
    async _pollLogs() {
        const runId = this._state.currentRunId;
        if (!runId) return;
        try {
            // Fetch a wide tail so every client in a round stays visible: with
            // many clients × epochs plus per-round aggregation lines, a 100-line
            // window scrolls past earlier clients between 2s polls. The backend
            // caps n at 2000; the full train.log remains complete in the export.
            const { logs } = await api.getLogs(runId, 1000);
            this._setLogConsoleText(logs);
            this._updateClientEpochProgressFromLogs(logs || '', this._state.lastLiveMetrics || null);
        } catch (e) {
            this._reportNonBlockingIssue?.('run log polling', e, {
                statusId: 'mon-status',
                statusMessage: `Run: ${runId} (log stream temporarily unavailable)`,
                dedupeMs: 15000,
            });
        }
    },

    /**
     * Extracts and updates client epoch progress from textual logs.
     * @param {string} logs Log text payload.
     * @returns {void}
     */
    _updateClientEpochProgressFromLogs(logs, metrics = null) {
        const labelEl = document.getElementById('client-progress-label');
        const pctEl = document.getElementById('client-progress-pct');
        const fillEl = document.getElementById('client-progress-fill');
        if (!labelEl || !pctEl || !fillEl) return;
        if (!metrics && logs === this._state.lastClientProgressLogs) return;

        const liveState = metrics && typeof metrics === 'object' && metrics.live_state
            ? metrics.live_state
            : null;
        if (liveState && typeof liveState === 'object') {
            const phase = String(liveState.phase || '').toLowerCase();
            const configuredClients = Number(liveState.configured_total_clients || this._state.currentConfig?.num_clients || 0);
            const participatingClients = Number(liveState.participating_clients_in_round || 0);
            const activeClientPosition = Number(liveState.active_client_position || 0);
            const activeClientId = Number(liveState.active_client_id || 0);
            const epochTotal = Number(liveState.client_epoch_total || this._state.currentConfig?.weights_sending_frequency || 0);
            const epochCompleted = Math.max(0, Number(liveState.client_epoch_completed || 0));
            const roundNow = Number(liveState.current_round || 0);
            const roundTotal = Number(liveState.total_rounds || 0);
            const completedRounds = Number(liveState.completed_rounds || 0);

            if (phase === 'client_training' && activeClientId > 0) {
                const safeEpochTotal = Number.isFinite(epochTotal) && epochTotal > 0 ? Math.floor(epochTotal) : 0;
                const safeEpochCompleted = Number.isFinite(epochCompleted)
                    ? Math.min(safeEpochTotal || Math.floor(epochCompleted), Math.floor(epochCompleted))
                    : 0;
                const pct = safeEpochTotal > 0 ? Math.min(100, Math.round((safeEpochCompleted / safeEpochTotal) * 100)) : 0;
                const participantLabel = participatingClients > 0
                    ? `${Math.max(1, activeClientPosition || 1)} / ${participatingClients} participating`
                    : `Client ID ${activeClientId}`;
                const configuredLabel = configuredClients > 0 ? ` (ID ${activeClientId} of ${configuredClients} configured)` : ` (ID ${activeClientId})`;

                if (safeEpochTotal > 0 && safeEpochCompleted <= 0) {
                    labelEl.textContent = `Client ${participantLabel}${configuredLabel} — starting epoch 1 / ${safeEpochTotal}`;
                } else {
                    labelEl.textContent = `Client ${participantLabel}${configuredLabel} — Epoch ${safeEpochCompleted} / ${safeEpochTotal || '—'}`;
                }
                pctEl.textContent = `${pct}%`;
                fillEl.style.width = `${pct}%`;
                return;
            }

            if (phase === 'aggregation') {
                const text = roundTotal > 0
                    ? `Aggregating global model for round ${Math.max(1, roundNow)} / ${roundTotal}`
                    : 'Aggregating global model...';
                labelEl.textContent = text;
                pctEl.textContent = '100%';
                fillEl.style.width = '100%';
                return;
            }

            if (phase === 'round_complete') {
                const text = roundTotal > 0
                    ? `Round ${completedRounds} / ${roundTotal} complete — preparing next client set...`
                    : 'Round complete — preparing next client set...';
                labelEl.textContent = text;
                pctEl.textContent = '100%';
                fillEl.style.width = '100%';
                return;
            }

            if (phase === 'completed') {
                labelEl.textContent = 'Client training complete for all configured rounds.';
                pctEl.textContent = '100%';
                fillEl.style.width = '100%';
                return;
            }
        }

        if (!logs || !this._state.currentConfig) {
            labelEl.textContent = 'Client progress waiting...';
            pctEl.textContent = '0%';
            fillEl.style.width = '0%';
            this._state.lastClientProgressLogs = logs;
            return;
        }

        const clientsTotal = Number(this._state.currentConfig.num_clients || 0);
        const defaultEpochs = Number(this._state.currentConfig.weights_sending_frequency || 0);

        const clientRegex = /Training Client\s+(\d+)/g;
        let clientMatch;
        let lastClient = null;
        while ((clientMatch = clientRegex.exec(logs)) !== null) {
            lastClient = { id: Number(clientMatch[1]), idx: clientMatch.index };
        }

        if (!lastClient) {
            labelEl.textContent = 'Client progress waiting for first client...';
            pctEl.textContent = '0%';
            fillEl.style.width = '0%';
            return;
        }

        const tail = logs.slice(lastClient.idx);
        const epochRegex = /Epoch\s+(\d+)\s*\/\s*(\d+)/g;
        let epochMatch;
        let lastEpoch = null;
        while ((epochMatch = epochRegex.exec(tail)) !== null) {
            lastEpoch = { now: Number(epochMatch[1]), total: Number(epochMatch[2]) };
        }

        const now = lastEpoch ? lastEpoch.now : 0;
        const total = lastEpoch ? lastEpoch.total : defaultEpochs;
        const pct = total > 0 ? Math.min(100, Math.round((now / total) * 100)) : 0;

        if (total > 0 && now <= 0) {
            labelEl.textContent = `Client ${lastClient.id}${clientsTotal ? ` / ${clientsTotal}` : ''} — starting epoch 1 / ${total}`;
        } else {
            labelEl.textContent = `Client ${lastClient.id}${clientsTotal ? ` / ${clientsTotal}` : ''} — Epoch ${now} / ${total || '—'}`;
        }
        pctEl.textContent = `${pct}%`;
        fillEl.style.width = `${pct}%`;
        this._state.lastClientProgressLogs = logs;
    },

    /**
     * Formats numeric metric values with a fixed precision.
     * @param {number|null|undefined} v Metric value.
     * @param {number} [digits=4] Decimal digits.
     * @returns {string} Formatted value or placeholder.
     */
    _formatMetric(v, digits = 4) {
        if (v == null || Number.isNaN(v) || !Number.isFinite(v)) return '-';
        const num = Number(v);
        const abs = Math.abs(num);
        if ((abs > 0 && abs < 0.0001) || abs >= 100000) {
            return num.toExponential(2).replace('+', '');
        }
        if (abs >= 1000) {
            const suffixes = [
                [1e12, 'T'],
                [1e9, 'B'],
                [1e6, 'M'],
                [1e3, 'K'],
            ];
            const match = suffixes.find(([value]) => abs >= value);
            if (match) {
                const scaled = num / match[0];
                const compactDigits = Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2;
                return `${scaled.toFixed(compactDigits)}${match[1]}`;
            }
        }
        return num.toFixed(digits);
    },

    /**
     * Writes a formatted metric value into a DOM element.
     * @param {string} id Element id.
     * @param {number|null|undefined} value Metric value.
     * @param {number} [digits=4] Decimal digits.
     * @returns {void}
     */
    _setMetricValue(id, value, digits = 4, fallback = '-') {
        const el = document.getElementById(id);
        if (!el) return;
        const formatted = this._formatMetric(value, digits);
        el.textContent = formatted;
        if (formatted === '-' && fallback !== '-') {
            el.textContent = fallback;
        }
        el.title = value == null || Number.isNaN(Number(value)) || !Number.isFinite(Number(value))
            ? ''
            : Number(value).toString();
    },

    /**
     * Applies monitoring panel visibility based on evaluation split mode.
     * @param {string} mode Evaluation split mode.
     * @returns {void}
     */
    _applyMonitoringModeVisibility(mode, { showTest = false } = {}) {
        const testPanel = document.getElementById('monitor-test-panel');
        const valPanel = document.getElementById('monitor-val-panel');
        if (!testPanel) return;
        const effectiveShowTest = mode !== 'train_val';
        testPanel.classList.toggle('hidden', !effectiveShowTest);
        if (valPanel) {
            valPanel.classList.toggle('xl:col-span-2', !effectiveShowTest);
        }
    },

    /**
     * Updates monitor latest-so-far metric cards from the live metrics payload.
     * @param {Record<string, *>} metrics Run metrics payload.
     * @returns {void}
     */
    _updateMonitorLatestMetrics(metrics) {
        const last = arr => (arr.length ? arr[arr.length - 1] : null);
        const toArray = (value) => (Array.isArray(value) ? value : []);

        const val = metrics?.aggregated_val || {};
        const test = metrics?.aggregated_test || {};
        const mode = this._state.currentConfig?.evaluation_split_mode || 'train_val_test';
        const valAcc = toArray(val.accuracy);
        const valLoss = toArray(val.loss);
        const valPrecision = toArray(val.precision);
        const valRecall = toArray(val.recall);
        const valF1 = toArray(val.f1);
        const valRocAuc = toArray(val.roc_auc);
        const testAcc = toArray(test.accuracy);
        const testLoss = toArray(test.loss);
        const testPrecision = toArray(test.precision);
        const testRecall = toArray(test.recall);
        const testF1 = toArray(test.f1);
        const testRocAuc = toArray(test.roc_auc);
        const showTest = mode !== 'train_val';

        this._setMetricValue('mon-metric-val-acc', last(valAcc), 3);
        this._setMetricValue('mon-metric-val-loss', last(valLoss), 4);
        this._setMetricValue('mon-metric-val-precision', last(valPrecision), 3);
        this._setMetricValue('mon-metric-val-recall', last(valRecall), 3);
        this._setMetricValue('mon-metric-val-f1', last(valF1), 3);
        this._setMetricValue('mon-metric-val-roc_auc', last(valRocAuc), 3, 'N/A');

        this._setMetricValue('mon-metric-test-acc', last(testAcc), 3);
        this._setMetricValue('mon-metric-test-loss', last(testLoss), 4);
        this._setMetricValue('mon-metric-test-precision', last(testPrecision), 3);
        this._setMetricValue('mon-metric-test-recall', last(testRecall), 3);
        this._setMetricValue('mon-metric-test-f1', last(testF1), 3);
        this._setMetricValue('mon-metric-test-roc_auc', last(testRocAuc), 3, 'N/A');

        const insightEl = document.getElementById('mon-metric-insight');
        if (insightEl) {
            this._applyMonitoringModeVisibility(mode, { showTest });
            insightEl.textContent = mode === 'train_val'
                ? 'Mode: train_val. Validation metrics are shown as latest-so-far values and test metrics are not available for this run.'
                : 'Mode: train_val_test. Validation metrics and test metrics are shown as latest-so-far values in separate splits.';
        }
    },

    /**
     * Polls metrics, updates charts, and detects terminal run state transitions.
     * @returns {Promise<void>}
     */
    async _pollMetrics() {
        const runId = this._state.currentRunId;
        if (!runId) return;
        try {
            const metrics = await api.getMetrics(runId);
            if (!metrics || !Object.keys(metrics).length) return;

            // Ensure live charts are initialized before updating
            if (!this._state.liveChartsInit) {
                metricsCharts.initLive();
                this._state.liveChartsInit = true;
            }
            this._state.lastLiveMetrics = metrics;
            metricsCharts.updateLive(metrics);
            this._updateProgressBar(metrics);
            this._updateMonitorLatestMetrics(metrics);
            this._updateClientEpochProgressFromLogs('', metrics);

            // Update monitor status
            const monStatus = document.getElementById('mon-status');
            if (monStatus) monStatus.textContent = `Run: ${runId}`;

            // Check if training finished
            const st = await api.getRunStatus(runId);
            if (!st.running && this._state.trainingStatus === 'running') {
                if (st.active || st.status === 'queued') {
                    if (monStatus) monStatus.textContent = `Run: ${runId} (${st.status})`;
                    return;
                }
                if (st.stop_reason) this._onTrainingAborted(runId, st.stop_reason);
                else this._onTrainingComplete(runId);
            }
        } catch (e) {
            this._reportNonBlockingIssue?.('run metrics polling', e, {
                statusId: 'mon-status',
                statusMessage: `Run: ${runId} (monitor sync delayed)`,
                dedupeMs: 15000,
            });
        }
    },

    /**
     * Handles aborted run state and updates monitor UI.
     * @param {string} runId Run identifier.
     * @param {string} reason Stop reason.
     * @returns {void}
     */
    _onTrainingAborted(runId, reason) {
        this._state.trainingStatus = 'idle';
        this._state.monitorStopGuardRunId = null;
        this._stopPolling();
        this._setTrainingButtons(false);
        this._state.lastLiveMetrics = null;
        this._state.lastClientProgressLogs = null;
        this._updateClientEpochProgressFromLogs('');
        document.getElementById('mon-download-row')?.classList.add('hidden');
        this._showToast(`Run ${runId} stopped: ${reason}`, 'warn');
    },

    /**
     * Updates aggregated-round progress bar from metrics and config.
     * @param {Record<string, *>} metrics Run metrics payload.
     * @returns {void}
     */
    _updateProgressBar(metrics) {
        const acc = (metrics.aggregated_val?.accuracy) || [];
        const liveState = metrics?.live_state || {};
        const doneFromState = Number(liveState.completed_rounds);
        const totalFromState = Number(liveState.total_rounds);
        const done = Number.isFinite(doneFromState) && doneFromState >= 0 ? Math.floor(doneFromState) : acc.length;
        const cfg = this._state.currentConfig;
        const total = Number.isFinite(totalFromState) && totalFromState > 0
            ? Math.floor(totalFromState)
            : (cfg ? Math.floor(cfg.local_model_epochs / cfg.weights_sending_frequency) : 0);
        const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;

        const fill  = document.getElementById('progress-fill');
        const label = document.getElementById('progress-label');
        const pctEl = document.getElementById('progress-pct');
        if (fill)  fill.style.width = `${pct}%`;
        if (label) label.textContent = total > 0 ? `Round ${done} / ${total}` : `Round ${done}`;
        if (pctEl) pctEl.textContent = `${pct}%`;

        // Repetition indicator: only meaningful for multi-repeat runs.
        const repeatRow = document.getElementById('repeat-progress-row');
        const repeatLabel = document.getElementById('repeat-progress-label');
        const totalRepeats = Math.floor(Number(liveState.total_repeats) || Number(cfg?.repeat) || 1);
        const currentRepeat = Math.floor(Number(liveState.current_repeat) || 1);
        if (repeatRow && repeatLabel) {
            if (totalRepeats > 1) {
                repeatLabel.textContent = `Repetition ${Math.min(currentRepeat, totalRepeats)} / ${totalRepeats}`;
                repeatRow.classList.remove('hidden');
            } else {
                repeatRow.classList.add('hidden');
            }
        }
    },

    /**
     * Handles successful run completion and reveals download link.
     * @param {string} runId Run identifier.
     * @returns {void}
     */
    _onTrainingComplete(runId) {
        this._state.trainingStatus = 'done';
        this._state.monitorStopGuardRunId = null;
        this._stopPolling();
        this._setTrainingButtons(false);
        this._updateClientEpochProgressFromLogs('', this._state.lastLiveMetrics || null);

        // Reveal download button
        const dlRow = document.getElementById('mon-download-row');
        const dlBtn = document.getElementById('mon-download-btn');
        if (dlRow && dlBtn) {
            api.bindExportLink(dlBtn, runId, (e) => {
                const msg = e.response?.data?.detail || e.message || 'Failed to download run archive';
                this._showToast(msg, 'error');
            });
            dlRow.classList.remove('hidden');
        }

        // Final log + metrics refresh
        this._pollLogs();
        this._pollMetrics();
    },

    // ── Defaults ─────────────────────────────────────────────────────────────

    /**
     * Persists the current setup as application defaults.
     * @returns {Promise<void>}
     */
    async _saveDefaults() {
        try {
            const config = this._collectConfig();
            const res = await api.saveDefaults(config);
            this._showToast(res.message || 'Defaults saved', 'success');
        } catch (e) {
            const msg = e.response?.data?.detail || e.message || 'Failed to save defaults';
            this._showToast(msg, 'error');
        }
    },

};
