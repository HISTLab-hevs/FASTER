window.AppSetupCustomDatasetModule = {
    /**
     * Clears the current custom dataset state and resets the visible dataset summary.
     * @returns {void}
     */
    _resetCustomDatasetSelectionUi() {
        const nameEl = document.getElementById('custom-dataset-file-name');
        const metaEl = document.getElementById('custom-dataset-meta');
        const statusEl = document.getElementById('custom-dataset-status');

        this._state.customDatasetRef = '';
        this._state.customDatasetFileName = '';
        this._state.customDatasetFormat = 'csv';
        this._state.customDatasetType = 'csv';

        if (nameEl) {
            nameEl.textContent = 'No file';
            nameEl.removeAttribute('title');
        }
        if (metaEl) metaEl.textContent = '';
        if (statusEl) {
            statusEl.textContent = '';
            statusEl.classList.add('hidden');
        }
        this._updateCustomDatasetSimpleModeWarning?.();
    },

    /**
     * Builds the select-option HTML for available custom datasets.
     * @param {Array<Object>} datasets Available custom datasets.
     * @returns {string} Option markup.
     */
    _renderCustomDatasetOptions(datasets) {
        const items = this._sortCustomDatasets ? this._sortCustomDatasets(datasets, 'dataset_name_asc') : (Array.isArray(datasets) ? datasets : []);
        if (!items.length) {
            return '<option value="">- no datasets found, please upload -</option>';
        }

        const isAdmin = !!(this._state?.isAdmin);
        const currentUsername = String(this._state?.username || '').toLowerCase();

        return '<option value="">- upload new or select -</option>' + items.map((d) => {
            const owner = String(d.owner || 'unknown');
            const isOwner = owner.toLowerCase() === currentUsername;
            const ownerPrefix = (isAdmin && !isOwner) ? `[${owner}] ` : '';

            const rawName = String(d.dataset_name || d.dataset_ref || 'custom_dataset');
            const formatLabel = this._customDatasetFormatLabel(d.dataset_format, d.dataset_type);
            const fullName = `${ownerPrefix}${rawName}`;
            const rows = Number.isFinite(Number(d.rows)) ? Number(d.rows) : null;
            const meta = rows != null ? ` (${formatLabel}, ${rows} rows)` : ` (${formatLabel})`;
            const text = `${fullName}${meta}`;
            const visible = this._truncateLabel(text, 64);
            return `<option value="${this._escapeHtml(d.dataset_ref)}" title="${this._escapeHtml(text)}">${this._escapeHtml(visible)}</option>`;
        }).join('');
    },

    /**
     * Applies the selected custom dataset to local state and summary UI.
     * @param {string} datasetRef Selected dataset reference.
     * @param {Array<Object>} datasets Available custom datasets.
     * @param {HTMLSelectElement|null} [selectEl] Source select element.
     * @returns {void}
     */
    _applyCustomDatasetSelection(datasetRef, datasets, selectEl = null) {
        if (!datasetRef) {
            this._resetCustomDatasetSelectionUi();
            return;
        }

        const dsObj = (Array.isArray(datasets) ? datasets : []).find((d) => d.dataset_ref === datasetRef);
        const selectedOpt = selectEl?.options?.[selectEl.selectedIndex] || null;

        this._state.customDatasetRef = datasetRef;
        if (dsObj) {
            this._state.customDatasetFormat = this._normalizeCustomDatasetFormat(dsObj.dataset_format, dsObj.dataset_type);
            this._state.customDatasetType = this._normalizeCustomDatasetType(dsObj.dataset_format, dsObj.dataset_type);
        }

        const rawName = String(dsObj?.dataset_name || '').trim();
        const optText = rawName || selectedOpt?.getAttribute('title') || selectedOpt?.text || datasetRef;
        const cleanName = optText
            .replace(/^\[[^\]]+\]\s*/, '')
            .split(' (')[0]
            .replace(/\.(csv|tsv|npz|zip)$/i, '');

        this._state.customDatasetFileName = cleanName;

        const nameEl = document.getElementById('custom-dataset-file-name');
        if (nameEl) {
            const visible = this._truncateLabel(cleanName, 48);
            nameEl.textContent = visible || 'Custom dataset';
            nameEl.title = cleanName || datasetRef;
        }

        const metaEl = document.getElementById('custom-dataset-meta');
        if (metaEl) {
            const rows = Number.isFinite(Number(dsObj?.rows)) ? Number(dsObj.rows) : null;
            const features = Number.isFinite(Number(dsObj?.features)) ? Number(dsObj.features) : null;
            const classes = Number.isFinite(Number(dsObj?.classes)) ? Number(dsObj.classes) : null;
            if (rows != null || features != null || classes != null) {
                const formatLabel = this._customDatasetFormatLabel(dsObj?.dataset_format, dsObj?.dataset_type);
                metaEl.textContent = `${formatLabel} | ${rows ?? 0} samples, ${features ?? 0} features, ${classes ?? 0} classes`;
            } else {
                metaEl.textContent = '';
            }
        }

        const statusEl = document.getElementById('custom-dataset-status');
        if (statusEl) {
            statusEl.textContent = '';
            statusEl.classList.add('hidden');
        }
        this._updateCustomDatasetSimpleModeWarning?.();
    },

    /**
     * Loads custom dataset options into the setup select and restores the current selection if possible.
     * @returns {Promise<void>}
     */
    async _loadCustomDatasetOptions() {
        const selectWrap = document.getElementById('custom-dataset-select-wrap');
        const selectEl = document.getElementById('cfg-custom_dataset_ref');
        if (!selectWrap || !selectEl) return;

        selectWrap.classList.remove('hidden');
        let datasets = [];

        try {
            datasets = await api.listCustomDatasets();
            this._state.datasetsCustomCache = Array.isArray(datasets) ? datasets : [];
            selectEl.innerHTML = this._renderCustomDatasetOptions(datasets);

            if (this._state.customDatasetRef && Array.from(selectEl.options).some((o) => o.value === this._state.customDatasetRef)) {
                selectEl.value = this._state.customDatasetRef;
                this._applyCustomDatasetSelection(this._state.customDatasetRef, datasets, selectEl);
            }
        } catch (e) {
            selectEl.innerHTML = '<option value="">- failed to load datasets -</option>';
            this._reportNonBlockingIssue?.('custom dataset options load', e, {
                dedupeMs: 15000,
            });
        }

        selectEl.onchange = (e) => {
            this._applyCustomDatasetSelection(e.target.value, datasets, selectEl);
        };

        this._updateCustomDatasetSimpleModeWarning?.();
    },
};
