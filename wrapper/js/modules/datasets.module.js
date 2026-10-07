window.AppDatasetsModule = {
    /**
     * Returns the current dataset search metadata.
     * @returns {{field: string, query: string}} Search state.
     */
    _datasetSearchState() {
        const rawField = String(this._state.datasetsSearchField || 'all');
        const field = (!this._state?.isAdmin && rawField === 'owner') ? 'all' : rawField;
        return {
            field,
            query: String(this._state.datasetsSearchQuery || '').trim().toLowerCase(),
        };
    },

    /**
     * Returns the current dataset sort key.
     * @returns {string} Sort token.
     */
    _datasetSortState() {
        const raw = String(this._state.datasetsSortField || 'dataset_name_asc');
        if (!this._state?.isAdmin && raw === 'owner_asc') return 'dataset_name_asc';
        return raw;
    },

    /**
     * Normalizes a dataset value for filtering and sorting.
     * @param {unknown} value Source value.
     * @returns {string} Normalized string.
     */
    _datasetTextValue(value) {
        return String(value || '').trim().toLowerCase();
    },

    /**
     * Converts a dataset created timestamp into a sortable numeric value.
     * @param {string} value Created timestamp.
     * @returns {number} Sortable number.
     */
    _datasetCreatedSortValue(value) {
        const raw = String(value || '').trim();
        if (!raw) return 0;
        const ts = Date.parse(raw);
        return Number.isFinite(ts) ? ts : 0;
    },

    /**
     * Sorts dataset rows by the active dataset sort key.
     * @param {Array<Record<string, *>>} items Dataset rows.
     * @param {string} [sortKey] Optional sort key override.
     * @returns {Array<Record<string, *>>} Sorted dataset rows.
     */
    _sortCustomDatasets(items, sortKey = this._datasetSortState()) {
        const rows = Array.isArray(items) ? [...items] : [];
        const byText = (getter, dir = 'asc') => rows.sort((left, right) => {
            const a = this._datasetTextValue(getter(left));
            const b = this._datasetTextValue(getter(right));
            if (a === b) return 0;
            const cmp = a < b ? -1 : 1;
            return dir === 'desc' ? -cmp : cmp;
        });
        const byNumber = (getter, dir = 'asc') => rows.sort((left, right) => {
            const a = Number(getter(left));
            const b = Number(getter(right));
            const aFinite = Number.isFinite(a) ? a : -Infinity;
            const bFinite = Number.isFinite(b) ? b : -Infinity;
            if (aFinite === bFinite) return 0;
            const cmp = aFinite < bFinite ? -1 : 1;
            return dir === 'desc' ? -cmp : cmp;
        });
        const byDate = (getter, dir = 'asc') => rows.sort((left, right) => {
            const a = this._datasetCreatedSortValue(getter(left));
            const b = this._datasetCreatedSortValue(getter(right));
            if (a === b) return 0;
            const cmp = a < b ? -1 : 1;
            return dir === 'desc' ? -cmp : cmp;
        });

        switch (sortKey) {
            case 'dataset_name_desc': return byText((d) => d.dataset_name, 'desc');
            case 'created_at_desc': return byDate((d) => d.created_at, 'desc');
            case 'created_at_asc': return byDate((d) => d.created_at, 'asc');
            case 'owner_asc': return byText((d) => d.owner, 'asc');
            case 'dataset_format_asc': return byText((d) => d.dataset_format || d.dataset_type, 'asc');
            case 'rows_desc': return byNumber((d) => d.rows, 'desc');
            case 'dataset_name_asc':
            default:
                return byText((d) => d.dataset_name, 'asc');
        }
    },

    /**
     * Builds the filtered and sorted custom dataset list.
     * @returns {Array<Record<string, *>>} Dataset rows.
     */
    _filteredSortedCustomDatasets() {
        const items = Array.isArray(this._state.datasetsCustomCache) ? [...this._state.datasetsCustomCache] : [];
        const { field, query } = this._datasetSearchState();
        const sortKey = this._datasetSortState();

        const filtered = !query ? items : items.filter((d) => {
            const fields = {
                all: [d.dataset_name, d.description, d.owner, d.dataset_format, d.dataset_type, d.created_at],
                dataset_name: [d.dataset_name],
                owner: [d.owner],
                description: [d.description],
                dataset_format: [d.dataset_format, d.dataset_type],
                created_at: [d.created_at],
            };
            const haystack = (fields[field] || fields.all)
                .map((value) => this._datasetTextValue(value))
                .join(' ');
            return haystack.includes(query);
        });
        return this._sortCustomDatasets(filtered, sortKey);
    },

    /**
     * Refreshes Datasets tab content (defaults + custom datasets list).
     * @returns {Promise<void>}
     */
    async _refreshDatasetsManager() {
        const customBody = document.getElementById('datasets-custom-list');
        const statusEl = document.getElementById('datasets-status');
        if (!customBody) return;

        const searchFieldEl = document.getElementById('datasets-search-field');
        const searchInputEl = document.getElementById('datasets-search-input');
        const sortFieldEl = document.getElementById('datasets-sort-field');
        if (searchFieldEl) searchFieldEl.value = this._datasetSearchState().field;
        if (searchInputEl) searchInputEl.value = this._state.datasetsSearchQuery || '';
        if (sortFieldEl) sortFieldEl.value = this._datasetSortState();

        if (statusEl) {
            statusEl.textContent = 'Loading datasets...';
            statusEl.classList.remove('hidden');
        }

        const renderEmpty = (text) => {
            customBody.innerHTML = `<div class="ui-empty-state">${this._escapeHtml(text)}</div>`;
        };

        try {
            const datasets = await api.listCustomDatasets();
            this._state.datasetsCustomCache = Array.isArray(datasets) ? datasets : [];
            this._renderCustomDatasetsRows();
            if (statusEl) {
                statusEl.textContent = `Custom datasets loaded (${this._state.datasetsCustomCache.length})`;
            }
        } catch (e) {
            renderEmpty('Failed to load custom datasets');
            if (statusEl) {
                statusEl.textContent = e.response?.data?.detail || e.message || 'Failed to load datasets';
            }
        }
    },

    /**
     * Renders built-in default datasets list.
     * @returns {void}
     */
    _renderDefaultDatasetsRows() {
        const body = document.getElementById('datasets-default-list');
        if (!body) return;
        const defaults = (UI_DATA.datasetOptions || [])
            .filter(([v]) => v !== 'custom_csv' && v !== 'custom_image_npz')
            .map(([v, label]) => ({ key: v, label }));

        body.innerHTML = defaults.map((d) => `
            <div class="bg-slate-800 border border-slate-700/60 rounded-lg p-3">
                <div class="flex items-center justify-between gap-2">
                    <div>
                        <div class="text-sm font-semibold text-slate-100">${this._escapeHtml(d.label)}</div>
                        <div class="text-[11px] text-slate-400">Key: ${this._escapeHtml(d.key)}</div>
                    </div>
                    <span class="ui-badge-pill ui-badge-info">Default</span>
                </div>
            </div>
        `).join('');
    },

    /**
     * Renders custom datasets rows with edit/delete controls.
     * @returns {void}
     */
    _renderCustomDatasetsRows() {
        const body = document.getElementById('datasets-custom-list');
        if (!body) return;

        const isAdmin = !!(this._state?.isAdmin);
        const currentUsername = String(this._state?.username || '').toLowerCase();
        const items = this._filteredSortedCustomDatasets();

        if (!items.length) {
            body.innerHTML = '<div class="ui-empty-state">No custom datasets found.</div>';
            return;
        }

        body.innerHTML = items.map((d) => {
            const ref = String(d.dataset_ref || '');
            const rows = Number.isFinite(Number(d.rows)) ? Number(d.rows) : null;
            const feats = Number.isFinite(Number(d.features)) ? Number(d.features) : null;
            const cls = Number.isFinite(Number(d.classes)) ? Number(d.classes) : null;
            const created = String(d.created_at || '-');
            const desc = String(d.description || '');
            const name = String(d.dataset_name || ref || 'custom_dataset');
            const owner = String(d.owner || 'unknown');
            const isOwner = owner.toLowerCase() === currentUsername;

            const rowsTxt = rows != null ? rows : '-';
            const featsTxt = feats != null ? feats : '-';
            const clsTxt = cls != null ? cls : '-';
            const formatTxt = this._customDatasetFormatLabel(d.dataset_format, d.dataset_type);

            const ownerBadge = (isAdmin && !isOwner) 
                ? `<span class="ml-2 ui-badge-pill ui-badge-warn uppercase tracking-tighter">Owner: ${this._escapeHtml(owner)}</span>`
                : '';

            return `
                <div class="bg-slate-800 border border-slate-700/60 rounded-xl p-4 transition-all hover:border-slate-600/60" data-dataset-ref="${this._escapeHtml(ref)}">
                    <div class="grid grid-cols-1 xl:grid-cols-12 gap-4">
                        <div class="xl:col-span-4">
                            <label class="block text-[10px] uppercase tracking-wider font-bold text-slate-500 mb-1.5 flex items-center justify-between">
                                <span>Dataset Name</span>
                                ${ownerBadge}
                            </label>
                            <input type="text" maxlength="20" class="fl-input text-xs font-medium" data-ds-name value="${this._escapeHtml(name)}" placeholder="Max 20 chars">
                        </div>
                        <div class="xl:col-span-5">
                            <label class="block text-[10px] uppercase tracking-wider font-bold text-slate-500 mb-1.5">Description</label>
                            <textarea class="fl-input text-xs leading-relaxed" rows="3" maxlength="200" data-ds-description style="resize:none; min-height:84px; max-height:84px; overflow:auto;" placeholder="Add a short dataset description (max 200 chars)...">${this._escapeHtml(desc)}</textarea>
                        </div>
                        <div class="xl:col-span-3">
                            <label class="block text-[10px] uppercase tracking-wider font-bold text-slate-500 mb-1.5">Properties</label>
                            <div class="bg-slate-900/40 rounded-lg p-2.5 border border-slate-700/30">
                                <div class="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
                                    <div class="flex justify-between items-center"><span class="text-slate-500">Format:</span> <span class="text-slate-200 font-mono">${this._escapeHtml(formatTxt)}</span></div>
                                    <div class="flex justify-between items-center"><span class="text-slate-500">Rows:</span> <span class="text-slate-200 font-mono">${rowsTxt}</span></div>
                                    <div class="flex justify-between items-center"><span class="text-slate-500">Classes:</span> <span class="text-slate-200 font-mono">${clsTxt}</span></div>
                                    <div class="flex justify-between items-center"><span class="text-slate-500">Features:</span> <span class="text-slate-200 font-mono">${featsTxt}</span></div>
                                </div>
                                <div class="mt-2 pt-2 border-t border-slate-700/30 flex items-center justify-between gap-3 text-[10px]" title="${this._escapeHtml(created)}">
                                    <span class="text-slate-500 uppercase tracking-wider font-bold">Created</span>
                                    <span class="text-slate-300 font-mono whitespace-nowrap">${this._escapeHtml(created.includes('T') ? created.split('T')[0] : created.split(' ')[0])}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="flex items-center justify-end gap-3 mt-4 pt-3 border-t border-slate-700/40">
                        <button class="ui-btn-compact ui-btn-danger" data-ds-action="delete">
                            <i class="ph ph-trash mr-1.5"></i>Delete
                        </button>
                        <button class="ui-btn-compact ui-btn-success admin-promote-btn" data-ds-action="save">
                            <i class="ph ph-floppy-disk mr-1.5"></i>Save Changes
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    },

    /**
     * Updates the dataset tab search field state.
     * @param {string} field Search field key.
     * @returns {void}
     */
    _setDatasetSearchField(field) {
        const next = String(field || 'all');
        this._state.datasetsSearchField = (!this._state?.isAdmin && next === 'owner') ? 'all' : next;
        // Reflect any coerced value back onto the select; the dropdown follows it.
        const fieldEl = document.getElementById('datasets-search-field');
        if (fieldEl) fieldEl.value = this._state.datasetsSearchField;
        this._renderCustomDatasetsRows();
    },

    /**
     * Updates the dataset tab search query state.
     * @param {string} query Search query.
     * @returns {void}
     */
    _setDatasetSearchQuery(query) {
        this._state.datasetsSearchQuery = String(query || '');
        this._renderCustomDatasetsRows();
    },

    /**
     * Updates the dataset tab sorting state.
     * @param {string} sortField Sort key.
     * @returns {void}
     */
    _setDatasetSortField(sortField) {
        const next = String(sortField || 'dataset_name_asc');
        this._state.datasetsSortField = (!this._state?.isAdmin && next === 'owner_asc') ? 'dataset_name_asc' : next;
        // Reflect any coerced value back onto the select; the dropdown follows it.
        const sortEl = document.getElementById('datasets-sort-field');
        if (sortEl) sortEl.value = this._state.datasetsSortField;
        this._renderCustomDatasetsRows();
    },

    /**
     * Handles click actions in custom datasets manager list.
     * @param {MouseEvent} event Click event.
     * @returns {Promise<void>}
     */
    async _onDatasetsManagerClick(event) {
        const btn = event.target.closest('[data-ds-action]');
        if (!btn) return;
        const card = btn.closest('[data-dataset-ref]');
        const datasetRef = String(card?.getAttribute('data-dataset-ref') || '').trim();
        if (!datasetRef) return;

        const action = btn.getAttribute('data-ds-action');
        if (action === 'delete') {
            const ok = await this._confirmDialog(`Delete custom dataset ${datasetRef}?`);
            if (!ok) return;
            try {
                await api.deleteCustomDataset(datasetRef);
                this._showToast('Dataset deleted', 'success');
                if (this._state.customDatasetRef === datasetRef) {
                    this._state.customDatasetRef = '';
                    this._state.customDatasetFileName = '';
                }
                await this._refreshDatasetsManager();
                await this._toggleCustomDatasetInputs?.();
            } catch (e) {
                this._showToast(e.response?.data?.detail || 'Failed to delete dataset', 'error');
            }
            return;
        }

        if (action === 'save') {
            const name = String(card?.querySelector('[data-ds-name]')?.value || '').trim();
            const description = String(card?.querySelector('[data-ds-description]')?.value || '').trim();

            if (!name) {
                this._showToast('Dataset name is required', 'warn');
                return;
            }
            if (name.length > 20) {
                this._showToast('Dataset name too long (max 20)', 'error');
                return;
            }
            if (!/^[A-Za-z0-9_.\- ]+$/.test(name)) {
                this._showToast('Dataset name has invalid characters', 'error');
                return;
            }
            if (description.length > 200) {
                this._showToast('Description too long (max 200)', 'error');
                return;
            }

            const original = (this._state.datasetsCustomCache || []).find((d) => String(d.dataset_ref) === datasetRef) || {};
            const originalName = String(original.dataset_name || '').trim();
            const originalDescription = String(original.description || '').trim();

            try {
                if (name !== originalName) {
                    await api.renameCustomDataset(datasetRef, name);
                }
                if (description !== originalDescription) {
                    await api.updateCustomDatasetDescription(datasetRef, description);
                }
                this._showToast('Dataset metadata saved', 'success');
                if (this._state.customDatasetRef === datasetRef) {
                    this._state.customDatasetFileName = name;
                }
                await this._refreshDatasetsManager();
                await this._toggleCustomDatasetInputs?.();
            } catch (e) {
                this._showToast(e.response?.data?.detail || 'Failed to save dataset metadata', 'error');
            }
        }
    },

    /**
     * Uploads a new dataset from datasets tab.
     * @returns {Promise<void>}
     */
    async _uploadDatasetFromDatasetsTab() {
        const input = document.getElementById('datasets-upload-file-input');
        const file = input?.files?.[0];
        const description = String(document.getElementById('datasets-upload-description')?.value || '').trim();
        if (!file) {
            this._showToast('Choose a dataset file first', 'warn');
            return;
        }
        await this._uploadCustomDatasetFromFile(file, { description });
        if (input) input.value = '';
        const nameEl = document.getElementById('datasets-upload-file-name');
        if (nameEl) nameEl.textContent = 'No dataset selected';
        const descEl = document.getElementById('datasets-upload-description');
        if (descEl) descEl.value = '';
        await this._refreshDatasetsManager();
    },
};
