window.AppRenderModule = {
    // Page rendering
    /**
     * Resolves the public route from current hash.
     * @returns {string} Normalized route key.
     */
    _publicRouteFromPath() {
        const path = window.location.pathname;
        const parts = path.split('/').filter(Boolean);
        // Expecting /Faster/route
        if (parts.length >= 2 && parts[0].toLowerCase() === 'faster') {
            const rawRoute = parts[1];
            const routeMap = {
                docs: 'docs', Documentation: 'docs',
                creators: 'creators', Creators: 'creators',
                support: 'support', Contribute: 'support',
                'Signin': 'signin', signin: 'signin', landingpage: 'landingpage'
            };
            return routeMap[rawRoute] || routeMap[rawRoute.toLowerCase()] || 'signin';
        }
        return 'signin';
    },

    /**
     * Resolves in-page chapter id from hash route.
     * @returns {string} Chapter anchor id.
     */
    _chapterFromPath() {
        const path = window.location.pathname;
        const parts = path.split('/').filter(Boolean);
        // Expecting /Faster/route/chapter
        if (parts.length >= 3 && parts[0].toLowerCase() === 'faster') {
            return parts.slice(2).join('/');
        }
        return '';
    },

    /**
     * Smooth-scrolls to a documentation chapter element.
     * @param {string} chapterId Target element id.
     * @returns {void}
     */
    _scrollToChapter(chapterId) {
        if (!chapterId) return;
        const target = document.getElementById(chapterId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    },

    /**
     * Handles hash changes for public pages when user is not authenticated.
     * @returns {Promise<void>}
     */
    async _onPopState() {
        if (api.isAuthenticated()) {
            const path = window.location.pathname;
            const parts = path.split('/').filter(Boolean);
            if (parts.length >= 2 && parts[0].toLowerCase() === 'faster') {
                const slug = parts[1];
                const allTabs = [...(UI_DATA.navTabs || []), ...(UI_DATA.adminNavTabs || [])];
                const tabData = allTabs.find(t => t[3] === slug);
                if (tabData) {
                    this._switchTab(tabData[2]);
                    return;
                }
                if (slug === 'signin') {
                    this._showDashboard();
                    return;
                }
                if (['Documentation', 'Creators', 'Contribute'].includes(slug)) {
                    this._openAuthResource(slug);
                    return;
                }
            }
        }

        const route = this._publicRouteFromPath();
        await this._showPublicPage(route);
    },

    /**
     * Navigates to landing/sign-in view.
     * @returns {Promise<void>}
     */
    async _showLanding() {
        history.pushState(null, '', '/Faster/signin');
        await this._showPublicPage('signin');
    },

    /**
     * Renders a public page route and binds landing events.
     * @param {string} route Public route key.
     * @returns {Promise<void>}
     */
    async _showPublicPage(route) {
        const chapter = this._chapterFromPath();
        const publicTemplateByRoute = {
            docs: 'docs.html',
            creators: 'creators.html',
            support: 'support.html',
        };

        const titleMap = {
            docs: 'Documentation',
            creators: 'Creators',
            support: 'Contribute',
            signin: 'Sign In'
        };
        if (titleMap[route]) {
            document.title = `FASTER — ${titleMap[route]}`;
        }

        const slugMap = { docs: 'Documentation', creators: 'Creators', support: 'Contribute', signin: 'Signin' };
        if (slugMap[route]) {
            const preferredPath = `/Faster/${slugMap[route]}`;
            const currentPath = window.location.pathname;
            if (currentPath === `/Faster/${route}` || currentPath.toLowerCase() === `/faster/${route}`) {
                const chapterPart = chapter ? `/${chapter}` : '';
                history.replaceState(null, '', preferredPath + chapterPart);
            }
        }

        if (publicTemplateByRoute[route]) {
            document.getElementById('app').innerHTML = await this._loadTemplate(publicTemplateByRoute[route]);
            this._attachLandingEvents();
            requestAnimationFrame(() => this._scrollToChapter(chapter));
            return;
        }

        document.getElementById('app').innerHTML = await this._renderLanding();
        this._attachLandingEvents();
        // Keep sign-in landing pinned at top; login is opened by hash state, not by scrolling.
        setTimeout(() => window.scrollTo({ top: 0, behavior: 'auto' }), 0);
    },

    /**
     * Renders authenticated dashboard shell and attaches dashboard events.
     * @returns {Promise<void>}
     */
    async _showDashboard() {
        document.getElementById('app').innerHTML = await this._renderDashboard();
        this._attachDashboardEvents();
    },

    /**
     * Opens documentation/creators/support content in authenticated side panel.
     * @param {string} route Resource route key.
     * @returns {Promise<void>}
     */
    async _openAuthResource(route) {
        const map = {
            docs: 'docs.html', Documentation: 'docs.html',
            creators: 'creators.html', Creators: 'creators.html',
            support: 'support.html', Contribute: 'support.html'
        };
        const tpl = map[route] || 'docs.html';
        const html = await this._loadTemplate(tpl);
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const pageSection = doc.querySelector('.landing-shell > section') || doc.querySelector('.doc-layout')?.closest('section');
        const container = document.getElementById('auth-resource-content');
        const panel = document.getElementById('auth-resource-panel');
        if (!container) return;

        document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
        document.querySelectorAll('.nav-tab').forEach(btn => btn.classList.remove('active'));
        if (panel) panel.classList.remove('hidden');
        container.innerHTML = pageSection ? pageSection.outerHTML : '<p>Resource page unavailable.</p>';

        container.querySelectorAll('a[href^="#/"], a[href^="/Faster/"]').forEach(a => {
            a.addEventListener('click', (e) => {
                const href = a.getAttribute('href') || '';
                let path = '';
                if (href.startsWith('#/')) {
                    path = href.replace(/^#\//, '');
                } else if (href.startsWith('/Faster/')) {
                    path = href.replace(/^\/Faster\//, '');
                }

                const [targetRoute, ...rest] = path.split('/');
                if (!targetRoute) return;

                if (targetRoute === 'signin') {
                    e.preventDefault();
                    this._switchTab?.(0);
                    return;
                }

                const targetNormalized = targetRoute;
                if (!['docs', 'creators', 'support', 'Documentation', 'Creators', 'Contribute'].includes(targetNormalized)) return;
                e.preventDefault();
                const routeToOpen = targetNormalized.length > 10 ? targetNormalized : {
                    docs: 'Documentation', creators: 'Creators', support: 'Contribute'
                }[targetNormalized] || targetNormalized;
                this._openAuthResource(routeToOpen);
                if (rest.length) {
                    const chapter = rest.join('/');
                    setTimeout(() => {
                        const target = document.getElementById(chapter);
                        target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 0);
                }
            });
        });

        const slugMap = {
            docs: 'Documentation', creators: 'Creators', support: 'Contribute',
            Documentation: 'Documentation', Creators: 'Creators', Contribute: 'Contribute'
        };
        const displaySlug = slugMap[route] || route;
        const chapter = this._chapterFromPath();
        const newPath = `/Faster/${displaySlug}${chapter ? '/' + chapter : ''}`;

        if (window.location.pathname !== newPath) {
            history.pushState({ tab: -1, resource: displaySlug }, '', newPath);
        }

        const titleMap = {
            docs: 'Documentation', Documentation: 'Documentation',
            creators: 'Creators', Creators: 'Creators',
            support: 'Contribute', Contribute: 'Contribute'
        };
        document.title = `FASTER — ${titleMap[route] || 'Resource'}`;

        this._state.tab = -1;

        // Scroll to chapter if present in URL (same behavior as _showPublicPage)
        requestAnimationFrame(() => this._scrollToChapter(chapter));
    },

    // Template rendering
    /**
     * Loads and caches an HTML template from wrapper templates.
     * @param {string} name Template filename.
     * @returns {Promise<string>} Raw template HTML.
     */
    async _loadTemplate(name) {
        if (this._templateCache[name]) return this._templateCache[name];

        const res = await fetch(`/wrapper/templates/${name}?v=${TEMPLATE_VERSION}`, {
            cache: 'no-store',
        });
        if (!res.ok) throw new Error(`Template load failed: ${name}`);

        const html = await res.text();
        this._templateCache[name] = html;
        return html;
    },

    /**
     * Applies placeholder replacements to a template string.
     * @param {string} template Source template.
     * @param {Record<string, *>} replacements Replacement map.
     * @returns {string} Interpolated HTML string.
     */
    _applyTemplate(template, replacements) {
        return Object.entries(replacements).reduce((html, [key, value]) => {
            return html.replaceAll(`{{${key}}}`, value == null ? '' : String(value));
        }, template);
    },

    /**
     * Escapes HTML-sensitive characters in text content.
     * @param {*} value Raw value.
     * @returns {string} Escaped string.
     */
    _escapeHtml(value) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;',
        };
        return String(value ?? '').replace(/[&<>"']/g, c => map[c]);
    },

    /**
     * Produces a label markup with tooltip help when description exists.
     * @param {string} key Parameter key.
     * @param {string} label Label text.
     * @returns {string} HTML label fragment.
     */
    _labelWithHelp(key, label) {
        const desc = UI_DATA.paramDescriptions[key];
        const safeLabel = this._escapeHtml(label);
        if (!desc) return safeLabel;
        const safeDesc = this._escapeHtml(desc);
        return `<span class="label-with-help">${safeLabel}<span class="help-dot" tabindex="0" role="img" aria-label="${safeDesc}" data-tooltip="${safeDesc}">?</span></span>`;
    },

    /**
     * Renders landing template.
     * @returns {Promise<string>} Landing HTML.
     */
    async _renderLanding() {
        return this._loadTemplate('landing.html');
    },

    /**
     * Renders dashboard template with computed dynamic sections.
     * @returns {Promise<string>} Dashboard HTML.
     */
    async _renderDashboard() {
        const template = await this._loadTemplate('dashboard.html');

        return this._applyTemplate(template, {
            USERNAME: this._escapeHtml(this._state.username),
            NAV_TABS: this._renderNavTabs(),
            ADMIN_MOBILE_TABS: this._renderAdminMobileTabs(),
            METHOD_DATASET_FIELDS: this._renderMethodAndDatasetFields(),
            DATA_DISTRIBUTION_FIELDS: this._renderDataDistributionFields(),
            TRAINING_PARAMETER_FIELDS: this._renderTrainingParameterFields(),
            GENETIC_PROGRAMMING_FIELDS: this._renderGeneticProgrammingFields(),
            HISTORY_CHART_CARDS: this._renderHistoryChartCards(),
            AVAILABLE_PRIMITIVES_LABEL: this._labelWithHelp('available_primitives', 'Available Primitives (csv)'),
        });
    },

    /**
     * Builds sidebar navigation tab markup.
     * @returns {string} HTML tabs fragment.
     */
    _renderNavTabs() {
        const tabs = [...(UI_DATA.navTabs || [])];
        if (this._state.isAdmin) tabs.push(...(UI_DATA.adminNavTabs || []));
        return tabs.map(([icon, label, idx]) => `
            <button class="nav-tab w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 ${idx === 0 ? 'active' : ''}"
                data-tab="${idx}">
                <i class="ph ${icon} text-lg"></i>
                <span class="nav-label">${label}</span>
            </button>
        `).join('');
    },

    _renderAdminMobileTabs() {
        if (!this._state.isAdmin) return '';
        return `
            <button class="mobile-tab-item w-full text-left px-3 py-2 text-xs rounded hover:bg-blue-50" data-tab="5">Platform Analytics</button>
            <button class="mobile-tab-item w-full text-left px-3 py-2 text-xs rounded hover:bg-blue-50" data-tab="6">User Management</button>
        `;
    },

    /**
     * Builds method and dataset form fields.
     * @returns {string} HTML fields fragment.
     */
    _renderMethodAndDatasetFields() {
        return `
            <div class="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-4">
                <!-- Methods -->
                <div class="bg-slate-50/30 border border-slate-200/50 rounded-xl p-4">
                    <label class="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2.5">Method Selection</label>
                    ${this._field('method', 'Algorithm(s)', 'multiselect', UI_DATA.methodOptions)}
                </div>

                <!-- Dataset -->
                <div class="bg-slate-50/30 border border-slate-200/50 rounded-xl p-4">
                    <label class="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2.5">Dataset & Source</label>
                    <div class="space-y-3">
                        <select id="cfg-dataset_name" class="fl-input w-full">
                            ${(UI_DATA.datasetOptions || []).map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}
                        </select>
                        <div id="custom-dataset-select-wrap" class="hidden">
                            <select id="cfg-custom_dataset_ref" class="fl-input w-full text-xs" title="Select existing dataset">
                                <option value="">- select custom dataset -</option>
                            </select>
                        </div>
                        <div id="custom-dataset-wrap" class="hidden border border-slate-200/60 rounded-lg p-2.5 bg-white/40 shadow-sm transition-all">
                             <div id="custom-dataset-status" class="hidden text-[10px] text-emerald-700 bg-emerald-50 border border-emerald-100 rounded px-2 py-1 mb-2"></div>
                             <div class="flex items-center justify-between gap-2 mb-2">
                                <span id="custom-dataset-file-name" class="text-[10px] text-slate-500 truncate italic font-medium">No file selected</span>
                                <button id="custom-dataset-file-btn" type="button" class="px-2.5 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold rounded-lg flex items-center gap-1.5 shrink-0 transition-colors">
                                    <i class="ph ph-upload-simple"></i> Upload
                                </button>
                             </div>
                             <div id="custom-dataset-simple-warning" class="hidden mb-2 rounded border border-amber-200 bg-amber-50 px-2.5 py-2 text-[10px] leading-snug text-amber-900"></div>
                             <div class="flex items-center justify-between gap-2 border-t border-slate-200/40 pt-2 mt-1">
                                <button id="custom-dataset-docs-link" type="button" class="text-[9px] text-blue-600 hover:text-blue-700 font-semibold uppercase tracking-tighter">Format Guide</button>
                                <div id="custom-dataset-meta" class="text-[9px] font-mono text-slate-400"></div>
                             </div>
                             <input id="custom-dataset-file-input" type="file" accept=".csv,.tsv,.npz,.zip,text/csv,text/tab-separated-values,application/octet-stream,application/zip" class="hidden">
                        </div>
                    </div>
                </div>

                <!-- Base Config -->
                <div class="bg-slate-50/30 border border-slate-200/50 rounded-xl p-4">
                    <label class="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2.5">Core Parameters</label>
                    <div class="space-y-4">
                        ${this._field('evaluation_split_mode', 'Evaluation Split', 'select', UI_DATA.evaluationSplitModeOptions)}
                        <div class="grid grid-cols-2 gap-3">
                            ${this._field('num_clients', 'Clients', 'number')}
                            ${this._field('repeat', 'Repeats', 'number', null, '1', { advanced: true })}
                        </div>
                        <div class="grid grid-cols-2 gap-3">
                            ${this._field('initial_eligible_clients', 'Initial Eligible', 'number', null, null, { advanced: true, min: 1 })}
                            ${this._field('seed', 'Seed', 'number', null, '1', { advanced: true, min: 0 })}
                        </div>
                        <div id="client-pool-warning" class="setup-field setup-field-wide hidden rounded-lg border px-2.5 py-1.5 text-[11px] leading-tight" data-advanced-field="true"></div>
                    </div>
                </div>
            </div>

            <div id="simple-gp-mode-note" class="hidden mt-3 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-[11px] leading-tight text-blue-900">
                Simple mode uses the default FedGP preset and hides GP tuning. Switch to Advanced mode to customize search parameters.
            </div>

            <!-- Run Names (Compact) -->
            <div class="bg-white/40 border border-slate-200/40 rounded-xl p-4 mb-2">
                <div class="flex items-center justify-between mb-3">
                    <label class="block text-[10px] font-bold text-slate-500 uppercase tracking-wider">Run Names Assignment</label>
                    <span id="run-name-inputs-hint" class="text-[10px] text-slate-400 italic">Assign unique labels per selected method</span>
                </div>
                <div id="run-name-inputs" class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3"></div>
                <input id="cfg-run_names" type="text" class="hidden" aria-hidden="true" tabindex="-1">
                <p id="run-name-inputs-error" class="hidden text-[10px] text-red-500 mt-2 font-semibold"></p>
            </div>
        `;
    },

    /**
     * Builds data distribution fields.
     * @returns {string} HTML fields fragment.
     */
    _renderDataDistributionFields() {
        return `
            <div class="setup-field setup-field-wide setup-distribution-panel" data-advanced-field="true">
                <div class="setup-distribution-section">
                    <div class="setup-distribution-section-header">
                        <div>
                            <p class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Global Settings</p>
                            <p class="text-[11px] text-slate-500">These values shape the data split before the round plans are applied.</p>
                        </div>
                    </div>
                    <div class="setup-distribution-top-grid">
                        ${this._field('server_data_percentage', 'Server Data %', 'number', null, '0.0001')}
                        ${this._field('imbalance_rate', 'Imbalance Rate', 'number', null, '0.0001')}
                        ${this._field('train_val_split', 'Train/Val Split', 'number', null, '0.0001')}
                        <div class="setup-field setup-field-checkbox setup-distribution-checkbox">
                            <label class="block text-xs text-slate-400 mb-1">${this._labelWithHelp('iid', 'IID Distribution')}</label>
                            <div class="setup-checkbox-tile">
                                <input type="checkbox" id="cfg-iid" class="setup-checkbox-input">
                                <span class="text-[11px] leading-tight text-slate-500">Enable IID splitting</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="setup-distribution-section">
                    <div class="setup-distribution-section-header">
                        <div>
                            <p class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Round Plans</p>
                            <p class="text-[11px] text-slate-500">Set the active mode and keep the two plans visually balanced for quick scanning.</p>
                        </div>
                    </div>
                    <div class="setup-distribution-plan-grid">
                        <div class="setup-distribution-plan-card">
                            <div class="setup-distribution-plan-card-header">
                                <div>
                                    <p class="text-xs font-semibold text-slate-700">Round Coverage Plan</p>
                                    <p class="text-[11px] text-slate-500">How the training split is covered across aggregation rounds.</p>
                                </div>
                                <span class="setup-distribution-plan-chip">Training split</span>
                            </div>
                            <div class="setup-distribution-plan-control">
                                ${this._field('round_train_schedule_mode', 'Coverage Mode', 'select', UI_DATA.roundTrainScheduleModeOptions, { advanced: true })}
                            </div>
                            <div id="round-train-schedule-simple-note" class="setup-distribution-note hidden" data-advanced-field="true">
                                Simple mode automatically spreads the federated training split across aggregation rounds and reuses the prepared client distribution in each round. Validation and test stay fixed every round.
                            </div>
                            <div id="round-train-schedule-summary-top" class="setup-distribution-summary-row">
                                <span id="round-train-schedule-round-count-top" class="setup-distribution-summary-pill">0 rounds</span>
                                <span id="round-train-schedule-total-top" class="setup-distribution-summary-pill">Total: 0.00%</span>
                                <span id="round-train-schedule-status-top" class="setup-distribution-summary-pill">Waiting for aggregation rounds</span>
                            </div>
                        </div>

                        <div class="setup-distribution-plan-card">
                            <div class="setup-distribution-plan-card-header">
                                <div>
                                    <p class="text-xs font-semibold text-slate-700">Round Client Plan</p>
                                    <p class="text-[11px] text-slate-500">How each round is assigned across the full planned client pool.</p>
                                </div>
                                <span class="setup-distribution-plan-chip">Client pool</span>
                            </div>
                            <div class="setup-field setup-distribution-plan-control">
                                <label class="block text-xs text-slate-400 mb-1">${this._labelWithHelp('round_client_allocation_mode', 'Client Mode')}</label>
                                <select id="cfg-round_client_allocation_mode" class="fl-input">
                                    ${UI_DATA.roundClientAllocationModeOptions.map(([value, label]) => `<option value="${value}">${label}</option>`).join('')}
                                </select>
                            </div>
                            <div id="round-client-allocation-runtime-note" class="setup-distribution-note hidden" data-advanced-field="true">
                                Allocation percentages describe the intended client plan, not a guaranteed per-client sample count. Faster resolves each round against the realized subset, so very small shares can round down to 0 samples, especially when the round itself covers only a tiny fraction of the training pool. Any client that ends up with 0 samples is skipped from training and aggregation for that round.
                            </div>
                            <div id="round-client-allocation-summary-top" class="setup-distribution-summary-row">
                                <span id="round-client-allocation-round-count-top" class="setup-distribution-summary-pill">0 rounds</span>
                                <span id="round-client-allocation-client-count-top" class="setup-distribution-summary-pill">0 clients</span>
                                <span id="round-client-allocation-active-label-top" class="setup-distribution-summary-pill">Select a round</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="round-editing-section" class="setup-distribution-section" data-advanced-field="true">
                    <div class="setup-distribution-section-header">
                        <div>
                            <p class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Round Editing</p>
                            <p class="text-[11px] text-slate-500">The detailed editors sit below the summaries so both plans stay easy to compare.</p>
                        </div>
                    </div>
                    <div class="setup-distribution-editor-grid">
                        <div id="round-train-schedule-manual-wrap" class="setup-distribution-editor-card hidden" data-advanced-field="true">
                            <label class="block text-xs text-slate-400 mb-1">${this._labelWithHelp('round_train_schedule_percentages', 'Round Coverage Plan')}</label>
                            <input id="cfg-round_train_schedule_percentages" type="hidden" aria-hidden="true" tabindex="-1">
                            <div class="setup-distribution-editor-body">
                                <p class="text-[11px] text-slate-500">Choose how much of the federated training split each aggregation round should cover. The full plan must add up to 100% across all rounds.</p>
                                <div class="setup-distribution-button-row">
                                    <button id="round-train-schedule-even-btn" type="button" class="ui-btn-compact ui-btn-neutral">Split Evenly</button>
                                    <button id="round-train-schedule-front-btn" type="button" class="ui-btn-compact ui-btn-neutral">Front-load Early Rounds</button>
                                    <button id="round-train-schedule-back-btn" type="button" class="ui-btn-compact ui-btn-neutral">Back-load Later Rounds</button>
                                </div>
                                <div id="round-train-schedule-summary-editor" class="setup-distribution-summary-row">
                                    <span class="setup-distribution-summary-pill" id="round-train-schedule-round-count-editor">0 rounds</span>
                                    <span class="setup-distribution-summary-pill" id="round-train-schedule-total-editor">Total: 0.00%</span>
                                    <span class="setup-distribution-summary-pill" id="round-train-schedule-status-editor">Waiting for aggregation rounds</span>
                                </div>
                                <div id="round-train-schedule-grid" class="max-h-72 overflow-y-auto pr-1 round-train-schedule-editor-grid"></div>
                            </div>
                        </div>

                        <div id="round-client-allocation-manual-wrap" class="setup-distribution-editor-card hidden" data-advanced-field="true">
                            <input id="cfg-round_client_allocation_percentages" type="hidden" aria-hidden="true" tabindex="-1">
                            <label class="block text-xs text-slate-400 mb-1">${this._labelWithHelp('round_client_allocation_percentages', 'Round Client Plan')}</label>
                            <div class="setup-distribution-editor-body">
                                <p class="text-[11px] text-slate-500">Define how each round&apos;s scheduled training subset should be divided across the full client pool. If churn makes some clients inactive or the round is very small, Faster may still assign 0 samples to some clients; those clients are skipped from training and aggregation for that round.</p>
                                <div class="setup-distribution-button-row">
                                    <button id="round-client-allocation-even-btn" type="button" class="ui-btn-compact ui-btn-neutral">Split Current Round Evenly</button>
                                    <button id="round-client-allocation-copy-btn" type="button" class="ui-btn-compact ui-btn-neutral">Copy Current Round To All</button>
                                </div>
                                <div id="round-client-allocation-summary-editor" class="setup-distribution-summary-row">
                                    <span id="round-client-allocation-editor-round-count" class="setup-distribution-summary-pill">0 rounds</span>
                                    <span id="round-client-allocation-editor-client-count" class="setup-distribution-summary-pill">0 clients</span>
                                    <span id="round-client-allocation-editor-active-label" class="setup-distribution-summary-pill">Select a round</span>
                                </div>
                                <div id="round-client-allocation-grid" class="setup-distribution-client-grid"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Builds training parameter fields.
     * @returns {string} HTML fields fragment.
     */
    _renderTrainingParameterFields() {
        return [
            this._field('batch_size', 'Batch Size', 'number'),
            this._field('server_learning_rate', 'Server Learning Rate', 'number', null, 'any'),
            `<input id="cfg-learning_rate" type="hidden">`,
            this._field('momentum', 'Momentum', 'number', null, 'any', { advanced: true }),
            this._field('mu', 'µ (FedProx)', 'number', null, 'any', { advanced: true }),
            `<div class="setup-field setup-field-checkbox" data-advanced-field="true">
                <label class="block text-xs text-slate-400 mb-1">${this._labelWithHelp('fedprox_weighted', 'FedProx Weighted Agg.')}</label>
                <label class="setup-checkbox-tile">
                    <input type="checkbox" id="cfg-fedprox_weighted" class="setup-checkbox-input" checked>
                    <span class="text-[11px] leading-tight text-slate-500">Weighted by client samples (uncheck = uniform)</span>
                </label>
            </div>`,
            this._field('death_prob', 'Death Prob.', 'number', null, '0.0001', { advanced: true, min: 0, max: 0.9 }),
            this._field('new_client_prob', 'New Client Prob.', 'number', null, '0.0001', { advanced: true, min: 0, max: 0.9 }),
            this._field('server_warmup_epochs', 'Server Warm-up Epochs', 'number'),
            this._field('local_model_epochs', 'Local Epochs', 'number'),
            this._field('weights_sending_frequency', 'Weights Freq.', 'number', null, null, { advanced: true }),
            `<div id="client-churn-warning" class="setup-field setup-field-wide hidden rounded-lg border px-2.5 py-1.5 text-[11px] leading-tight" data-advanced-field="true"></div>`,
            `<div class="setup-field setup-field-wide lr-compact-block" data-advanced-field="true">
                <label class="block text-[11px] text-slate-400 mb-1">${this._labelWithHelp('client_learning_rates', 'Client Learning Rates')}</label>
                <input id="cfg-client_learning_rates" type="text" class="hidden" aria-hidden="true" tabindex="-1">
                <div class="lr-layout">
                    <div class="lr-layout-top">
                        <div class="lr-mode-select-wrap">
                            <select id="cfg-client_lr_mode" class="fl-input lr-compact-input lr-mode-select">
                                <option value="uniform">Uniform: one value for all clients</option>
                                <option value="manual">Manual: N inputs (one per client)</option>
                                <option value="random">Random: range [min, max] per client</option>
                            </select>
                        </div>
                        <div id="client-lr-mode-mobile" class="lr-mode-mobile" role="group" aria-label="Client learning rate mode">
                            <button type="button" class="lr-mode-mobile-btn" data-lr-mode="uniform">Uniform</button>
                            <button type="button" class="lr-mode-mobile-btn" data-lr-mode="manual">Manual</button>
                            <button type="button" class="lr-mode-mobile-btn" data-lr-mode="random">Random</button>
                        </div>
                    </div>
                    <div id="client-lr-uniform-wrap" class="lr-mode-panel">
                        <div class="lr-uniform-inline">
                            <label class="lr-inline-label" for="cfg-client_lr_uniform">Uniform client learning rate</label>
                            <input id="cfg-client_lr_uniform" type="number" step="any" class="fl-input lr-compact-input lr-inline-input" placeholder="e.g. 0.001">
                        </div>
                    </div>
                    <div id="client-lr-manual-wrap" class="hidden lr-mode-panel">
                        <p class="text-[10px] text-slate-500 mb-0.5">One value per client.</p>
                        <div id="client-lr-manual-grid" class="lr-client-rail"></div>
                    </div>
                    <div id="client-lr-random-wrap" class="hidden lr-mode-panel">
                        <p class="text-[10px] text-slate-500 mb-0.5">Set min/max per client; values are sampled at launch.</p>
                        <div id="client-lr-random-grid" class="lr-client-rail lr-client-rail-random"></div>
                    </div>
                </div>
            </div>`,
        ].join('');
    },

    /**
     * Builds genetic programming fields.
     * @returns {string} HTML fields fragment.
     */
    _renderGeneticProgrammingFields() {
        return [
            this._field('individuals', 'Individuals', 'number'),
            this._field('generations', 'Generations', 'number'),
            this._field('mutation_rate', 'Mutation Rate', 'number', null, 'any'),
            this._field('crossover_rate', 'Crossover Rate', 'number', null, 'any'),
            this._field('elitism_size', 'Elitism Size', 'number'),
            this._field('gp_patience', 'GP Patience', 'number'),
            this._field('max_tree_size', 'Max Tree Size', 'number', null, 'any'),
            `<div>
                <label class="block text-xs text-slate-400 mb-1">${this._labelWithHelp('min_tree_size', 'Min Tree Size')}</label>
                <div class="flex items-center gap-2">
                    <input type="checkbox" id="cfg-min_tree_size_enabled"
                        class="rounded border-slate-600 bg-slate-700 text-teal-500">
                    <input id="cfg-min_tree_size" type="number" step="any" value="2" disabled
                        class="fl-input flex-1">
                </div>
            </div>`,
            this._field('mutation_subtree_maxsize', 'Mutation Subtree Max Size', 'number', null, { min: 1 }),
            this._field('gp_fitness_metric', 'Fitness Metric', 'select', UI_DATA.gpFitnessMetricOptions),
            this._field('gp_initialization', 'Initialization', 'select', UI_DATA.gpInitializationOptions),
            this._field('mutation_type', 'Mutation Type', 'select', UI_DATA.mutationTypeOptions),
            this._field('crossover_type', 'Crossover Type', 'select', UI_DATA.crossoverTypeOptions),
            this._field('selection_type', 'Selection Type', 'select', UI_DATA.selectionTypeOptions),
            this._field('gp_transfer_learning', 'Transfer Learning Strategy', 'select', UI_DATA.gpTransferLearningOptions),
            `<div id="gp-population-warning" class="setup-field setup-field-wide hidden rounded-lg border px-2.5 py-1.5 text-[11px] leading-tight"></div>`,
        ].join('');
    },

    /**
     * Builds history chart card containers.
     * @returns {string} HTML chart cards fragment.
     */
    _renderHistoryChartCards() {
        return UI_DATA.historyChartCards.map(([id, label]) => `
            <div id="${id}-card" class="bg-slate-800 border border-slate-700/60 rounded-xl p-4">
                <p class="text-xs font-medium text-slate-400 mb-2">${label}</p>
                <div class="chart-box"><canvas id="${id}"></canvas></div>
            </div>
        `).join('');
    },

    /**
     * Renders a generic input/select field block.
     * @param {string} key Field key.
     * @param {string} label Field label.
     * @param {string} type Field type.
     * @param {Array<Array<string>>|null} [options=null] Select options.
     * @param {string|null} [step=null] Numeric step value.
     * @returns {string} HTML field fragment.
     */
    _field(key, label, type, options = null, stepOrOpts = null, maybeOpts = null) {
        const id = `cfg-${key}`;
        const labelHtml = this._labelWithHelp(key, label);
        const hasInlineOpts = stepOrOpts && typeof stepOrOpts === 'object' && !Array.isArray(stepOrOpts);
        const step = hasInlineOpts ? null : stepOrOpts;
        const opts = hasInlineOpts ? stepOrOpts : (maybeOpts || {});
        const wrapperAttrs = [];
        const wrapperClasses = ['setup-field'];
        if (opts.advanced) wrapperAttrs.push('data-advanced-field="true"');
        if (opts.wide) wrapperClasses.push('setup-field-wide');

        if (type === 'select') {
            const safeOptions = Array.isArray(options) && options.length
                ? options
                : [['', 'No options available']];
            return `<div class="${wrapperClasses.join(' ')}" ${wrapperAttrs.join(' ')}>
                <label class="block text-xs text-slate-400 mb-1">${labelHtml}</label>
                <select id="${id}" class="fl-input">
                    ${safeOptions.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}
                </select>
            </div>`;
        }
        if (type === 'multiselect') {
            const safeOptions = Array.isArray(options) && options.length
                ? options
                : [['', 'No options available']];
            return `<div class="${[...wrapperClasses, 'setup-field-wide'].join(' ')}" ${wrapperAttrs.join(' ')}>
                <label class="block text-xs text-slate-400 mb-1">${labelHtml}</label>
                <div id="${id}-picker" class="method-chip-picker">
                    ${safeOptions.map(([v, t, mode]) => `<button type="button" class="method-chip" data-value="${v}"${mode === 'advanced' ? ' data-advanced-method="true"' : ''}>${t}</button>`).join('')}
                </div>
                <select id="${id}" class="fl-input method-multiselect-native" multiple size="5" aria-hidden="true" tabindex="-1">
                    ${safeOptions.map(([v, t, mode]) => `<option value="${v}"${mode === 'advanced' ? ' data-advanced-method="true"' : ''}>${t}</option>`).join('')}
                </select>
                <p class="text-xs text-slate-500 mt-1">Select one or more aggregation methods. <span class="help-dot" tabindex="0" role="img" aria-label="If you select multiple methods, launching creates one job per selected method with the same shared parameters." data-tooltip="If you select multiple methods, launching creates one job per selected method with the same shared parameters.">?</span></p>
            </div>`;
        }
        const stepAttr = step ? `step="${step}"` : (type === 'number' ? 'step="1"' : '');
        const minAttr = opts.min !== undefined ? `min="${opts.min}"` : '';
        const maxAttr = opts.max !== undefined ? `max="${opts.max}"` : '';
        return `<div class="${wrapperClasses.join(' ')}" ${wrapperAttrs.join(' ')}>
            <label class="block text-xs text-slate-400 mb-1">${labelHtml}</label>
            <input id="${id}" type="${type}" ${stepAttr} ${minAttr} ${maxAttr} class="fl-input">
        </div>`;
    },

    /**
     * Renders a checkbox field block.
     * @param {string} key Field key.
     * @param {string} label Field label.
     * @returns {string} HTML checkbox fragment.
     */
    _checkboxField(key, label, opts = {}) {
        const labelHtml = this._labelWithHelp(key, label);
        const attrs = opts.advanced ? ' data-advanced-field="true"' : '';
        return `<div class="setup-field flex flex-col justify-end"${attrs}>
            <label class="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" id="cfg-${key}"
                    class="w-4 h-4 rounded border-slate-600 bg-slate-700 text-teal-500 focus:ring-teal-500">
                <span class="text-xs text-slate-300">${labelHtml}</span>
            </label>
        </div>`;
    },
};
