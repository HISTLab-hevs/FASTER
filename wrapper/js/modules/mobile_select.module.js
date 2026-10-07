/**
 * mobile_select.module.js
 *
 * Replaces native HTML <select> elements with a custom JavaScript-driven dropdown
 * component. This solves critical mobile UI issues where native OS popups cannot be
 * reliably constrained by CSS, resulting in overflow beyond `max-width: 100vw`.
 *
 * Every single-choice <select> in the app is upgraded automatically: `observe()`
 * scans the document and watches for markup added later, so callers no longer have
 * to register dropdowns by id. Each instance keeps itself in sync with the native
 * element (option list, disabled/hidden flags and programmatic `.value` writes),
 * which removes the need for manual refresh calls after repopulating a select.
 *
 * Selects marked with `data-msb-skip` (or the hidden multi-select mirror behind the
 * aggregation-method chip picker) are left untouched.
 */
window.MobileSelect = (function () {

    /** @type {WeakMap<HTMLSelectElement, Object>} Active dropdown instances, keyed by element */
    const instances = new WeakMap();

    /** @type {MutationObserver|null} Document-level observer upgrading selects added later */
    let documentObserver = null;

    /**
     * Reports whether a select must keep its native rendering.
     *
     * @private
     * @param {HTMLSelectElement} selectEl - Candidate select element.
     * @returns {boolean} True when the element must not be upgraded.
     */
    function _isExcluded(selectEl) {
        return selectEl.multiple
            || selectEl.hasAttribute('data-msb-skip')
            || selectEl.classList.contains('method-multiselect-native');
    }

    /**
     * Builds and injects the custom dropdown markup into the DOM, effectively hijacking
     * the designated select element. Hides the native element but maintains it inline
     * so form sequences continue to extract standard values seamlessly.
     *
     * @private
     * @param {HTMLSelectElement} selectEl - Target select element reference.
     */
    function _buildDropdown(selectEl) {
        const wrapper = document.createElement('div');
        wrapper.className = 'msb-wrapper';
        if (selectEl.id) wrapper.setAttribute('data-msb', selectEl.id);
        // Grid/flex placement classes belong to the slot the select occupied, so they
        // move onto the wrapper that now sits in that slot.
        Array.from(selectEl.classList)
            .filter((cls) => /^(sm:|md:|lg:)?col-span-/.test(cls) || cls === 'flex-1')
            .forEach((cls) => wrapper.classList.add(cls));

        const trigger = document.createElement('button');
        trigger.type = 'button';
        // Mirror the sizing utilities of the native control so an upgraded dropdown
        // keeps the exact footprint the layout was designed around.
        const sizingClasses = ['text-xs', 'text-sm', 'w-full']
            .filter((cls) => selectEl.classList.contains(cls));
        trigger.className = ['msb-trigger', 'fl-input', ...sizingClasses].join(' ');
        trigger.setAttribute('aria-haspopup', 'listbox');
        trigger.setAttribute('aria-expanded', 'false');

        const panel = document.createElement('div');
        panel.className = 'msb-panel';
        panel.setAttribute('role', 'listbox');
        panel.setAttribute('tabindex', '-1');
        panel.hidden = true;

        /** Generates custom list item elements aligned to standard options targets */
        function buildOptions() {
            panel.innerHTML = '';
            Array.from(selectEl.options).forEach((opt) => {
                if (opt.hidden || opt.disabled) return;
                const item = document.createElement('div');
                item.className = 'msb-option' + (opt.selected ? ' msb-option-selected' : '');
                item.setAttribute('role', 'option');
                item.setAttribute('data-value', opt.value);
                item.textContent = opt.text;

                // Clicking an option updates the native value and forcefully fires 'change' Event
                item.addEventListener('click', () => {
                    selectEl.value = opt.value;
                    selectEl.dispatchEvent(new Event('change', { bubbles: true }));
                    closePanel();
                });
                panel.appendChild(item);
            });
        }

        /** Syncs the main trigger button preview representation label texts */
        function updateTriggerText() {
            const selected = selectEl.options[selectEl.selectedIndex];
            trigger.textContent = selected ? selected.text : '—';
            trigger.disabled = selectEl.disabled;
        }

        /** Expands the selection layer logic panel limits */
        function openPanel() {
            if (selectEl.disabled) return;
            buildOptions();
            panel.hidden = false;
            trigger.setAttribute('aria-expanded', 'true');
            wrapper.classList.add('msb-open');
        }

        /** Forces dropdown occlusion representation updates */
        function closePanel() {
            panel.hidden = true;
            trigger.setAttribute('aria-expanded', 'false');
            wrapper.classList.remove('msb-open');
            updateTriggerText();
            panel.querySelectorAll('.msb-option').forEach(el => {
                el.classList.toggle('msb-option-selected', el.getAttribute('data-value') === selectEl.value);
            });
        }

        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            if (panel.hidden) openPanel();
            else closePanel();
        });

        document.addEventListener('click', (e) => {
            if (!wrapper.contains(e.target)) closePanel();
        }, true);

        wrapper.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closePanel();
        });

        // External events directly overriding select values are trapped and reflected here
        selectEl.addEventListener('change', updateTriggerText);

        // Options are frequently rebuilt via innerHTML, or toggled through the
        // hidden/disabled flags. Watching the element keeps the trigger label and
        // the open panel aligned without callers having to refresh manually.
        const optionObserver = new MutationObserver(() => {
            updateTriggerText();
            if (!panel.hidden) buildOptions();
        });
        optionObserver.observe(selectEl, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['hidden', 'disabled', 'selected', 'value'],
        });

        // Programmatic `el.value = x` assignments fire no event, so the property is
        // wrapped on this instance to refresh the visible label in the same tick.
        const valueDescriptor = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value');
        if (valueDescriptor && valueDescriptor.get && valueDescriptor.set) {
            Object.defineProperty(selectEl, 'value', {
                configurable: true,
                get() {
                    return valueDescriptor.get.call(this);
                },
                set(next) {
                    valueDescriptor.set.call(this, next);
                    updateTriggerText();
                },
            });
        }

        selectEl.style.display = 'none';
        selectEl.parentNode.insertBefore(wrapper, selectEl);
        wrapper.appendChild(trigger);
        wrapper.appendChild(panel);
        wrapper.appendChild(selectEl);

        updateTriggerText();

        instances.set(selectEl, { wrapper, trigger, panel, selectEl, openPanel, closePanel, updateTriggerText });
    }

    /**
     * Upgrades one select element, resolved either by id or by direct reference.
     *
     * @param {string|HTMLSelectElement} target - Element id or select element.
     */
    function init(target) {
        const el = typeof target === 'string' ? document.getElementById(target) : target;
        if (!el || el.tagName !== 'SELECT') return;
        if (_isExcluded(el)) return;
        if (instances.has(el)) return;
        if (!el.parentNode) return;
        _buildDropdown(el);
    }

    /**
     * Upgrades every eligible select currently present under one root node.
     *
     * @param {ParentNode} [root=document] - Subtree to scan.
     */
    function upgradeAll(root = document) {
        const scope = root && root.querySelectorAll ? root : document;
        if (scope.tagName === 'SELECT') init(scope);
        scope.querySelectorAll('select').forEach((el) => init(el));
    }

    /**
     * Starts the document-level watcher that upgrades dropdowns rendered later.
     * Safe to call repeatedly; the observer is created once.
     */
    function observe() {
        upgradeAll(document);
        if (documentObserver) return;
        documentObserver = new MutationObserver((records) => {
            for (const record of records) {
                record.addedNodes.forEach((node) => {
                    if (node.nodeType !== 1) return;
                    upgradeAll(node);
                });
            }
        });
        documentObserver.observe(document.body, { childList: true, subtree: true });
    }

    /**
     * Re-syncs a mounted dropdown with its native select. Kept for explicit callers;
     * instances already refresh themselves when their options or value change.
     *
     * @param {string|HTMLSelectElement} target - Element id or select element.
     */
    function refresh(target) {
        const el = typeof target === 'string' ? document.getElementById(target) : target;
        const inst = el && instances.get(el);
        if (!inst) return;
        if (!inst.panel.hidden) {
            inst.panel.hidden = true;
            inst.trigger.setAttribute('aria-expanded', 'false');
            inst.wrapper.classList.remove('msb-open');
        }
        inst.updateTriggerText();
    }

    return { init, refresh, observe, upgradeAll };
})();
