Frontend Architecture
=====================

The user interface is a single-page application served from ``wrapper/``. It is
not a separate deployed frontend service; it is mounted and served by the
Faster web backend.

Where the frontend lives
------------------------

- ``wrapper/index.html`` bootstraps the SPA.
- ``wrapper/js/app.js`` owns top-level state and page orchestration.
- ``wrapper/js/api.js`` is the frontend/backend command bridge.
- ``wrapper/js/modules/`` contains domain-focused runtime modules such as setup,
  training, history, datasets, jobs, admin, rendering, and events.
- ``wrapper/templates/`` contains the public pages and the authenticated
  dashboard shell.
- ``wrapper/css/`` contains the split stylesheet entrypoints.

Frontend to backend relationship
--------------------------------

The SPA does not call Job Manager directly. Its active runtime relationship is:

1. the SPA sends commands to the Faster backend via ``POST /api``
2. the Faster backend validates, persists, and shapes data for the UI
3. the Faster backend talks to Job Manager when background execution is needed

This means Job Manager is part of the overall runtime topology, but it is not a
frontend dependency boundary.

Contributor code map
--------------------

When working on a UI concern, start here:

- experiment setup, scenario loading, and custom dataset UX:
  ``wrapper/js/modules/setup.module.js``
  This flow now covers tabular uploads, NPZ image uploads, and ZIP archives
  that unpack into directory-based ``train/``/``val/``/``test/`` image
  datasets. It also owns simple/advanced mode behavior, including the
  fixed FedGP preset used in Simple mode, the structured round-schedule editor,
  the round-by-round client-allocation editor, and the mode-aware warnings
  shown in the setup wizard. Advanced mode is where the user can explicitly
  set round coverage, per-round client allocation, churn controls, initial
  eligible clients, seed, and the broader tuning fields that stay automatic or
  hidden in Simple mode.
- scenario loading and scenario save/apply behavior:
  ``wrapper/js/modules/training_scenarios.module.js``
  Saved scenarios can carry a lightweight ``ui_mode`` marker. When that marker
  is absent, the frontend infers the most appropriate setup mode from the
  loaded content and switches the wizard accordingly before applying the form
  values.
- training submission and config shaping:
  ``wrapper/js/modules/training.module.js``
- dataset manager and dataset metadata presentation:
  ``wrapper/js/modules/datasets.module.js``
- history, comparison, and run detail views:
  ``wrapper/js/modules/history.module.js``
- runtime status, alerts, and polling:
  ``wrapper/js/modules/runtime.module.js``
- authenticated dashboard shell and global state:
  ``wrapper/js/app.js``

Default model contract and setup mode behavior
----------------------------------------------

The active frontend flow exposes two setup modes, but only ``Advanced`` mode
exposes custom-model controls. ``Simple`` mode keeps the default model path and
therefore depends on the current built-in model/data contract.

Code-grounded contract summary:

- Default image model (``utils/model.py::Net``) consumes tensors shaped
  ``N,C,H,W`` and integer class labels. Its fully connected head is fixed to
  ``Linear(64*4*4, ...)`` after the convolution stack, so it is designed for
  the current MedMNIST-like spatial regime (commonly ``28x28``) rather than
  arbitrary image sizes.
- Built-in MedMNIST/FashionMNIST loaders apply ``ToTensor`` plus
  ``Normalize((0.5,), (0.5,))``.
- Custom NPZ/ZIP image loaders convert data to float tensors, normalize channel
  layout to ``N,C,H,W``, and divide by ``255`` when values are in byte scale,
  but they do not auto-resize to a guaranteed default-model-safe resolution.
- Default tabular model (``utils/model.py::TabularNet``) expects numeric
  tabular features and integer class indices.

Operational guidance:

Not every dataset is guaranteed to work with the default model path in Simple
mode. If data shape/task assumptions do not match, users should switch to
Advanced mode and provide a compatible custom model.

Training flow and metric semantics
----------------------------------

The current run flow starts with a short server-side warm-up when configured,
then moves into round-based client training. Each round trains the selected
clients on its planned share of the federated training split, aggregates the
global model, and evaluates that updated model on validation and, when
available, test data.

Only the training data is partitioned across clients. Each round evaluates the
aggregated global model on the whole validation set (and the whole test set in
``train_val_test`` runs), so the reported validation/test metrics describe global
model quality across the full data distribution. Per-client metrics (loss,
accuracy, precision, recall, F1, ROC AUC) are evaluated on each client's own
training data.

In the live monitor, metric cards show the latest recorded value for each split. 
The current metric set includes accuracy, loss, precision, recall, F1, 
and ROC AUC. Validation is always shown, test metrics appear only for
``train_val_test`` runs, and ROC AUC can surface as ``N/A`` when the evaluated
split does not contain enough class structure to compute it.

History uses the same split-aware metric buckets, but it presents them
differently depending on the view:

- single-run details show the final recorded validation/test values for the
  run
- comparison tables show the best value seen in each metric series across the
  selected runs
- if the selected runs do not all include test results, comparison falls back
  to validation so the view stays consistent

Round scheduling UX
-------------------

The setup flow now treats round scheduling as a first-class product workflow
instead of a raw config entry field.

- Manual round coverage is edited through a structured per-round schedule UI
  with summary totals and helper presets. The percentages are a plan, not a
  promise of exact realized sample counts.
- Manual client allocation is edited round by round, with overview chips that
  let users inspect many aggregation rounds without expanding a full matrix all
  at once.
- User-facing validation is intentionally written in product language even
  though the backend keeps config-oriented field names internally.
- Churn-aware redistribution is explained in the UI wherever round-level client
  allocation is shown: the allocation is defined against the full planned
  client pool, then redistributed across eligible clients if churn makes some
  clients inactive in a round.

The frontend is documented here architecturally for discoverability. It is not
fully autodocumented through Sphinx because the active UI code is plain
JavaScript rather than Python modules suitable for autodoc.
