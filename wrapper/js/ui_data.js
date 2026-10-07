/**
 * ui_data.js — Static UI datasets and configuration mappings.
 * 
 * This file contains constant arrays and dictionaries used to generate 
 * dynamic markup fragments, populate dropdowns, and provide tooltip descriptions 
 * across the application's frontend.
 */

window.UI_DATA = {
    /**
     * Tooltip descriptions for all federated learning parameters.
     * Used dynamically by the info-dot hover popovers in the training setup form.
     * @type {Object<string, string>}
     */
    paramDescriptions: {
        method: 'Federated learning algorithm used for model aggregation. In Simple mode, FedGP uses the default non-configurable GP preset until you switch to Advanced mode.',
        num_clients: 'Fixed total client pool size used for federated training.',
        initial_eligible_clients: 'How many client IDs begin in the eligible set before per-round churn is applied. Valid range: 1 to num_clients. The lowest client IDs are used first.',
        batch_size: 'Size of the mini-batch used for local training on each client.',
        server_warmup_epochs: 'Number of server warm-up epochs performed before federated rounds begin.',
        local_model_epochs: 'Number of training epochs per client before communication.',
        weights_sending_frequency: 'Frequency (in epochs) at which local models send weights to the server.',
        server_learning_rate: 'Learning rate used only for the initial server-side warm-up optimizer before federated rounds begin.',
        client_learning_rates: 'Controls the effective client learning rates: Uniform (one value for all clients), Manual (one per client), or Random (min/max range per client).',
        death_prob: 'Probability, applied independently to each eligible client before every round, that the client becomes inactive. Valid range: 0.0 to 0.9.',
        new_client_prob: 'Probability, applied independently to each inactive client before every round, that the client becomes eligible again. Dropout is not permanent and the eligible set carries over between rounds. Valid range: 0.0 to 0.9.',
        seed: 'Base seed for the whole run: it seeds dataset partitioning, training, and client churn so a run is fully reproducible. Each selected aggregator launches its own run sharing this base seed; repeats derive a distinct seed (base + repeat index).',
        momentum: 'Momentum factor used in the local optimizer.',
        dataset_name: 'Dataset used for training.',
        server_data_percentage: 'Share of the training data retained by the central server (used for the server warm-up).',
        iid: 'Whether the training data is distributed identically and independently across clients (True/False). It affects only how clients see training data; the whole validation/test set evaluates the global model.',
        imbalance_rate: 'Degree of training-data imbalance among clients under non-IID partitioning: values closer to 0 are more balanced, values closer to 1 are more imbalanced. Must be strictly between 0 and 1; use iid instead for a fully balanced split.',
        save_path: 'File path where the training results and logs will be saved.',
        individuals: 'Number of individuals in each generation for genetic programming.',
        generations: 'Total number of generations to evolve in genetic programming.',
        elitism_size: 'Number of top individuals preserved unmodified in the next generation.',
        mutation_rate: 'Probability of applying mutation to an individual in genetic programming.',
        crossover_rate: 'Probability of performing crossover between two individuals.',
        gp_patience: 'Number of generations without improvement before early stopping GP (if less than total generations).',
        gp_initialization: 'Strategy used to initialize GP trees.',
        mutation_type: 'Type of mutation used in GP.',
        selection_type: 'Selection method to choose parents in GP.',
        crossover_type: 'Crossover strategy used to combine two individuals.',
        min_tree_size: 'Minimum number of nodes in a GP tree (integer or False to use the backend default of 1).',
        max_tree_size: 'Maximum number of nodes in a GP tree. The backend requires an integer in the range 1 to 64.',
        mutation_subtree_maxsize: 'Max height of the random subtree grafted on mutation. Smaller values limit tree bloat. Integer from 1 to 64 (default 2).',
        gp_fitness_metric: 'Metric the GP optimizes: accuracy/precision/recall/F1 (maximized, macro-averaged) or loss (minimized). Default accuracy.',
        available_primitives: 'Comma-separated list of allowed operations for GP trees.',
        gp_transfer_learning: "Transfer learning strategy for GP: disabled, full_reuse, elite_reuse, elite_mutation_warm_start, or hybrid.",
        mu: 'FedProx: regularization term to penalize deviation from the global model.',
        fedprox_weighted: 'FedProx aggregation: checked = weighted by client sample counts (n_k/n, the paper rule); unchecked = uniform average across clients.',
        repeat: 'Number of times to repeat the same experiment for statistical robustness. Each repeat uses a distinct seed derived from the base seed (base_seed + repeat_index), so repetitions are statistically independent yet fully reproducible: re-running with the same base seed reproduces the exact same set of runs.',
        train_val_split: 'Train/Validation split ratio (e.g. 0.9). Applies only to FashionMNIST and CIFAR; MedMNIST and custom datasets carry their own train/val/test splits.',
        evaluation_split_mode: 'Evaluation mode: train_val_test evaluates both validation and test; train_val evaluates validation only.',
        round_train_schedule_mode: 'Controls the round coverage plan for the federated training split. Automatic mode spreads coverage for you. Manual mode lets you decide how much of the training split each round should cover. No split reuses 100% of the training data every round (clients still split that pool among themselves). Validation and test stay fixed every round.',
        round_train_schedule_percentages: 'Round coverage plan editor. Add one percentage for each aggregation round so the full plan totals 100%. This controls round coverage only; client allocation is configured separately.',
        round_client_allocation_mode: 'Controls the round client plan. Automatic mode reuses the prepared client split for every round. Manual mode lets you edit the client split round by round. The percentages are a plan, not a guarantee that every positively weighted client will receive samples in every round.',
        round_client_allocation_percentages: 'Round client plan editor. The allocation is defined against the full planned client pool, but actual sample counts are resolved after the round subset is realized. If churn makes some clients inactive or the round coverage is very small, some clients can still round down to 0 samples and will be skipped from training and aggregation for that round.',
        custom_model_mode: 'Choose model source: default architecture, inline code, or uploaded Python file.',
        custom_model_code: 'Python code defining build_model(n_channels, n_classes).',
        custom_dataset_ref: 'Reference to a validated uploaded custom dataset (.csv, .tsv, .npz, or .zip image-folder archive).',
        custom_dataset_path: 'Server-resolved path for a validated custom dataset asset.',
    },

    /**
     * Landing page feature highlights.
     * Format: [iconClass, title, description]
     * @type {Array<Array<string>>}
     */
    landingFeatures: [
        ['ph-chart-pie', 'Multiple Aggregators', 'FedAvg, FedProx, FedNova, FedGP'],
        ['ph-gauge', 'Scalable Runs', 'Compare multiple runs with robust analytics'],
        ['ph-database', 'Medical Datasets', 'MedMNIST: PathMNIST, PneumoniaMNIST...'],
    ],

    /**
     * Primary sidebar navigation tabs for general users.
     * Format: [iconClass, label, tabIndexOffset]
     * @type {Array<Array<string|number>>}
     */
    navTabs: [
        ['ph-database', 'Datasets', 0, 'Datasets'],
        ['ph-flask', 'Experiment Configuration', 1, 'Setup'],
        ['ph-chart-line', 'Training Monitor', 2, 'Monitor'],
        ['ph-books', 'Run History & Analysis', 3, 'History'],
        ['ph-cpu', 'Infrastructure Status', 4, 'Status'],
    ],

    /**
     * Additional sidebar navigation tabs for administrators.
     * Format: [iconClass, label, tabIndexOffset]
     * @type {Array<Array<string|number>>}
     */
    adminNavTabs: [
        ['ph-chart-bar', 'Platform Analytics', 5, 'Analytics'],
        ['ph-user-plus', 'User Management', 6, 'Users'],
    ],

    /**
     * Dropdown options for federated learning aggregation methods.
     * Format: [value, label]
     * @type {Array<Array<string>>}
     */
    methodOptions: [
        ['fed_avgw', 'FedAvgW (weighted)'],
        ['fed_avg', 'FedAvg'],
        ['fed_prox', 'FedProx'],
        ['fed_nova', 'FedNova'],
        ['fed_gp', 'FedGP'],
    ],

    /**
     * Dropdown options for datasets exposed by the frontend setup form.
     * Values in this list must stay aligned with backend-accepted dataset_name values.
     * Format: [value, label]
     * @type {Array<Array<string>>}
     */
    datasetOptions: [
        ['pathmnist', 'PathMNIST'], ['pneumoniamnist', 'PneumoniaMNIST'],
        ['bloodmnist', 'BloodMNIST'], ['dermamnist', 'DermaMNIST'],
        ['octmnist', 'OCTMNIST'], ['organamnist', 'OrganAMNIST'],
        ['organcmnist', 'OrganCMNIST'], ['organsmnist', 'OrganSMNIST'],
        ['tissuemnist', 'TissueMNIST'], ['fashionmnist', 'FashionMNIST'],
        ['cifar100', 'CIFAR100'], ['cifar10', 'CIFAR10'],
        ['custom', 'Custom dataset (CSV, NPZ, or ZIP image folders)'],
    ],

    /** @type {Array<Array<string>>} */
    evaluationSplitModeOptions: [
        ['train_val_test', 'Train/Val/Test'],
        ['train_val', 'Train/Val only'],
    ],

    /** @type {Array<Array<string>>} */
    roundTrainScheduleModeOptions: [
        ['auto', 'Automatic coverage plan'],
        ['manual', 'Manual coverage plan'],
        ['no_split', 'No split (reuse full data each round)'],
    ],

    /** @type {Array<Array<string>>} */
    roundClientAllocationModeOptions: [
        ['auto', 'Automatic client plan'],
        ['manual', 'Manual client plan'],
    ],

    /** @type {Array<Array<string>>} */
    gpInitializationOptions: [
        ['genHalfAndHalf', 'Half-and-half'], ['genFull', 'Full'], ['genGrow', 'Grow'],
    ],

    /** @type {Array<Array<string>>} */
    gpFitnessMetricOptions: [
        ['accuracy', 'Accuracy (maximize)'],
        ['loss', 'Loss (minimize)'],
        ['f1', 'F1 macro (maximize)'],
        ['precision', 'Precision macro (maximize)'],
        ['recall', 'Recall macro (maximize)'],
    ],

    /** @type {Array<Array<string>>} */
    mutationTypeOptions: [
        ['mutUniform', 'mutUniform'], ['mutNodeReplacement', 'mutNodeReplacement'],
        ['mutInsert', 'mutInsert'],
    ],

    /** @type {Array<Array<string>>} */
    crossoverTypeOptions: [
        ['cxOnePoint', 'cxOnePoint'], ['cxOnePointLeafBiased', 'cxOnePointLeafBiased'],
    ],

    /** @type {Array<Array<string>>} */
    selectionTypeOptions: [
        ['selTournament', 'selTournament'], ['selRoulette', 'selRoulette'], ['selRandom', 'selRandom'],
    ],

    /** @type {Array<Array<string>>} */
    gpTransferLearningOptions: [
        ['disabled', 'Disabled (cold-start each round)'],
        ['full_reuse', 'full_reuse'],
        ['elite_reuse', 'elite_reuse'],
        ['elite_mutation_warm_start', 'elite_mutation_warm_start'],
        ['hybrid', 'hybrid'],
    ],

    /**
     * Configuration for rendering the metric charts in the History tab.
     * Format: [htmlId, chartTitle]
     * @type {Array<Array<string>>}
     */
    historyChartCards: [
        ['hist-val-acc', 'Global Validation Accuracy'],
        ['hist-val-loss', 'Global Validation Loss'],
        ['hist-test-acc', 'Global Test Accuracy'],
        ['hist-test-loss', 'Global Test Loss'],
    ],
};
