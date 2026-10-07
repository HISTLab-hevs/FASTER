Reference Index
===============

The pages below point at the current Python module paths used by the Faster
application. They are grouped by runtime responsibility so contributors can
start from the relevant subsystem before drilling into autodoc output.

Entrypoints and backend packages
--------------------------------

.. toctree::
   :maxdepth: 1

   modules/web_backend_app
   modules/web_backend___init__
   modules/web_backend_service
   modules/web_backend_api_handlers
   modules/web_backend_api_handlers_api
   modules/web_backend_api_handlers_auth
   modules/web_backend_api_handlers_admin
   modules/web_backend_api_handlers_content
   modules/web_backend_api_handlers_run_workflows
   modules/web_backend_api_handlers_runs
   modules/web_backend_api_handlers_shared
   modules/web_backend_jm_client
   modules/web_backend_job_cli
   modules/web_backend_contracts
   modules/web_backend_services___init__
   modules/web_backend_services_alerts
   modules/web_backend_services_datasets
   modules/web_backend_services_defaults
   modules/web_backend_services_run_artifacts
   modules/web_backend_services_run_lifecycle
   modules/web_backend_services_scenarios

Auth and persistence
--------------------

.. toctree::
   :maxdepth: 1

   modules/auth___init__
   modules/auth_authenticator
   modules/auth_user_cli
   modules/database___init__
   modules/database_config
   modules/database_connection
   modules/database_exceptions
   modules/database_interfaces
   modules/database_models
   modules/database_repositories___init__
   modules/database_repositories_app_settings_repository
   modules/database_repositories_alerts_repository
   modules/database_repositories_base
   modules/database_repositories_custom_datasets_repository
   modules/database_repositories_runs_repository
   modules/database_repositories_scenarios_repository
   modules/database_repositories_users_repository

Training engine and utilities
-----------------------------

.. toctree::
   :maxdepth: 1

   modules/main
   modules/utils_aggregation_methods
   modules/utils_config
   modules/utils_dataset
   modules/utils_evaluate
   modules/utils_extract_results
   modules/utils_gp
   modules/utils_gp_population
   modules/utils_gp_primitives
   modules/utils_gpu_check
   modules/utils_metrics
   modules/utils_model
   modules/utils_multiplot_results
   modules/utils_plot_results
   modules/round_schedule
   modules/utils_run_config_validation
   modules/utils_seed
   modules/utils_train

Bundled Job Manager subsystem
-----------------------------

.. toctree::
   :maxdepth: 1

   modules/job_manager_api_run
   modules/job_manager_api_api
   modules/job_manager_api_auth
   modules/job_manager_api_models
   modules/job_manager_api_config
   modules/job_manager_api_db
   modules/job_manager_api_job_db
   modules/job_manager_api_user_db
   modules/job_manager_api_job_manager
   modules/job_manager_api_notifier
   modules/job_manager_api_simulation
   modules/job_manager_api_logger
   modules/job_manager_api_manage_users
