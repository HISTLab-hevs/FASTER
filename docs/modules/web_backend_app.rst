web_backend.app
===============

The Faster web runtime is the FastAPI ASGI application exposed as
``web_backend.app:app``. It owns the HTTP surface for the SPA, API routes,
upload and download endpoints, and the browser-facing security headers.

Application Entrypoint
----------------------

.. automodule:: web_backend.app
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:

Backend Services
----------------

- :doc:`web_backend___init__`
- :doc:`web_backend_api_handlers`
- :doc:`web_backend_jm_client`
- :doc:`web_backend_service`

Training Engine
---------------

- :doc:`main`
- :doc:`utils_aggregation_methods`
- :doc:`utils_config`
- :doc:`utils_dataset`
- :doc:`utils_evaluate`
- :doc:`utils_extract_results`
- :doc:`utils_gp`
- :doc:`utils_gp_population`
- :doc:`utils_gp_primitives`
- :doc:`utils_gpu_check`
- :doc:`utils_metrics`
- :doc:`utils_model`
- :doc:`utils_multiplot_results`
- :doc:`utils_plot_results`
- :doc:`utils_train`

Authentication
--------------

- :doc:`auth___init__`
- :doc:`auth_authenticator`
- :doc:`auth_user_cli`

Persistence Layer
-----------------

- :doc:`database___init__`
- :doc:`database_config`
- :doc:`database_connection`
- :doc:`database_exceptions`
- :doc:`database_interfaces`
- :doc:`database_models`
- :doc:`database_repositories___init__`
- :doc:`database_repositories_alerts_repository`
- :doc:`database_repositories_base`
- :doc:`database_repositories_custom_datasets_repository`
- :doc:`database_repositories_runs_repository`
- :doc:`database_repositories_scenarios_repository`
- :doc:`database_repositories_users_repository`

Bundled Job Manager Subsystem
-----------------------------

FASTER integrates with a bundled generic Job Manager subsystem. The Faster-side
integration points and the Job Manager subsystem reference are documented in
this single Sphinx tree.

- :doc:`job_manager_api_run`
- :doc:`job_manager_api_api`
- :doc:`job_manager_api_job_manager`
- :doc:`job_manager_api_job_db`
