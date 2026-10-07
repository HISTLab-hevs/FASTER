-- FASTER MariaDB database.
-- Docker Compose initializes the target database first, then applies this file.
-- For manual bootstrap, run against the selected database, for example:
-- mariadb -u root -p faster < database/schema/baseline.sql

CREATE TABLE users (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  identifier VARCHAR(190) NOT NULL,
  email VARCHAR(320) NULL,
  password_hash TEXT NOT NULL,
  role ENUM('user','admin') NOT NULL DEFAULT 'user',
  active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME(6) NOT NULL,
  created_by VARCHAR(190) NOT NULL,
  UNIQUE KEY uq_users_identifier (identifier),
  UNIQUE KEY uq_users_email (email)
) ENGINE=InnoDB;

CREATE TABLE runs (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  run_id VARCHAR(190) NOT NULL,
  owner_identifier VARCHAR(190) NOT NULL,
  display_name VARCHAR(190) NOT NULL,
  status ENUM('pending','queued','running','completed','stopped','failed') NOT NULL DEFAULT 'queued',
  method VARCHAR(64) NULL,
  dataset_name VARCHAR(128) NULL,
  evaluation_split_mode VARCHAR(32) NULL,
  model_name VARCHAR(128) NULL,
  stop_reason TEXT NULL,
  best_accuracy DOUBLE NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uq_runs_run_id (run_id),
  KEY ix_runs_owner_created (owner_identifier, created_at),
  KEY ix_runs_status_created (status, created_at),
  CONSTRAINT fk_runs_owner_identifier
    FOREIGN KEY (owner_identifier) REFERENCES users(identifier)
    ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE run_configs (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  run_id VARCHAR(190) NOT NULL,
  config_json LONGTEXT NOT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uq_run_configs_run_id (run_id),
  CONSTRAINT fk_run_configs_run_id
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
    ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE jobs (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  job_id VARCHAR(190) NOT NULL,
  kind VARCHAR(32) NOT NULL DEFAULT 'pending',
  owner_identifier VARCHAR(190) NOT NULL,
  run_id VARCHAR(190) NULL,
  position INT NULL,
  summary_json LONGTEXT NULL,
  config_json LONGTEXT NULL,
  source_app VARCHAR(255) NULL,
  description TEXT NULL,
  job_type VARCHAR(32) DEFAULT 'default',
  status VARCHAR(50) NOT NULL DEFAULT 'queued',
  start_time VARCHAR(50) NULL,
  end_time VARCHAR(50) NULL,
  error TEXT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uq_jobs_job_id (job_id),
  KEY ix_jobs_owner_status (owner_identifier, status),
  KEY ix_jobs_start_time (start_time),
  KEY ix_jobs_created (created_at),
  CONSTRAINT fk_jobs_owner_identifier
    FOREIGN KEY (owner_identifier) REFERENCES users(identifier)
    ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE scenarios (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  owner_identifier VARCHAR(190) NOT NULL,
  scenario_name VARCHAR(190) NOT NULL,
  scenario_yaml LONGTEXT NOT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uq_scenarios_owner_name (owner_identifier, scenario_name),
  KEY ix_scenarios_owner_updated (owner_identifier, updated_at),
  CONSTRAINT fk_scenarios_owner_identifier
    FOREIGN KEY (owner_identifier) REFERENCES users(identifier)
    ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE alerts (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  owner_identifier VARCHAR(190) NOT NULL,
  level ENUM('info','warn','error') NOT NULL DEFAULT 'info',
  message TEXT NOT NULL,
  consumed TINYINT(1) NOT NULL DEFAULT 0,
  created_at DATETIME(6) NOT NULL,
  KEY ix_alerts_owner_consumed_created (owner_identifier, consumed, created_at),
  CONSTRAINT fk_alerts_owner_identifier
    FOREIGN KEY (owner_identifier) REFERENCES users(identifier)
    ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE custom_datasets (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  owner_identifier VARCHAR(190) NOT NULL,
  dataset_ref VARCHAR(190) NOT NULL,
  dataset_name VARCHAR(190) NOT NULL,
  dataset_type ENUM('custom_csv','custom_image_npz','custom_image_folder') NOT NULL,
  storage_uri TEXT NOT NULL,
  rows_count INT NULL,
  features_count INT NULL,
  classes_count INT NULL,
  created_at DATETIME(6) NOT NULL,
  UNIQUE KEY uq_custom_datasets_owner_ref (owner_identifier, dataset_ref),
  KEY ix_custom_datasets_owner_created (owner_identifier, created_at),
  CONSTRAINT fk_custom_datasets_owner_identifier
    FOREIGN KEY (owner_identifier) REFERENCES users(identifier)
    ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE app_settings (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  setting_key VARCHAR(190) NOT NULL,
  setting_json LONGTEXT NOT NULL,
  updated_by VARCHAR(190) NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uq_app_settings_key (setting_key)
) ENGINE=InnoDB;
