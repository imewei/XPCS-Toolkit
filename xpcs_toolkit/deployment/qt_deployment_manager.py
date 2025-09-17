"""
Qt Compliance Deployment and Maintenance Manager.

This module provides comprehensive deployment, configuration management,
and maintenance capabilities for the Qt compliance system across different
environments and deployment scenarios.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PySide6.QtCore import QObject, Signal, QTimer

from ..monitoring import get_integrated_monitoring_system, initialize_integrated_monitoring
from ..performance import get_qt_optimization_engine, OptimizationConfiguration
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CI_CD = "ci_cd"


class DeploymentMode(Enum):
    """Deployment modes."""

    STANDALONE = "standalone"
    INTEGRATED = "integrated"
    PLUGIN = "plugin"
    LIBRARY = "library"


class MaintenanceOperation(Enum):
    """Maintenance operation types."""

    UPDATE = "update"
    HEALTH_CHECK = "health_check"
    OPTIMIZATION = "optimization"
    BACKUP = "backup"
    CLEANUP = "cleanup"
    VALIDATION = "validation"


@dataclass
class DeploymentConfiguration:
    """Configuration for Qt compliance deployment."""

    # Environment settings
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    deployment_mode: DeploymentMode = DeploymentMode.INTEGRATED

    # Installation paths
    install_directory: Optional[str] = None
    config_directory: Optional[str] = None
    log_directory: Optional[str] = None
    cache_directory: Optional[str] = None

    # Feature configuration
    enable_monitoring: bool = True
    enable_optimization: bool = True
    enable_performance_tracking: bool = True
    enable_auto_updates: bool = False

    # Environment-specific settings
    log_level: str = "INFO"
    max_log_files: int = 10
    log_rotation_size_mb: int = 100

    # Performance settings
    optimization_level: str = "balanced"
    cache_size_mb: int = 50
    monitoring_interval_seconds: int = 300

    # Maintenance settings
    enable_auto_maintenance: bool = True
    maintenance_window_hours: List[int] = field(default_factory=lambda: [2, 3, 4])  # 2-4 AM
    backup_retention_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def for_environment(cls, environment: DeploymentEnvironment) -> 'DeploymentConfiguration':
        """Create configuration for specific environment."""
        config = cls(environment=environment)

        if environment == DeploymentEnvironment.DEVELOPMENT:
            config.log_level = "DEBUG"
            config.enable_auto_updates = False
            config.enable_auto_maintenance = False
            config.optimization_level = "conservative"

        elif environment == DeploymentEnvironment.TESTING:
            config.log_level = "DEBUG"
            config.enable_performance_tracking = True
            config.enable_auto_updates = False
            config.optimization_level = "conservative"

        elif environment == DeploymentEnvironment.STAGING:
            config.log_level = "INFO"
            config.enable_performance_tracking = True
            config.enable_auto_updates = True
            config.optimization_level = "balanced"

        elif environment == DeploymentEnvironment.PRODUCTION:
            config.log_level = "WARNING"
            config.enable_performance_tracking = True
            config.enable_auto_updates = False  # Manual updates in production
            config.optimization_level = "aggressive"
            config.monitoring_interval_seconds = 60  # More frequent monitoring

        elif environment == DeploymentEnvironment.CI_CD:
            config.log_level = "INFO"
            config.enable_monitoring = False  # Minimal overhead for CI
            config.enable_auto_maintenance = False
            config.optimization_level = "conservative"

        return config


@dataclass
class DeploymentStatus:
    """Status of Qt compliance deployment."""

    timestamp: float
    environment: DeploymentEnvironment
    version: str
    installation_path: str
    status: str  # "installed", "running", "error", "maintenance"

    # Component status
    monitoring_active: bool = False
    optimization_active: bool = False
    performance_tracking_active: bool = False

    # Health metrics
    uptime_seconds: float = 0.0
    last_health_check: Optional[float] = None
    error_count: int = 0
    warning_count: int = 0

    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MaintenanceRecord:
    """Record of maintenance operation."""

    timestamp: float
    operation: MaintenanceOperation
    success: bool
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class QtDeploymentManager(QObject):
    """
    Comprehensive Qt compliance deployment and maintenance manager.

    Provides:
    - Multi-environment deployment support
    - Configuration management
    - Health monitoring and maintenance
    - Automated updates and optimization
    - Backup and recovery
    - Environment validation
    """

    # Signals
    deployment_started = Signal(str)  # environment
    deployment_completed = Signal(str, bool)  # environment, success
    maintenance_started = Signal(str)  # operation
    maintenance_completed = Signal(str, bool)  # operation, success
    health_check_completed = Signal(object)  # DeploymentStatus

    def __init__(self, config: Optional[DeploymentConfiguration] = None, parent: QObject = None):
        """Initialize deployment manager."""
        super().__init__(parent)

        self.config = config or DeploymentConfiguration()

        # Setup deployment paths
        self._setup_deployment_paths()

        # Current deployment status
        self._deployment_status = DeploymentStatus(
            timestamp=time.perf_counter(),
            environment=self.config.environment,
            version=self._get_version(),
            installation_path=str(self.install_path),
            status="initializing"
        )

        # Maintenance tracking
        self._maintenance_history: List[MaintenanceRecord] = []
        self._maintenance_timer: Optional[QTimer] = None

        # Component references
        self._monitoring_system = None
        self._optimization_engine = None

        logger.info(f"Qt deployment manager initialized for {self.config.environment.value} environment")

    def _setup_deployment_paths(self):
        """Setup deployment directory structure."""
        # Determine base paths
        if self.config.install_directory:
            self.install_path = Path(self.config.install_directory)
        else:
            self.install_path = self._get_default_install_path()

        if self.config.config_directory:
            self.config_path = Path(self.config.config_directory)
        else:
            self.config_path = self.install_path / "config"

        if self.config.log_directory:
            self.log_path = Path(self.config.log_directory)
        else:
            self.log_path = self.install_path / "logs"

        if self.config.cache_directory:
            self.cache_path = Path(self.config.cache_directory)
        else:
            self.cache_path = self.install_path / "cache"

        # Create directories
        for path in [self.install_path, self.config_path, self.log_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Deployment paths configured: {self.install_path}")

    def _get_default_install_path(self) -> Path:
        """Get default installation path based on environment."""
        base_name = "xpcs_qt_compliance"

        if self.config.environment == DeploymentEnvironment.DEVELOPMENT:
            return Path.home() / ".local" / "share" / f"{base_name}_dev"
        elif self.config.environment == DeploymentEnvironment.TESTING:
            return Path.home() / ".local" / "share" / f"{base_name}_test"
        elif self.config.environment == DeploymentEnvironment.STAGING:
            return Path("/opt") / f"{base_name}_staging" if platform.system() != "Windows" else Path.home() / "AppData" / "Local" / f"{base_name}_staging"
        elif self.config.environment == DeploymentEnvironment.PRODUCTION:
            return Path("/opt") / base_name if platform.system() != "Windows" else Path.home() / "AppData" / "Local" / base_name
        else:  # CI_CD
            return Path(tempfile.gettempdir()) / f"{base_name}_ci"

    def _get_version(self) -> str:
        """Get current Qt compliance system version."""
        try:
            # Try to get version from package
            import pkg_resources
            return pkg_resources.get_distribution("xpcs-toolkit").version
        except:
            # Fallback version
            return "2.0.0"

    def deploy(self) -> bool:
        """Deploy Qt compliance system."""
        logger.info(f"Starting deployment to {self.config.environment.value} environment")
        self.deployment_started.emit(self.config.environment.value)

        try:
            # Update deployment status
            self._deployment_status.status = "deploying"

            # Step 1: Validate environment
            if not self._validate_deployment_environment():
                raise Exception("Environment validation failed")

            # Step 2: Install configuration
            self._install_configuration()

            # Step 3: Setup logging
            self._setup_logging()

            # Step 4: Initialize components
            self._initialize_components()

            # Step 5: Setup maintenance
            if self.config.enable_auto_maintenance:
                self._setup_maintenance_scheduling()

            # Step 6: Perform initial health check
            self._perform_health_check()

            # Update status
            self._deployment_status.status = "running"
            self._deployment_status.timestamp = time.perf_counter()

            logger.info("Qt compliance deployment completed successfully")
            self.deployment_completed.emit(self.config.environment.value, True)
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            logger.debug(traceback.format_exc())
            self._deployment_status.status = "error"
            self.deployment_completed.emit(self.config.environment.value, False)
            return False

    def _validate_deployment_environment(self) -> bool:
        """Validate deployment environment."""
        logger.info("Validating deployment environment")

        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False

        # Check Qt availability
        try:
            from PySide6.QtCore import QCoreApplication
            logger.debug("PySide6 available")
        except ImportError:
            logger.error("PySide6 not available")
            return False

        # Check write permissions
        if not os.access(self.install_path.parent, os.W_OK):
            logger.error(f"No write permission to {self.install_path.parent}")
            return False

        # Check disk space (require at least 100MB)
        if shutil.disk_usage(self.install_path.parent).free < 100 * 1024 * 1024:
            logger.error("Insufficient disk space")
            return False

        # Environment-specific validations
        if self.config.environment == DeploymentEnvironment.PRODUCTION:
            # Production requires more stringent checks
            try:
                import psutil
                if psutil.virtual_memory().available < 512 * 1024 * 1024:  # 512MB
                    logger.warning("Low available memory for production deployment")
            except ImportError:
                logger.warning("psutil not available - cannot check memory")

        logger.info("Environment validation passed")
        return True

    def _install_configuration(self):
        """Install configuration files."""
        logger.info("Installing configuration")

        # Create main configuration file
        config_file = self.config_path / "qt_compliance.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)

        # Create environment-specific configurations
        self._create_environment_configs()

        # Create monitoring configuration
        if self.config.enable_monitoring:
            self._create_monitoring_config()

        # Create optimization configuration
        if self.config.enable_optimization:
            self._create_optimization_config()

        logger.debug(f"Configuration installed to {self.config_path}")

    def _create_environment_configs(self):
        """Create environment-specific configuration files."""
        env_config = {
            "environment": self.config.environment.value,
            "deployment_mode": self.config.deployment_mode.value,
            "paths": {
                "install": str(self.install_path),
                "config": str(self.config_path),
                "logs": str(self.log_path),
                "cache": str(self.cache_path)
            },
            "features": {
                "monitoring": self.config.enable_monitoring,
                "optimization": self.config.enable_optimization,
                "performance_tracking": self.config.enable_performance_tracking,
                "auto_updates": self.config.enable_auto_updates
            }
        }

        env_file = self.config_path / f"environment_{self.config.environment.value}.json"
        with open(env_file, 'w') as f:
            json.dump(env_config, f, indent=2)

    def _create_monitoring_config(self):
        """Create monitoring configuration."""
        from ..monitoring import MonitoringConfiguration

        monitoring_config = MonitoringConfiguration()
        monitoring_config.enable_automatic_reports = True
        monitoring_config.report_interval_minutes = self.config.monitoring_interval_seconds // 60
        monitoring_config.report_output_directory = str(self.log_path / "monitoring")

        config_file = self.config_path / "monitoring.json"
        with open(config_file, 'w') as f:
            json.dump(monitoring_config.__dict__, f, indent=2, default=str)

    def _create_optimization_config(self):
        """Create optimization configuration."""
        from ..performance import OptimizationLevel

        optimization_config = OptimizationConfiguration()

        # Map string to enum
        level_map = {
            "conservative": OptimizationLevel.CONSERVATIVE,
            "balanced": OptimizationLevel.BALANCED,
            "aggressive": OptimizationLevel.AGGRESSIVE,
            "maximum": OptimizationLevel.MAXIMUM
        }

        optimization_config.optimization_level = level_map.get(
            self.config.optimization_level,
            OptimizationLevel.BALANCED
        )

        optimization_config.cache_size_limit = self.config.cache_size_mb * 10  # Approximate conversion

        config_file = self.config_path / "optimization.json"
        with open(config_file, 'w') as f:
            json.dump(optimization_config.to_dict(), f, indent=2, default=str)

    def _setup_logging(self):
        """Setup logging configuration."""
        import logging
        from logging.handlers import RotatingFileHandler

        # Create log file
        log_file = self.log_path / "qt_compliance.log"

        # Configure rotating file handler
        handler = RotatingFileHandler(
            log_file,
            maxBytes=self.config.log_rotation_size_mb * 1024 * 1024,
            backupCount=self.config.max_log_files
        )

        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Set log level
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        handler.setLevel(level)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        logger.info(f"Logging configured: {log_file}")

    def _initialize_components(self):
        """Initialize Qt compliance components."""
        logger.info("Initializing Qt compliance components")

        # Initialize monitoring system
        if self.config.enable_monitoring:
            try:
                self._monitoring_system = initialize_integrated_monitoring()
                self._deployment_status.monitoring_active = True
                logger.debug("Monitoring system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize monitoring: {e}")

        # Initialize optimization engine
        if self.config.enable_optimization:
            try:
                self._optimization_engine = get_qt_optimization_engine()
                self._deployment_status.optimization_active = True
                logger.debug("Optimization engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize optimization: {e}")

        logger.info("Component initialization completed")

    def _setup_maintenance_scheduling(self):
        """Setup automated maintenance scheduling."""
        if not self.config.enable_auto_maintenance:
            return

        # Setup maintenance timer
        self._maintenance_timer = QTimer(self)
        self._maintenance_timer.timeout.connect(self._scheduled_maintenance)

        # Calculate interval (check every hour, run maintenance during window)
        self._maintenance_timer.start(3600000)  # 1 hour in milliseconds

        logger.info(f"Maintenance scheduling enabled: window {self.config.maintenance_window_hours}")

    def _scheduled_maintenance(self):
        """Perform scheduled maintenance if in maintenance window."""
        current_hour = datetime.now().hour

        if current_hour in self.config.maintenance_window_hours:
            logger.info("Performing scheduled maintenance")
            self.perform_maintenance_operation(MaintenanceOperation.HEALTH_CHECK)

            # Perform optimization every other day
            if datetime.now().day % 2 == 0:
                self.perform_maintenance_operation(MaintenanceOperation.OPTIMIZATION)

            # Perform cleanup weekly
            if datetime.now().weekday() == 6:  # Sunday
                self.perform_maintenance_operation(MaintenanceOperation.CLEANUP)

    def perform_maintenance_operation(self, operation: MaintenanceOperation) -> bool:
        """Perform specific maintenance operation."""
        logger.info(f"Starting maintenance operation: {operation.value}")
        self.maintenance_started.emit(operation.value)

        start_time = time.perf_counter()
        success = False
        details = {}
        error_message = None

        try:
            if operation == MaintenanceOperation.HEALTH_CHECK:
                success = self._perform_health_check()
                details = self._deployment_status.to_dict()

            elif operation == MaintenanceOperation.OPTIMIZATION:
                success = self._perform_optimization()

            elif operation == MaintenanceOperation.CLEANUP:
                success = self._perform_cleanup()
                details = {"cleaned_files": 0, "freed_space_mb": 0}

            elif operation == MaintenanceOperation.BACKUP:
                success = self._perform_backup()

            elif operation == MaintenanceOperation.VALIDATION:
                success = self._perform_validation()

            else:
                raise ValueError(f"Unknown maintenance operation: {operation}")

        except Exception as e:
            error_message = str(e)
            logger.error(f"Maintenance operation {operation.value} failed: {e}")

        duration = time.perf_counter() - start_time

        # Record maintenance operation
        record = MaintenanceRecord(
            timestamp=time.perf_counter(),
            operation=operation,
            success=success,
            duration_seconds=duration,
            details=details,
            error_message=error_message
        )

        self._maintenance_history.append(record)

        # Keep only recent maintenance records
        if len(self._maintenance_history) > 1000:
            self._maintenance_history = self._maintenance_history[-500:]

        self.maintenance_completed.emit(operation.value, success)
        logger.info(f"Maintenance operation {operation.value} completed: {'success' if success else 'failed'}")

        return success

    def _perform_health_check(self) -> bool:
        """Perform comprehensive health check."""
        try:
            # Update basic status
            self._deployment_status.timestamp = time.perf_counter()
            self._deployment_status.last_health_check = time.perf_counter()

            # Check resource usage
            try:
                import psutil
                process = psutil.Process()
                self._deployment_status.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self._deployment_status.cpu_usage_percent = process.cpu_percent()
            except ImportError:
                pass

            # Check disk usage
            self._deployment_status.disk_usage_mb = self._calculate_disk_usage()

            # Check component health
            if self._monitoring_system:
                try:
                    status = self._monitoring_system.get_system_status()
                    self._deployment_status.monitoring_active = status.qt_error_detector_active
                except Exception as e:
                    logger.warning(f"Monitoring health check failed: {e}")
                    self._deployment_status.monitoring_active = False

            if self._optimization_engine:
                try:
                    stats = self._optimization_engine.get_optimization_stats()
                    self._deployment_status.optimization_active = True
                except Exception as e:
                    logger.warning(f"Optimization health check failed: {e}")
                    self._deployment_status.optimization_active = False

            # Emit health check signal
            self.health_check_completed.emit(self._deployment_status)

            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def _perform_optimization(self) -> bool:
        """Perform system optimization."""
        if not self._optimization_engine:
            logger.warning("Optimization engine not available")
            return False

        try:
            results = self._optimization_engine.apply_comprehensive_optimization()
            successful_optimizations = sum(1 for r in results if r.success)

            logger.info(f"Optimization completed: {successful_optimizations}/{len(results)} successful")
            return successful_optimizations > 0

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False

    def _perform_cleanup(self) -> bool:
        """Perform system cleanup."""
        try:
            cleaned_files = 0
            freed_space = 0

            # Clean old log files
            for log_file in self.log_path.glob("*.log.*"):
                if (time.time() - log_file.stat().st_mtime) > (30 * 24 * 3600):  # 30 days
                    file_size = log_file.stat().st_size
                    log_file.unlink()
                    cleaned_files += 1
                    freed_space += file_size

            # Clean cache files
            for cache_file in self.cache_path.glob("**/*"):
                if cache_file.is_file() and (time.time() - cache_file.stat().st_mtime) > (7 * 24 * 3600):  # 7 days
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    cleaned_files += 1
                    freed_space += file_size

            # Clean temporary files
            temp_dir = self.install_path / "temp"
            if temp_dir.exists():
                for temp_file in temp_dir.glob("**/*"):
                    if temp_file.is_file():
                        file_size = temp_file.stat().st_size
                        temp_file.unlink()
                        cleaned_files += 1
                        freed_space += file_size

            freed_space_mb = freed_space / 1024 / 1024
            logger.info(f"Cleanup completed: {cleaned_files} files, {freed_space_mb:.1f}MB freed")

            return True

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    def _perform_backup(self) -> bool:
        """Perform configuration backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.install_path / "backups" / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Backup configuration
            if self.config_path.exists():
                shutil.copytree(self.config_path, backup_dir / "config")

            # Backup maintenance history
            history_file = backup_dir / "maintenance_history.json"
            with open(history_file, 'w') as f:
                json.dump([record.to_dict() for record in self._maintenance_history], f, indent=2, default=str)

            # Clean old backups
            self._cleanup_old_backups()

            logger.info(f"Backup completed: {backup_dir}")
            return True

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def _cleanup_old_backups(self):
        """Cleanup old backup files."""
        backup_base = self.install_path / "backups"
        if not backup_base.exists():
            return

        cutoff_time = time.time() - (self.config.backup_retention_days * 24 * 3600)

        for backup_dir in backup_base.iterdir():
            if backup_dir.is_dir() and backup_dir.stat().st_mtime < cutoff_time:
                shutil.rmtree(backup_dir)
                logger.debug(f"Removed old backup: {backup_dir}")

    def _perform_validation(self) -> bool:
        """Perform system validation."""
        try:
            # Validate configuration
            config_valid = self._validate_configuration()

            # Validate component functionality
            components_valid = self._validate_components()

            # Validate environment consistency
            environment_valid = self._validate_deployment_environment()

            validation_success = config_valid and components_valid and environment_valid

            logger.info(f"Validation completed: {'passed' if validation_success else 'failed'}")
            return validation_success

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate configuration files."""
        try:
            # Check main config
            config_file = self.config_path / "qt_compliance.json"
            if not config_file.exists():
                logger.error("Main configuration file missing")
                return False

            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Validate required fields
            required_fields = ['environment', 'deployment_mode']
            for field in required_fields:
                if field not in config_data:
                    logger.error(f"Required configuration field missing: {field}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def _validate_components(self) -> bool:
        """Validate component functionality."""
        try:
            validation_success = True

            # Validate monitoring system
            if self.config.enable_monitoring and self._monitoring_system:
                try:
                    status = self._monitoring_system.get_system_status()
                    if not status.qt_error_detector_active:
                        logger.warning("Qt error detector not active")
                        validation_success = False
                except Exception as e:
                    logger.error(f"Monitoring validation failed: {e}")
                    validation_success = False

            # Validate optimization engine
            if self.config.enable_optimization and self._optimization_engine:
                try:
                    stats = self._optimization_engine.get_optimization_stats()
                    if not stats:
                        logger.warning("Optimization engine not responding")
                        validation_success = False
                except Exception as e:
                    logger.error(f"Optimization validation failed: {e}")
                    validation_success = False

            return validation_success

        except Exception as e:
            logger.error(f"Component validation failed: {e}")
            return False

    def _calculate_disk_usage(self) -> float:
        """Calculate disk usage in MB."""
        try:
            total_size = 0
            for path in [self.install_path, self.config_path, self.log_path, self.cache_path]:
                if path.exists():
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size

            return total_size / 1024 / 1024

        except Exception as e:
            logger.warning(f"Could not calculate disk usage: {e}")
            return 0.0

    def get_deployment_status(self) -> DeploymentStatus:
        """Get current deployment status."""
        # Update uptime
        self._deployment_status.uptime_seconds = time.perf_counter() - self._deployment_status.timestamp
        return self._deployment_status

    def get_maintenance_history(self, hours: Optional[int] = None) -> List[MaintenanceRecord]:
        """Get maintenance history."""
        if hours is None:
            return list(self._maintenance_history)

        cutoff_time = time.perf_counter() - (hours * 3600)
        return [
            record for record in self._maintenance_history
            if record.timestamp > cutoff_time
        ]

    def update_configuration(self, new_config: DeploymentConfiguration) -> bool:
        """Update deployment configuration."""
        try:
            # Backup current configuration
            self.perform_maintenance_operation(MaintenanceOperation.BACKUP)

            # Update configuration
            self.config = new_config

            # Reinstall configuration files
            self._install_configuration()

            # Reinitialize components if needed
            if new_config.enable_monitoring and not self._monitoring_system:
                self._monitoring_system = initialize_integrated_monitoring()
                self._deployment_status.monitoring_active = True

            if new_config.enable_optimization and not self._optimization_engine:
                self._optimization_engine = get_qt_optimization_engine()
                self._deployment_status.optimization_active = True

            logger.info("Configuration updated successfully")
            return True

        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False

    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        status = self.get_deployment_status()
        recent_maintenance = self.get_maintenance_history(hours=24)

        return {
            "deployment_info": {
                "environment": self.config.environment.value,
                "version": status.version,
                "installation_path": status.installation_path,
                "deployment_time": status.timestamp,
                "uptime_hours": status.uptime_seconds / 3600
            },
            "current_status": status.to_dict(),
            "configuration": self.config.to_dict(),
            "maintenance_summary": {
                "total_operations": len(self._maintenance_history),
                "recent_operations_24h": len(recent_maintenance),
                "success_rate": sum(1 for r in recent_maintenance if r.success) / len(recent_maintenance) if recent_maintenance else 1.0,
                "last_health_check": status.last_health_check
            },
            "component_health": {
                "monitoring": status.monitoring_active,
                "optimization": status.optimization_active,
                "performance_tracking": status.performance_tracking_active
            },
            "resource_usage": {
                "memory_mb": status.memory_usage_mb,
                "cpu_percent": status.cpu_usage_percent,
                "disk_mb": status.disk_usage_mb
            },
            "recommendations": self._generate_deployment_recommendations(status)
        }

    def _generate_deployment_recommendations(self, status: DeploymentStatus) -> List[str]:
        """Generate deployment recommendations."""
        recommendations = []

        # Resource usage recommendations
        if status.memory_usage_mb > 500:
            recommendations.append("High memory usage - consider enabling memory optimization")

        if status.cpu_usage_percent > 80:
            recommendations.append("High CPU usage - review system load and optimization settings")

        if status.disk_usage_mb > 1000:  # 1GB
            recommendations.append("High disk usage - perform cleanup maintenance")

        # Component health recommendations
        if self.config.enable_monitoring and not status.monitoring_active:
            recommendations.append("Monitoring system not active - check monitoring configuration")

        if self.config.enable_optimization and not status.optimization_active:
            recommendations.append("Optimization engine not active - verify optimization settings")

        # Maintenance recommendations
        if status.last_health_check is None or (time.perf_counter() - status.last_health_check) > 86400:  # 24 hours
            recommendations.append("Health check overdue - perform maintenance")

        # Environment-specific recommendations
        if self.config.environment == DeploymentEnvironment.PRODUCTION:
            if self.config.log_level == "DEBUG":
                recommendations.append("Debug logging enabled in production - consider changing to WARNING level")

            if not self.config.enable_auto_maintenance:
                recommendations.append("Auto-maintenance disabled in production - enable for better reliability")

        return recommendations


# Global instance
_qt_deployment_manager: Optional[QtDeploymentManager] = None


def get_qt_deployment_manager(config: Optional[DeploymentConfiguration] = None) -> QtDeploymentManager:
    """Get the global Qt deployment manager instance."""
    global _qt_deployment_manager

    if _qt_deployment_manager is None:
        _qt_deployment_manager = QtDeploymentManager(config)

    return _qt_deployment_manager


def deploy_qt_compliance_system(environment: DeploymentEnvironment,
                               deployment_mode: DeploymentMode = DeploymentMode.INTEGRATED) -> bool:
    """Deploy Qt compliance system to specified environment."""
    config = DeploymentConfiguration.for_environment(environment)
    config.deployment_mode = deployment_mode

    manager = get_qt_deployment_manager(config)
    return manager.deploy()


if __name__ == "__main__":
    # Example deployment
    import sys

    if len(sys.argv) > 1:
        env_name = sys.argv[1].upper()
        try:
            environment = DeploymentEnvironment[env_name]
            success = deploy_qt_compliance_system(environment)
            print(f"Deployment to {env_name}: {'SUCCESS' if success else 'FAILED'}")
            sys.exit(0 if success else 1)
        except KeyError:
            print(f"Unknown environment: {env_name}")
            print("Available environments: DEVELOPMENT, TESTING, STAGING, PRODUCTION, CI_CD")
            sys.exit(1)
    else:
        print("Usage: python qt_deployment_manager.py <ENVIRONMENT>")
        sys.exit(1)