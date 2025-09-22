"""
Application Startup Performance Optimization for XPCS Toolkit

This module optimizes application startup time by implementing lazy loading,
parallel initialization, and intelligent resource preloading strategies.
"""

import atexit
import importlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StartupMetrics:
    """Container for startup performance metrics."""

    component_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    success: bool = True
    error_message: str = ""


@dataclass
class ComponentInfo:
    """Information about a startup component."""

    name: str
    init_func: Callable
    dependencies: set[str]
    priority: int  # Lower number = higher priority
    lazy_load: bool = False
    critical: bool = True


class LazyImportManager:
    """Manages lazy importing of heavy modules to speed up startup."""

    def __init__(self):
        self._lazy_modules: dict[str, Any] = {}
        self._import_aliases: dict[str, str] = {}
        self._load_times: dict[str, float] = {}

    def register_lazy_import(self, alias: str, module_name: str):
        """
        Register a module for lazy importing.

        Parameters
        ----------
        alias : str
            Alias to use for the module
        module_name : str
            Full module name to import
        """
        self._import_aliases[alias] = module_name
        logger.debug(f"Registered lazy import: {alias} -> {module_name}")

    def get_module(self, alias: str) -> Any:
        """
        Get a module, importing it lazily if needed.

        Parameters
        ----------
        alias : str
            Module alias

        Returns
        -------
        Any
            Imported module
        """
        if alias not in self._lazy_modules:
            if alias not in self._import_aliases:
                raise ValueError(f"Unknown module alias: {alias}")

            module_name = self._import_aliases[alias]
            start_time = time.time()

            try:
                logger.debug(f"Lazy loading module: {module_name}")
                module = importlib.import_module(module_name)
                self._lazy_modules[alias] = module

                load_time = time.time() - start_time
                self._load_times[alias] = load_time

                logger.info(f"Lazy loaded {module_name} in {load_time:.3f}s")

            except Exception as e:
                logger.error(f"Failed to lazy load {module_name}: {e}")
                raise

        return self._lazy_modules[alias]

    def preload_modules(self, aliases: list[str], background: bool = True):
        """
        Preload modules in the background.

        Parameters
        ----------
        aliases : list[str]
            List of module aliases to preload
        background : bool
            Whether to load in background thread
        """
        if background:
            thread = threading.Thread(
                target=self._preload_worker,
                args=(aliases,),
                daemon=True,
                name="module_preloader",
            )
            thread.start()
        else:
            self._preload_worker(aliases)

    def _preload_worker(self, aliases: list[str]):
        """Background worker for preloading modules."""
        for alias in aliases:
            try:
                self.get_module(alias)
            except Exception as e:
                logger.warning(f"Failed to preload module {alias}: {e}")

    def get_load_statistics(self) -> dict[str, float]:
        """Get module loading statistics."""
        return self._load_times.copy()


class StartupProfiler:
    """Profiles application startup performance."""

    def __init__(self):
        self.metrics: list[StartupMetrics] = []
        self.total_startup_time = 0.0
        self.startup_start_time = None

    def start_startup_profiling(self):
        """Start overall startup profiling."""
        self.startup_start_time = time.time()
        logger.info("Starting application startup profiling")

    def end_startup_profiling(self):
        """End overall startup profiling."""
        if self.startup_start_time:
            self.total_startup_time = time.time() - self.startup_start_time
            logger.info(
                f"Total application startup time: {self.total_startup_time:.3f}s"
            )
            self._log_startup_summary()

    def profile_component(self, component_name: str):
        """
        Context manager for profiling component initialization.

        Parameters
        ----------
        component_name : str
            Name of the component being initialized
        """
        return StartupComponentProfiler(self, component_name)

    def add_metrics(self, metrics: StartupMetrics):
        """Add component metrics."""
        self.metrics.append(metrics)

    def _log_startup_summary(self):
        """Log startup performance summary."""
        if not self.metrics:
            return

        logger.info("=== Startup Performance Summary ===")

        # Sort by duration (longest first)
        sorted_metrics = sorted(self.metrics, key=lambda m: m.duration, reverse=True)

        for metric in sorted_metrics[:10]:  # Top 10 slowest
            logger.info(
                f"  {metric.component_name}: {metric.duration:.3f}s "
                f"(memory: {metric.memory_delta_mb:+.1f}MB)"
            )

        total_component_time = sum(m.duration for m in self.metrics)
        parallel_efficiency = (
            (total_component_time / self.total_startup_time)
            if self.total_startup_time > 0
            else 1.0
        )

        logger.info(f"Total component time: {total_component_time:.3f}s")
        logger.info(f"Parallel efficiency: {parallel_efficiency:.1%}")

    def export_metrics(self, filepath: str):
        """Export startup metrics to file."""
        import json

        data = {
            "total_startup_time": self.total_startup_time,
            "component_metrics": [
                {
                    "component_name": m.component_name,
                    "duration": m.duration,
                    "memory_delta_mb": m.memory_delta_mb,
                    "success": m.success,
                    "error_message": m.error_message,
                }
                for m in self.metrics
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Startup metrics exported to {filepath}")


class StartupComponentProfiler:
    """Context manager for profiling individual startup components."""

    def __init__(self, profiler: StartupProfiler, component_name: str):
        self.profiler = profiler
        self.component_name = component_name
        self.start_time = None
        self.memory_before = None

    def __enter__(self):
        self.start_time = time.time()
        process = psutil.Process()
        self.memory_before = process.memory_info().rss / (1024 * 1024)
        logger.debug(f"Starting initialization: {self.component_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        process = psutil.Process()
        memory_after = process.memory_info().rss / (1024 * 1024)

        duration = end_time - self.start_time
        memory_delta = memory_after - self.memory_before

        success = exc_type is None
        error_message = str(exc_val) if exc_val else ""

        metrics = StartupMetrics(
            component_name=self.component_name,
            start_time=self.start_time,
            end_time=end_time,
            duration=duration,
            memory_before_mb=self.memory_before,
            memory_after_mb=memory_after,
            memory_delta_mb=memory_delta,
            success=success,
            error_message=error_message,
        )

        self.profiler.add_metrics(metrics)

        if success:
            logger.debug(
                f"Completed initialization: {self.component_name} in {duration:.3f}s"
            )
        else:
            logger.error(
                f"Failed initialization: {self.component_name} - {error_message}"
            )


class ParallelStartupManager:
    """Manages parallel initialization of application components."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or min(4, (psutil.cpu_count() or 1))
        self.components: dict[str, ComponentInfo] = {}
        self.initialized_components: set[str] = set()
        self.profiler = StartupProfiler()

    def register_component(
        self,
        name: str,
        init_func: Callable,
        dependencies: set[str] | None = None,
        priority: int = 5,
        lazy_load: bool = False,
        critical: bool = True,
    ):
        """
        Register a component for initialization.

        Parameters
        ----------
        name : str
            Component name
        init_func : Callable
            Initialization function
        dependencies : set[str], optional
            Set of component names this depends on
        priority : int
            Priority level (lower = higher priority)
        lazy_load : bool
            Whether to load lazily on first use
        critical : bool
            Whether failure should stop startup
        """
        component = ComponentInfo(
            name=name,
            init_func=init_func,
            dependencies=dependencies or set(),
            priority=priority,
            lazy_load=lazy_load,
            critical=critical,
        )

        self.components[name] = component
        logger.debug(f"Registered startup component: {name}")

    def initialize_all(self) -> bool:
        """
        Initialize all registered components in optimal order.

        Returns
        -------
        bool
            True if all critical components initialized successfully
        """
        self.profiler.start_startup_profiling()

        try:
            # Separate lazy and immediate components
            immediate_components = {
                name: comp
                for name, comp in self.components.items()
                if not comp.lazy_load
            }

            lazy_components = {
                name: comp for name, comp in self.components.items() if comp.lazy_load
            }

            # Initialize immediate components in dependency order
            success = self._initialize_components(immediate_components)

            # Register lazy components for later initialization
            self._register_lazy_components(lazy_components)

            return success

        finally:
            self.profiler.end_startup_profiling()

    def _initialize_components(self, components: dict[str, ComponentInfo]) -> bool:
        """Initialize components respecting dependencies and priorities."""
        # Create dependency-sorted initialization order
        init_order = self._resolve_dependencies(components)

        # Group components by priority for parallel execution
        priority_groups = self._group_by_priority(init_order, components)

        overall_success = True

        for priority, component_names in priority_groups:
            logger.info(
                f"Initializing priority {priority} components: {component_names}"
            )

            # Initialize components in this priority group in parallel
            group_success = self._initialize_priority_group(component_names, components)

            if not group_success:
                # Check if any failed components were critical
                failed_critical = any(
                    name not in self.initialized_components
                    and components[name].critical
                    for name in component_names
                )

                if failed_critical:
                    logger.error("Critical component initialization failed")
                    overall_success = False
                    break

        return overall_success

    def _resolve_dependencies(self, components: dict[str, ComponentInfo]) -> list[str]:
        """Resolve component dependencies using topological sort."""
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []

        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return

            temp_visited.add(name)

            # Visit dependencies first
            component = components.get(name)
            if component:
                for dep in component.dependencies:
                    if dep in components:
                        visit(dep)

            temp_visited.remove(name)
            visited.add(name)
            result.append(name)

        for name in components:
            visit(name)

        return result

    def _group_by_priority(
        self, component_names: list[str], components: dict[str, ComponentInfo]
    ) -> list[tuple[int, list[str]]]:
        """Group components by priority level."""
        priority_groups = {}

        for name in component_names:
            component = components[name]
            priority = component.priority

            if priority not in priority_groups:
                priority_groups[priority] = []

            priority_groups[priority].append(name)

        # Sort by priority (lower number = higher priority)
        return sorted(priority_groups.items())

    def _initialize_priority_group(
        self, component_names: list[str], components: dict[str, ComponentInfo]
    ) -> bool:
        """Initialize a group of components in parallel."""
        if len(component_names) == 1:
            # Single component - initialize directly
            return self._initialize_single_component(component_names[0], components)

        # Multiple components - use thread pool
        with ThreadPoolExecutor(
            max_workers=min(len(component_names), self.max_workers)
        ) as executor:
            # Submit all initialization tasks
            future_to_name = {
                executor.submit(
                    self._initialize_single_component, name, components
                ): name
                for name in component_names
            }

            group_success = True

            # Wait for completion
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    component_success = future.result()
                    if not component_success and components[name].critical:
                        group_success = False
                except Exception as e:
                    logger.error(
                        f"Component {name} initialization failed with exception: {e}"
                    )
                    if components[name].critical:
                        group_success = False

            return group_success

    def _initialize_single_component(
        self, name: str, components: dict[str, ComponentInfo]
    ) -> bool:
        """Initialize a single component."""
        component = components[name]

        # Check dependencies
        if not component.dependencies.issubset(self.initialized_components):
            missing_deps = component.dependencies - self.initialized_components
            logger.warning(f"Component {name} missing dependencies: {missing_deps}")
            return False

        # Initialize with profiling
        with self.profiler.profile_component(name):
            try:
                component.init_func()
                self.initialized_components.add(name)
                logger.debug(f"Successfully initialized component: {name}")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize component {name}: {e}")
                return False

    def _register_lazy_components(self, lazy_components: dict[str, ComponentInfo]):
        """Register lazy components for on-demand initialization."""
        # In a full implementation, this would set up lazy loading mechanisms
        logger.info(f"Registered {len(lazy_components)} lazy components")

    def get_startup_metrics(self) -> StartupProfiler:
        """Get startup profiling metrics."""
        return self.profiler


class ConfigurationManager:
    """Manages application configuration for startup optimization."""

    def __init__(self):
        self.config = {
            "startup": {
                "parallel_init": True,
                "max_init_workers": 4,
                "lazy_loading": True,
                "preload_modules": True,
                "profile_startup": False,
            },
            "performance": {
                "cache_size_mb": 500,
                "thread_pool_size": 8,
                "enable_opengl": True,
                "optimize_plots": True,
            },
        }

    def load_config(self, config_path: str | None = None):
        """Load configuration from file."""
        if config_path is None:
            # Default config locations
            config_paths = [
                Path.home() / ".xpcs_toolkit" / "config.json",
                Path("config.json"),
                Path("xpcs_config.json"),
            ]

            for path in config_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path and Path(config_path).exists():
            try:
                import json

                with open(config_path) as f:
                    loaded_config = json.load(f)

                # Merge with defaults
                self._merge_config(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")

            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

    def _merge_config(self, loaded_config: dict[str, Any]):
        """Merge loaded configuration with defaults."""

        def merge_dict(target: dict[str, Any], source: dict[str, Any]):
            for key, value in source.items():
                if (
                    key in target
                    and isinstance(target[key], dict)
                    and isinstance(value, dict)
                ):
                    merge_dict(target[key], value)
                else:
                    target[key] = value

        merge_dict(self.config, loaded_config)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path."""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value


# Global instances
_lazy_import_manager = None
_startup_manager = None
_config_manager = None


def get_lazy_import_manager() -> LazyImportManager:
    """Get global lazy import manager."""
    global _lazy_import_manager
    if _lazy_import_manager is None:
        _lazy_import_manager = LazyImportManager()
    return _lazy_import_manager


def get_startup_manager() -> ParallelStartupManager:
    """Get global startup manager."""
    global _startup_manager
    if _startup_manager is None:
        _startup_manager = ParallelStartupManager()
    return _startup_manager


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
        _config_manager.load_config()
    return _config_manager


# Convenience functions
def lazy_import(alias: str, module_name: str):
    """Register a module for lazy importing."""
    get_lazy_import_manager().register_lazy_import(alias, module_name)


def get_module(alias: str):
    """Get a lazily imported module."""
    return get_lazy_import_manager().get_module(alias)


def register_startup_component(name: str, init_func: Callable, **kwargs):
    """Register a component for parallel startup initialization."""
    get_startup_manager().register_component(name, init_func, **kwargs)


def initialize_application() -> bool:
    """Initialize the entire application with optimizations."""
    return get_startup_manager().initialize_all()


def profile_startup(component_name: str):
    """Context manager for profiling startup components."""
    return get_startup_manager().profiler.profile_component(component_name)


# Register cleanup on exit
def _cleanup_startup_system():
    """Cleanup startup optimization system."""
    # This would cleanup any background threads or resources


atexit.register(_cleanup_startup_system)
