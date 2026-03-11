"""JAX MLX Plugin - MLX backend for JAX."""

import os
import sys
import warnings
from pathlib import Path

# jaxlib version this plugin was built against (major.minor). Used for runtime
# compatibility checking.
_BUILT_FOR_JAXLIB = (0, 9)
_LIB_NAME = "libpjrt_plugin_mlx.dylib"


class MLXPluginError(Exception):
    """Exception raised when MLX plugin initialization fails."""

    pass


def _get_search_paths():
    """Return list of (path, description) tuples for library search."""
    pkg_dir = Path(__file__).parent
    project_root = pkg_dir.parent.parent.parent

    return [
        (pkg_dir / _LIB_NAME, "package directory (editable install)"),
        (pkg_dir / "lib" / _LIB_NAME, "package lib/ (wheel install)"),
        (
            project_root / "build" / "*" / "lib" / _LIB_NAME,
            "build/*/lib/ (cmake build)",
        ),
        (Path("/usr/local/lib") / _LIB_NAME, "/usr/local/lib/"),
        (Path("/opt/homebrew/lib") / _LIB_NAME, "/opt/homebrew/lib/"),
    ]


def _find_library():
    """Find the pjrt_plugin_mlx shared library.

    Returns:
        Path to the library, or None if not found.
    """
    # Environment variable takes precedence
    if "JAX_MLX_LIBRARY_PATH" in os.environ:
        env_path = os.environ["JAX_MLX_LIBRARY_PATH"]
        if Path(env_path).exists():
            return env_path
        raise MLXPluginError(
            f"JAX_MLX_LIBRARY_PATH is set to '{env_path}', but the file does not exist."
        )

    for path, _ in _get_search_paths():
        # Handle glob patterns
        if "*" in str(path):
            for match in Path("/").glob(str(path).lstrip("/")):
                if match.exists():
                    return str(match)
        elif path.exists():
            return str(path)

    return None


def _check_jaxlib_version():
    """Check if the installed jaxlib version is compatible.

    Warns if the major.minor version doesn't match what the plugin was built for.
    """
    try:
        import jaxlib

        version_str = getattr(jaxlib, "__version__", None)
        if version_str is None:
            return

        parts = version_str.split(".")
        if len(parts) < 2:
            return

        installed = (int(parts[0]), int(parts[1]))
        if installed != _BUILT_FOR_JAXLIB:
            warnings.warn(
                f"jax-mlx was built for jaxlib {_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1]}.x, "
                f"but jaxlib {version_str} is installed. This may cause compatibility "
                f"issues with StableHLO bytecode parsing. Consider installing jaxlib "
                f">={_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1]}.0,"
                f"<{_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1] + 1}",
                stacklevel=3,
            )
    except Exception:
        pass  # Don't fail initialization due to version check issues


def initialize():
    """Initialize the MLX plugin with JAX.

    This function is called by JAX's plugin discovery mechanism.

    Raises:
        MLXPluginError: If Metal GPU is not available or plugin initialization fails.
    """
    # Check platform first
    if sys.platform != "darwin":
        raise MLXPluginError(
            f"MLX plugin requires macOS, but running on {sys.platform}. "
            "Apple MLX is only available on Apple devices."
        )

    # Check jaxlib version compatibility
    _check_jaxlib_version()

    library_path = _find_library()
    if library_path is None:
        searched = "\n".join(f"  - {desc}" for _, desc in _get_search_paths())
        raise MLXPluginError(
            f"Could not find {_LIB_NAME}. Searched paths:\n{searched}\n"
            "You can also set JAX_MLX_LIBRARY_PATH environment variable."
        )

    # Disable shardy partitioner - it produces sdy dialect ops that our StableHLO parser
    # doesn't support yet (JAX 0.9+ enables it by default)
    try:
        import jax

        jax.config.update("jax_use_shardy_partitioner", False)
    except Exception as e:
        warnings.warn(
            f"Failed to disable shardy partitioner: {e}. Some operations may not work correctly.",
            stacklevel=2,
        )

    # Preload libmlx.dylib into the global linker namespace so that our
    # PJRT plugin (which links against it) can dlopen successfully regardless
    # of where libmlx.dylib lives on this system.
    try:
        import ctypes
        import importlib.util

        mlx_spec = importlib.util.find_spec("mlx")
        if mlx_spec and mlx_spec.submodule_search_locations:
            mlx_pkg_dir = Path(mlx_spec.submodule_search_locations[0])
            mlx_dylib = mlx_pkg_dir / "lib" / "libmlx.dylib"
            if mlx_dylib.exists():
                ctypes.CDLL(str(mlx_dylib), ctypes.RTLD_GLOBAL)
    except Exception:
        pass  # If preloading fails, dlopen may still succeed via RPATH

    # Register the plugin using JAX's xla_bridge API
    try:
        from jax._src import xla_bridge as xb
    except ImportError as e:
        raise MLXPluginError(f"Failed to import JAX xla_bridge: {e}") from e

    if not hasattr(xb, "register_plugin"):
        raise MLXPluginError("JAX version does not support register_plugin API.")

    try:
        xb.register_plugin(
            "mlx",
            priority=1,  # Above CPU (0) so MLX is the default backend when available
            library_path=library_path,
            options=None,
        )
    except Exception as e:
        # Handle "already registered" case - this is fine, not an error
        if "ALREADY_EXISTS" in str(e) and "mlx" in str(e).lower():
            return
        raise MLXPluginError(f"Failed to register MLX plugin with JAX: {e}") from e
