import numpy as np

from xpcs_toolkit.plothandler.plot_constants import BASIC_COLORS as colors
from xpcs_toolkit.plothandler.plot_constants import EXTENDED_MARKERS as shapes
from xpcs_toolkit.utils.logging_config import get_logger
from xpcs_toolkit.utils.validation import (
    get_file_label_safe,
    validate_array_compatibility,
    validate_xf_fit_summary,
)

logger = get_logger(__name__)


def plot(xf_list, hdl, q_range, offset, plot_type=3):
    """Optimized tau-q plotting with vectorized operations and pre-computation"""
    logger.info(f"Starting tau-q plot for {len(xf_list)} files")
    logger.debug(
        f"Plot parameters: q_range={q_range}, offset={offset}, plot_type={plot_type}"
    )

    hdl.clear()
    ax = hdl.subplots(1, 1)

    # Pre-compute scale factors and colors/shapes to avoid repeated calculations
    num_files = len(xf_list)
    scale_factors = np.array([10 ** (offset * n) for n in range(num_files)])
    color_cycle = [colors[n % len(colors)] for n in range(num_files)]
    shape_cycle = [shapes[n % len(shapes)] for n in range(num_files)]

    plot_count = 0

    for n, xf in enumerate(xf_list):
        # Extract data with error checking using validation utility
        is_valid, fit_summary, _error_msg = validate_xf_fit_summary(xf)
        if not is_valid:
            continue

        s = scale_factors[n]
        x = fit_summary["q_val"]
        # Ensure x is a numpy array (convert from list if necessary)
        if not hasattr(x, "shape"):
            x = np.array(x)
        y = fit_summary["fit_val"][:, 0, 1]
        e = fit_summary["fit_val"][:, 1, 1]

        # Ensure all arrays have the same length to avoid boolean index mismatch
        file_label = get_file_label_safe(xf)
        is_compatible, min_length, _warning_msg = validate_array_compatibility(
            x, y, e, file_label=file_label
        )
        if not is_compatible:
            continue

        # Trim arrays to minimum length if needed
        x = x[:min_length]
        y = y[:min_length]
        e = e[:min_length]

        # Vectorized filtering of failed fittings
        valid_idx = e > 0
        if not np.any(valid_idx):
            logger.debug(
                f"Skipping file {getattr(xf, 'label', 'unknown')} - no valid data points"
            )
            continue

        x_valid = x[valid_idx]
        y_valid = y[valid_idx]
        e_valid = e[valid_idx]

        # Pre-computed color and shape
        color = color_cycle[n]
        shape = shape_cycle[n]

        # Plot with pre-computed scaled values
        y_scaled = y_valid / s
        e_scaled = e_valid / s

        ax.errorbar(
            x_valid,
            y_scaled,
            yerr=e_scaled,
            fmt=shape,
            markersize=3,
            label=xf.label,
            color=color,
            mfc="white",
        )
        plot_count += 1

        # Plot fit line if available
        if fit_summary.get("tauq_success", False):
            tauq_fit_line = fit_summary.get("tauq_fit_line")
            if (
                isinstance(tauq_fit_line, dict)
                and "fit_x" in tauq_fit_line
                and "fit_y" in tauq_fit_line
            ):
                fit_x = tauq_fit_line["fit_x"]
                fit_y = tauq_fit_line["fit_y"]
                ax.plot(fit_x, fit_y / s, color=color)

    # Set labels and scales
    ax.set_xlabel("$q (\\AA^{-1})$")
    ax.set_ylabel("$\\tau (s)$")
    ax.legend()

    # Use pre-computed scale options
    scale_options = ["linear", "log"]
    xscale = scale_options[plot_type % 2]
    yscale = scale_options[plot_type // 2]
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    hdl.draw()
    logger.info(f"Tau-q plot completed successfully - {plot_count} plots drawn")


def plot_pre(xf_list, hdl):
    """Optimized plot_pre with vectorized operations and reduced redundancy"""
    logger.info(f"Starting tau-q pre-plot for {len(xf_list)} files")

    hdl.clear()

    # Handle empty file list case
    if len(xf_list) == 0:
        logger.warning(
            "No files provided for tau-q pre-plot - showing instruction message"
        )
        ax = hdl.subplots(1, 1)
        ax.text(
            0.5,
            0.5,
            'No files with G2 fitting results available.\n\nTo see diffusion analysis:\n1. Go to "g2" tab\n2. Select files and click "plot" to fit G2\n3. Return to "Diffusion" tab',
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        hdl.fig.tight_layout()
        hdl.draw()
        logger.info("Displayed instruction message for missing G2 fitting results")
        return

    logger.info(f"Creating 2x2 subplot layout for {len(xf_list)} files")
    ax = hdl.subplots(2, 2, sharex=True).flatten()
    titles = ["contrast", "tau (s)", "stretch", "baseline"]

    # Pre-compute colors and shapes for all files
    color_cycle = [colors[idx % len(colors)] for idx in range(len(xf_list))]
    shape_cycle = [shapes[idx % len(shapes)] for idx in range(len(xf_list))]

    # Track bounds for final setup
    global_bounds = None
    x_range = None
    processed_files = 0

    for idx, xf in enumerate(xf_list):
        try:
            # Validate file using validation utility
            is_valid, fit_summary, _error_msg = validate_xf_fit_summary(xf)
            if not is_valid:
                continue

            color = color_cycle[idx]
            shape = shape_cycle[idx]

            x = fit_summary["q_val"]
            fit_val = fit_summary["fit_val"]

            # Ensure x is a numpy array (convert from list if necessary)
            if not hasattr(x, "shape"):
                x = np.array(x)

            # Validate array dimensions before plotting
            if len(x.shape) > 1:
                x = x.flatten()

            # Ensure x and fit_val have compatible shapes
            min_length = min(len(x), fit_val.shape[0])
            x = x[:min_length]
            fit_val = fit_val[:min_length]

            # Plot all parameters at once with vectorized operations
            for n in range(4):
                y = fit_val[:, 0, n]
                e = fit_val[:, 1, n]

                # Additional safety check for array lengths
                if len(x) != len(y):
                    logger.warning(
                        f"Size mismatch: x={len(x)}, y={len(y)}. Truncating to shorter length."
                    )
                    min_len = min(len(x), len(y))
                    x_plot = x[:min_len]
                    y_plot = y[:min_len]
                    e_plot = e[:min_len]
                else:
                    x_plot = x
                    y_plot = y
                    e_plot = e

                # Skip plotting if arrays are empty
                if len(x_plot) == 0 or len(y_plot) == 0:
                    logger.warning(
                        f"Empty arrays for file {getattr(xf, 'label', 'unknown')}, parameter {n}"
                    )
                    continue

                ax[n].errorbar(
                    x_plot,
                    y_plot,
                    yerr=e_plot,
                    fmt=shape,
                    markersize=3,
                    color=color,
                    mfc="white",
                )

            # Store bounds and x range from the last valid file
            if "bounds" in fit_summary:
                global_bounds = fit_summary["bounds"]
                x_range = (np.min(x), np.max(x))

            processed_files += 1

        except Exception as e:
            logger.error(f"Error plotting file {getattr(xf, 'label', 'unknown')}: {e}")
            continue

    # Set up axes after all data is plotted
    if global_bounds is not None and x_range is not None:
        xmin, xmax = x_range

        for n in range(4):
            ymin = global_bounds[0][n]
            ymax = global_bounds[1][n]

            # Set titles and scales
            ax[n].set_title(titles[n])

            if n == 1:  # tau parameter gets log scale
                ax[n].set_yscale("log")

            # Set y limits with buffer
            ax[n].set_ylim(ymin * 0.8, ymax * 1.2)

            # Set x labels for bottom row
            if n > 1:
                ax[n].set_xlabel("$q (\\AA^{-1})$")

            # Add bound lines
            ax[n].hlines(ymin, xmin, xmax, color="b", label="lower bound")
            ax[n].hlines(ymax, xmin, xmax, color="g", label="upper bound")

            # Show legend only in the last plot
            if n == 3:
                ax[n].legend()

    logger.info(f"Pre-plot final setup: processed {processed_files} files successfully")
    logger.info(f"Global bounds available: {global_bounds is not None}")

    hdl.draw()
    logger.info(
        f"Tau-q pre-plot completed successfully - processed {processed_files}/{len(xf_list)} files"
    )
    return
