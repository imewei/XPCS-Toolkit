"""Batch processing module for twotime correlation data."""

import os
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from xpcs_toolkit.utils.logging_config import get_logger
from xpcs_toolkit.xpcs_file import XpcsFile

# Use non-interactive backend for batch processing
matplotlib.use("Agg")

logger = get_logger(__name__)


def parse_q_phi_pair(q_phi_str: str) -> tuple[float, float]:
    """
    Parse q-phi pair string into q and phi values.

    Args:
        q_phi_str: String in format "q,phi" (e.g., "0.05,45")

    Returns:
        Tuple of (q_value, phi_value)

    Raises:
        ValueError: If string format is invalid
    """
    try:
        parts = q_phi_str.split(",")
        num_parts = 2
        if len(parts) != num_parts:
            raise ValueError(f"Q-phi pair must be in format 'q,phi', got: {q_phi_str}")

        q_value = float(parts[0].strip())
        phi_value = float(parts[1].strip())

        return q_value, phi_value
    except ValueError as e:
        raise ValueError(f"Invalid q-phi pair format '{q_phi_str}': {e}") from e


def extract_q_phi_from_label(label: str) -> tuple[float | None, float | None]:
    """
    Extract q and phi values from qbin label string.

    Args:
        label: Qbin label string (e.g., "qbin=5, q=0.0532 Å⁻¹, φ=45.2 deg")

    Returns:
        Tuple of (q_value, phi_value) or (None, None) if not found
    """
    q_value = None
    phi_value = None

    # Extract q value - handle scientific notation and units like "Å⁻¹"
    q_match = re.search(r"q=([0-9.eE+-]+)", label)
    if q_match:
        q_value = float(q_match.group(1))

    # Extract phi value - handle both "phi=" and "φ=" (Greek phi) and units like "deg"
    phi_match = re.search(r"(?:phi|φ)=([0-9.eE+-]+)", label)
    if phi_match:
        phi_value = float(phi_match.group(1))

    return q_value, phi_value


def find_qbins_for_q(
    xfile: XpcsFile, target_q: float
) -> list[tuple[int, str, float, float]]:
    """
    Find all qbins with the closest q-value(s) across all phi angles.

    Args:
        xfile: XpcsFile instance
        target_q: Target q-value to match

    Returns:
        List of tuples (qbin_index, label, q_value, phi_value) for closest
        matching qbins
    """
    qbin_labels = xfile.get_twotime_qbin_labels()
    valid_qbins = []

    # First pass: collect all valid qbins with their q/phi values
    for i, label in enumerate(qbin_labels):
        q_value, phi_value = extract_q_phi_from_label(label)
        if q_value is not None and phi_value is not None:
            valid_qbins.append((i, label, q_value, phi_value))
        else:
            logger.warning(f"Could not extract q/phi from qbin {i}: {label}")

    if not valid_qbins:
        logger.warning("No valid qbins found for q selection")
        return []

    # Find the closest q-value(s)
    q_values = [qbin[2] for qbin in valid_qbins]
    closest_q = min(q_values, key=lambda q: abs(q - target_q))
    closest_q_diff = abs(closest_q - target_q)

    # Get all qbins with the closest q-value
    tolerance = 1e-10  # Use tight tolerance for exact match
    matching_qbins = [
        (i, label, q_val, phi_val)
        for i, label, q_val, phi_val in valid_qbins
        if abs(q_val - closest_q) < tolerance
    ]

    logger.info(
        f"Found closest q={closest_q:.6f} "
        f"(diff={closest_q_diff:.6f} from target {target_q:.6f})"
    )
    logger.info(
        f"Selected {len(matching_qbins)} qbins with q={closest_q:.6f} "
        f"across different phi angles"
    )

    # Log the phi angles found
    phi_angles = sorted([phi_val for _, _, _, phi_val in matching_qbins])
    logger.info(f"Phi angles found: {phi_angles}")

    return matching_qbins


def find_qbins_for_phi(
    xfile: XpcsFile, target_phi: float
) -> list[tuple[int, str, float, float]]:
    """
    Find all qbins with the closest phi-value(s) across all q values.

    Args:
        xfile: XpcsFile instance
        target_phi: Target phi-value to match

    Returns:
        List of tuples (qbin_index, label, q_value, phi_value) for closest
        matching qbins
    """
    qbin_labels = xfile.get_twotime_qbin_labels()
    valid_qbins = []

    # First pass: collect all valid qbins with their q/phi values
    for i, label in enumerate(qbin_labels):
        q_value, phi_value = extract_q_phi_from_label(label)
        if q_value is not None and phi_value is not None:
            valid_qbins.append((i, label, q_value, phi_value))
        else:
            logger.warning(f"Could not extract q/phi from qbin {i}: {label}")

    if not valid_qbins:
        logger.warning("No valid qbins found for phi selection")
        return []

    # Find the closest phi-value(s)
    phi_values = [qbin[3] for qbin in valid_qbins]
    closest_phi = min(phi_values, key=lambda phi: abs(phi - target_phi))
    closest_phi_diff = abs(closest_phi - target_phi)

    # Get all qbins with the closest phi-value
    phi_tolerance = 1e-10  # Use tight tolerance for exact match
    matching_qbins = [
        (i, label, q_val, phi_val)
        for i, label, q_val, phi_val in valid_qbins
        if abs(phi_val - closest_phi) < phi_tolerance
    ]

    logger.info(
        f"Found closest phi={closest_phi:.2f}° "
        f"(diff={closest_phi_diff:.2f}° from target {target_phi:.2f}°)"
    )
    logger.info(
        f"Selected {len(matching_qbins)} qbins with phi={closest_phi:.2f}° "
        f"across different q values"
    )

    # Log the q values found
    q_values = sorted([q_val for _, _, q_val, _ in matching_qbins])
    logger.info(f"Q values found: {q_values}")

    return matching_qbins


def find_qbin_for_qphi(
    xfile: XpcsFile,
    target_q: float,
    target_phi: float,
) -> tuple[int, str, float, float] | None:
    """
    Find single qbin closest to specific q-phi pair.

    Args:
        xfile: XpcsFile instance
        target_q: Target q-value to match
        target_phi: Target phi-value to match

    Returns:
        Tuple (qbin_index, label, q_value, phi_value) for closest matching qbin or None
    """
    qbin_labels = xfile.get_twotime_qbin_labels()
    valid_qbins = []

    # First pass: collect all valid qbins with their q/phi values
    for i, label in enumerate(qbin_labels):
        q_value, phi_value = extract_q_phi_from_label(label)
        if q_value is not None and phi_value is not None:
            valid_qbins.append((i, label, q_value, phi_value))
        else:
            logger.warning(f"Could not extract q/phi from qbin {i}: {label}")

    if not valid_qbins:
        logger.warning("No valid qbins found for q-phi selection")
        return None

    # Find qbin with minimum combined distance
    # Normalize distances to make them comparable
    q_values = [qbin[2] for qbin in valid_qbins]
    phi_values = [qbin[3] for qbin in valid_qbins]
    q_range = max(q_values) - min(q_values) if len(set(q_values)) > 1 else 1.0
    phi_range = max(phi_values) - min(phi_values) if len(set(phi_values)) > 1 else 1.0

    best_match = None
    best_distance = float("inf")

    for i, label, q_value, phi_value in valid_qbins:
        # Normalized euclidean distance
        q_norm = abs(q_value - target_q) / q_range
        phi_norm = abs(phi_value - target_phi) / phi_range
        distance = (q_norm**2 + phi_norm**2) ** 0.5

        if distance < best_distance:
            best_distance = distance
            best_match = (i, label, q_value, phi_value)

    if best_match:
        q_diff = abs(best_match[2] - target_q)
        phi_diff = abs(best_match[3] - target_phi)
        logger.info(
            f"Found closest qbin: q={best_match[2]:.6f} (diff={q_diff:.6f}), "
            f"phi={best_match[3]:.2f}° (diff={phi_diff:.2f}°)"
        )
        logger.info(f"Selected qbin: {best_match[1]}")
    else:
        logger.warning(f"No qbin found for q={target_q:.4f}, phi={target_phi:.1f}°")

    return best_match


def create_twotime_plot_matplotlib(
    c2_matrix: np.ndarray, delta_t: float, title: str, dpi: int = 300
) -> plt.Figure:
    """
    Create twotime correlation plot using matplotlib.

    Args:
        c2_matrix: 2D correlation matrix
        delta_t: Time step for axes scaling
        title: Plot title
        dpi: Image resolution

    Returns:
        Matplotlib Figure object
    """
    # Clean C2 data to remove NaN/inf values
    finite_mask = np.isfinite(c2_matrix)
    if np.any(finite_mask):
        finite_values = c2_matrix[finite_mask]
        if len(finite_values) > 0:
            vmin, vmax = np.percentile(finite_values, [0.5, 99.5])
            if vmin >= vmax:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0
        logger.warning("All C2 values are non-finite, using default levels")

    # Replace non-finite values
    c2_clean = np.nan_to_num(c2_matrix, nan=vmin, posinf=vmax, neginf=vmin)

    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    # Create time axes
    size = c2_clean.shape[0]
    extent = [0, size * delta_t, 0, size * delta_t]

    # Plot correlation matrix
    im = ax.imshow(
        c2_clean,
        cmap="jet",
        aspect="equal",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("C₂(t₁, t₂)", rotation=270, labelpad=20)

    # Set labels and title
    ax.set_xlabel("t₁ (s)")
    ax.set_ylabel("t₂ (s)")
    ax.set_title(title)

    # Adjust layout
    plt.tight_layout()

    return fig


def generate_output_filename(
    input_path: str, q_value: float, phi_value: float, output_format: str = "png"
) -> str:
    """
    Generate standardized output filename.

    Args:
        input_path: Input HDF file path
        q_value: Q-value
        phi_value: Phi-value
        output_format: Image format

    Returns:
        Generated filename
    """
    basename = Path(input_path).stem
    return f"{basename}_q{q_value:.4f}_phi{phi_value:.1f}.{output_format}"


def process_single_file(file_path: str, args) -> int:
    """
    Process a single HDF file and generate twotime images.

    Args:
        file_path: Path to HDF file
        args: Command line arguments

    Returns:
        Number of images generated
    """
    logger.info(f"Processing file: {file_path}")

    try:
        # Load XpcsFile
        xfile = XpcsFile(file_path)

        # Verify this is a twotime file
        if "Twotime" not in xfile.atype:
            logger.warning(
                f"File {file_path} is not a twotime file "
                f"(type: {xfile.atype}), skipping"
            )
            return 0

        # Get all available qbins for debugging and summary
        qbin_labels = xfile.get_twotime_qbin_labels()
        logger.info(f"File contains {len(qbin_labels)} total qbins")

        # Log all qbin labels for debugging
        logger.debug("Available qbins:")
        for i, label in enumerate(qbin_labels):
            q_val, phi_val = extract_q_phi_from_label(label)
            logger.debug(f"  Qbin {i}: {label} -> q={q_val}, phi={phi_val}")

        # Determine qbins to process based on selection mode
        qbins_to_process = []

        if args.q is not None:
            # Mode 1: All phi angles at specific q
            qbins_to_process = find_qbins_for_q(xfile, args.q)
        elif args.phi is not None:
            # Mode 2: All q values at specific phi
            qbins_to_process = find_qbins_for_phi(xfile, args.phi)
        elif args.q_phi is not None:
            # Mode 3: Specific q-phi pair
            q_value, phi_value = parse_q_phi_pair(args.q_phi)
            qbin_match = find_qbin_for_qphi(xfile, q_value, phi_value)
            if qbin_match:
                qbins_to_process = [qbin_match]

        if not qbins_to_process:
            logger.warning(f"No matching qbins found for file {file_path}")
            return 0

        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)

        images_generated = 0

        # Process each qbin
        for qbin_index, label, q_value, phi_value in qbins_to_process:
            logger.info(f"Processing qbin {qbin_index}: {label}")

            try:
                # Get C2 data for this qbin
                c2_result = xfile.get_twotime_c2(
                    selection=qbin_index, correct_diag=True
                )
                if c2_result is None:
                    logger.warning(f"Failed to get C2 data for qbin {qbin_index}")
                    continue

                c2_matrix = c2_result["c2_mat"]
                delta_t = c2_result["delta_t"]

                # Generate plot
                title = f"{Path(file_path).name} - {label}"
                fig = None
                try:
                    fig = create_twotime_plot_matplotlib(
                        c2_matrix, delta_t, title, args.dpi
                    )

                    # Save image
                    output_filename = generate_output_filename(
                        file_path, q_value, phi_value, args.format
                    )
                    output_path = os.path.join(args.output, output_filename)

                    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
                    logger.info(f"Saved image: {output_path}")
                    images_generated += 1
                finally:
                    if fig is not None:
                        plt.close(fig)  # Always free memory

            except Exception as e:
                logger.error(f"Error processing qbin {qbin_index}: {e}")
                continue

        logger.info(f"\n--- PROCESSING SUMMARY FOR {Path(file_path).name} ---")
        logger.info(f"Total qbins available: {len(qbin_labels)}")
        logger.info(f"Qbins selected for processing: {len(qbins_to_process)}")
        logger.info(f"Images successfully generated: {images_generated}")
        logger.info(f"Processing completed for {file_path}")
        return images_generated

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return 0


def find_hdf_files(directory: str) -> list[str]:
    """
    Find all HDF files in directory recursively.

    Args:
        directory: Directory to search

    Returns:
        List of HDF file paths
    """
    hdf_extensions = [".h5", ".hdf5", ".hdf"]
    hdf_files = []

    for root, _dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in hdf_extensions):
                hdf_files.append(os.path.join(root, file))

    return sorted(hdf_files)


def process_directory(directory: str, args) -> int:
    """
    Process all HDF files in directory.

    Args:
        directory: Directory containing HDF files
        args: Command line arguments

    Returns:
        Total number of images generated
    """
    logger.info(f"Processing directory: {directory}")

    hdf_files = find_hdf_files(directory)
    if not hdf_files:
        logger.warning(f"No HDF files found in directory: {directory}")
        return 0

    logger.info(f"Found {len(hdf_files)} HDF files to process")

    total_images = 0
    successful_files = 0

    for i, file_path in enumerate(hdf_files, 1):
        logger.info(f"Processing file {i}/{len(hdf_files)}: {file_path}")

        try:
            images_count = process_single_file(file_path, args)
            total_images += images_count
            if images_count > 0:
                successful_files += 1
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")

    logger.info(
        f"Directory processing complete: {successful_files}/{len(hdf_files)} "
        f"files processed successfully"
    )
    logger.info(f"Total images generated: {total_images}")

    return total_images


def run_twotime_batch(args) -> int:
    """
    Main entry point for twotime batch processing.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("Starting twotime batch processing")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Selection mode: q={args.q}, phi={args.phi}, q_phi={args.q_phi}")
    logger.info(f"Image settings: format={args.format}, DPI={args.dpi}")

    # Validate arguments
    selection_count = sum(x is not None for x in [args.q, args.phi, args.q_phi])
    if selection_count != 1:
        logger.error(f"Exactly one selection argument required, got {selection_count}")
        logger.error(f"Arguments: q={args.q}, phi={args.phi}, q_phi={args.q_phi}")
        return 1

    # Validate argument values
    if args.q is not None and args.q <= 0:
        logger.error(f"Q-value must be positive, got: {args.q}")
        return 1

    max_phi = 360
    if args.phi is not None and not (0 <= args.phi <= max_phi):
        logger.warning(f"Phi-value {args.phi}° is outside typical range [0, 360]°")

    if args.q_phi is not None:
        try:
            q_val, phi_val = parse_q_phi_pair(args.q_phi)
            if q_val <= 0:
                logger.error(f"Q-value in q-phi pair must be positive, got: {q_val}")
                return 1
            logger.info(f"Validated q-phi pair: q={q_val}, phi={phi_val}")
        except ValueError as e:
            logger.error(f"Invalid q-phi pair format: {e}")
            return 1

    logger.info("Argument validation passed")

    try:
        # Validate input path
        if not os.path.exists(args.input):
            logger.error(f"Input path does not exist: {args.input}")
            return 1

        # Process input (file or directory)
        if os.path.isfile(args.input):
            images_generated = process_single_file(args.input, args)
        elif os.path.isdir(args.input):
            images_generated = process_directory(args.input, args)
        else:
            logger.error(f"Input path is neither file nor directory: {args.input}")
            return 1

        if images_generated == 0:
            logger.warning("No images were generated")
            return 1

        logger.info(
            f"Batch processing completed successfully. "
            f"Generated {images_generated} images."
        )
        return 0

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        return 1
