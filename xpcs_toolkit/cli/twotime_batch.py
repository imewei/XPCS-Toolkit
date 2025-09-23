"""Batch processing module for twotime correlation data."""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from xpcs_toolkit.utils.logging_config import get_logger
from xpcs_toolkit.xpcs_file import XpcsFile

# Use non-interactive backend for batch processing
matplotlib.use("Agg")

logger = get_logger(__name__)


def parse_q_phi_pair(q_phi_str: str) -> Tuple[float, float]:
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
        if len(parts) != 2:
            raise ValueError(f"Q-phi pair must be in format 'q,phi', got: {q_phi_str}")

        q_value = float(parts[0].strip())
        phi_value = float(parts[1].strip())

        return q_value, phi_value
    except ValueError as e:
        raise ValueError(f"Invalid q-phi pair format '{q_phi_str}': {e}")


def extract_q_phi_from_label(label: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract q and phi values from qbin label string.

    Args:
        label: Qbin label string (e.g., "qbin=5, q=0.0532, phi=45.2")

    Returns:
        Tuple of (q_value, phi_value) or (None, None) if not found
    """
    q_value = None
    phi_value = None

    # Extract q value
    q_match = re.search(r"q=([0-9.]+)", label)
    if q_match:
        q_value = float(q_match.group(1))

    # Extract phi value
    phi_match = re.search(r"phi=([0-9.]+)", label)
    if phi_match:
        phi_value = float(phi_match.group(1))

    return q_value, phi_value


def find_qbins_for_q(
    xfile: XpcsFile, target_q: float, tolerance: float = 0.01
) -> List[Tuple[int, str, float, float]]:
    """
    Find all qbins matching a specific q-value across all phi angles.

    Args:
        xfile: XpcsFile instance
        target_q: Target q-value to match
        tolerance: Tolerance for q-value matching

    Returns:
        List of tuples (qbin_index, label, q_value, phi_value) for matching qbins
    """
    qbin_labels = xfile.get_twotime_qbin_labels()
    matching_qbins = []

    for i, label in enumerate(qbin_labels):
        q_value, phi_value = extract_q_phi_from_label(label)

        if q_value is not None and abs(q_value - target_q) <= tolerance:
            matching_qbins.append((i, label, q_value, phi_value))

    logger.info(
        f"Found {len(matching_qbins)} qbins matching q={target_q:.4f} (tolerance={tolerance})"
    )
    return matching_qbins


def find_qbins_for_phi(
    xfile: XpcsFile, target_phi: float, tolerance: float = 1.0
) -> List[Tuple[int, str, float, float]]:
    """
    Find all qbins matching a specific phi-value across all q values.

    Args:
        xfile: XpcsFile instance
        target_phi: Target phi-value to match
        tolerance: Tolerance for phi-value matching (degrees)

    Returns:
        List of tuples (qbin_index, label, q_value, phi_value) for matching qbins
    """
    qbin_labels = xfile.get_twotime_qbin_labels()
    matching_qbins = []

    for i, label in enumerate(qbin_labels):
        q_value, phi_value = extract_q_phi_from_label(label)

        if phi_value is not None and abs(phi_value - target_phi) <= tolerance:
            matching_qbins.append((i, label, q_value, phi_value))

    logger.info(
        f"Found {len(matching_qbins)} qbins matching phi={target_phi:.1f}° (tolerance={tolerance}°)"
    )
    return matching_qbins


def find_qbin_for_qphi(
    xfile: XpcsFile,
    target_q: float,
    target_phi: float,
    q_tolerance: float = 0.01,
    phi_tolerance: float = 1.0,
) -> Optional[Tuple[int, str, float, float]]:
    """
    Find single qbin matching specific q-phi pair.

    Args:
        xfile: XpcsFile instance
        target_q: Target q-value to match
        target_phi: Target phi-value to match
        q_tolerance: Tolerance for q-value matching
        phi_tolerance: Tolerance for phi-value matching (degrees)

    Returns:
        Tuple (qbin_index, label, q_value, phi_value) for best matching qbin or None
    """
    qbin_labels = xfile.get_twotime_qbin_labels()
    best_match = None
    best_distance = float("inf")

    for i, label in enumerate(qbin_labels):
        q_value, phi_value = extract_q_phi_from_label(label)

        if q_value is not None and phi_value is not None:
            q_diff = abs(q_value - target_q)
            phi_diff = abs(phi_value - target_phi)

            if q_diff <= q_tolerance and phi_diff <= phi_tolerance:
                # Calculate combined distance for best match
                distance = (q_diff / q_tolerance) + (phi_diff / phi_tolerance)
                if distance < best_distance:
                    best_distance = distance
                    best_match = (i, label, q_value, phi_value)

    if best_match:
        logger.info(
            f"Found qbin matching q={target_q:.4f}, phi={target_phi:.1f}°: {best_match[1]}"
        )
    else:
        logger.warning(
            f"No qbin found matching q={target_q:.4f}, phi={target_phi:.1f}°"
        )

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
                f"File {file_path} is not a twotime file (type: {xfile.atype}), skipping"
            )
            return 0

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
                fig = create_twotime_plot_matplotlib(
                    c2_matrix, delta_t, title, args.dpi
                )

                # Save image
                output_filename = generate_output_filename(
                    file_path, q_value, phi_value, args.format
                )
                output_path = os.path.join(args.output, output_filename)

                fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)  # Free memory

                logger.info(f"Saved image: {output_path}")
                images_generated += 1

            except Exception as e:
                logger.error(f"Error processing qbin {qbin_index}: {e}")
                continue

        logger.info(f"Generated {images_generated} images from {file_path}")
        return images_generated

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return 0


def find_hdf_files(directory: str) -> List[str]:
    """
    Find all HDF files in directory recursively.

    Args:
        directory: Directory to search

    Returns:
        List of HDF file paths
    """
    hdf_extensions = [".h5", ".hdf5", ".hdf"]
    hdf_files = []

    for root, dirs, files in os.walk(directory):
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
        f"Directory processing complete: {successful_files}/{len(hdf_files)} files processed successfully"
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
    logger.info(f"Selection mode: q={args.q}, phi={args.phi}, q-phi={args.q_phi}")
    logger.info(f"Image settings: format={args.format}, DPI={args.dpi}")

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
            f"Batch processing completed successfully. Generated {images_generated} images."
        )
        return 0

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        return 1
