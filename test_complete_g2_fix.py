#!/usr/bin/env python3
"""
Test script to verify the complete G2 plotting workflow.
"""

import os
import sys
from xpcs_toolkit.file_locator import FileLocator, create_xpcs_dataset
from xpcs_toolkit.utils.logging_config import get_logger

# Setup logging
logger = get_logger(__name__)

def test_complete_g2_workflow():
    """Test the complete G2 plotting workflow including viewer filtering."""
    print("Testing complete G2 workflow...")

    # Find an actual XPCS file to use for testing
    data_path = "/Users/b80985/Projects/data/A0198"
    hdf_files = [f for f in os.listdir(data_path) if f.endswith('.hdf')]
    if not hdf_files:
        print("No HDF files found in test directory")
        return False

    test_file = hdf_files[0]
    print(f"Using test file: {test_file}")

    # Create file locator and load file
    locator = FileLocator(data_path)
    full_path = os.path.join(data_path, test_file)
    try:
        xf_obj = create_xpcs_dataset(full_path)
        if xf_obj is None:
            print("Failed to load test file")
            return False
        # Add to cache to simulate the viewer workflow
        locator.cache[full_path] = xf_obj
    except Exception as e:
        print(f"Error loading test file: {e}")
        return False

    print(f"File loaded successfully. Analysis type: {xf_obj.atype}")

    # Test 1: Data extraction works
    print("\n=== Test 1: G2 Data Extraction ===")
    try:
        q, tel, g2, g2_err, labels = xf_obj.get_g2_data()
        print(f"✅ Data extraction successful:")
        print(f"   Q shape: {q.shape}, G2 shape: {g2.shape}")
    except Exception as e:
        print(f"❌ Data extraction failed: {e}")
        return False

    # Test 2: File filtering for G2 plotting (simulating viewer logic)
    print("\n=== Test 2: G2 File Filtering ===")
    try:
        # Get all files
        xf_list = locator.get_xf_list()
        print(f"Total files: {len(xf_list)}")

        # Filter for G2-compatible files (simulating viewer logic)
        g2_compatible_files = []
        for xf in xf_list:
            if any(atype in xf.atype for atype in ["Multitau", "Twotime"]):
                g2_compatible_files.append(xf)
                print(f"✅ G2-compatible file found: {xf.atype}")

        if g2_compatible_files:
            print(f"✅ Filtering successful: {len(g2_compatible_files)} G2-compatible files")
        else:
            print("❌ No G2-compatible files found")
            return False

    except Exception as e:
        print(f"❌ File filtering failed: {e}")
        return False

    # Test 3: Complete workflow simulation
    print("\n=== Test 3: Complete Workflow ===")
    try:
        # Simulate the complete plotting workflow
        file_to_plot = g2_compatible_files[0]
        q, tel, g2, g2_err, labels = file_to_plot.get_g2_data()

        # Basic validation of data quality
        import numpy as np
        if np.any(np.isfinite(g2)) and np.any(np.isfinite(tel)) and len(labels) > 0:
            print("✅ Complete workflow successful - ready for plotting!")
            print(f"   Data quality: G2 finite values: {np.sum(np.isfinite(g2))}/{g2.size}")
            print(f"   Time points: {len(tel)}, Q bins: {len(labels)}")
            return True
        else:
            print("❌ Data quality issues detected")
            return False

    except Exception as e:
        print(f"❌ Complete workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_complete_g2_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)