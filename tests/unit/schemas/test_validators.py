"""Unit tests for XPCS schema validators.

This module tests the schema validation functionality including:
- QMapSchema construction and validation
- G2Data schema validation
- MaskSchema validation
- PartitionSchema validation
- Edge cases: invalid units, shape mismatch

Test IDs: Schema tests for User Story 6
"""

from __future__ import annotations

import numpy as np
import pytest


class TestQMapSchemaValidConstruction:
    """Test valid QMapSchema construction."""

    def test_qmap_schema_valid_construction(self):
        """Valid QMapSchema instantiation with correct parameters."""
        from xpcsviewer.schemas.validators import QMapSchema

        # Arrange - all arrays must have same 2D shape
        size = (512, 512)
        sqmap = np.random.randint(0, 50, size=size, dtype=np.int32)
        dqmap = np.random.randint(0, 50, size=size, dtype=np.int32)
        phis = np.random.rand(*size)  # 2D array, same shape

        # Act
        schema = QMapSchema(
            sqmap=sqmap,
            dqmap=dqmap,
            phis=phis,
            sqmap_unit="nm^-1",
            dqmap_unit="nm^-1",
            phis_unit="deg",  # Must be 'deg' or 'rad'
        )

        # Assert
        assert schema is not None
        assert schema.sqmap.shape == size
        assert schema.dqmap.shape == size
        assert schema.phis.shape == size

    def test_qmap_schema_with_mask(self):
        """QMapSchema with optional mask parameter."""
        from xpcsviewer.schemas.validators import QMapSchema

        # Arrange
        size = (256, 256)
        sqmap = np.zeros(size, dtype=np.int32)
        dqmap = np.zeros(size, dtype=np.int32)
        phis = np.zeros(size)  # 2D array
        mask = np.ones(size, dtype=np.int32)

        # Act
        schema = QMapSchema(
            sqmap=sqmap,
            dqmap=dqmap,
            phis=phis,
            sqmap_unit="nm^-1",
            dqmap_unit="nm^-1",
            phis_unit="deg",
            mask=mask,
        )

        # Assert
        assert schema.mask is not None
        assert schema.mask.shape == size


class TestQMapSchemaInvalidUnits:
    """Test QMapSchema unit validation."""

    def test_qmap_schema_invalid_sqmap_unit(self):
        """Invalid sqmap unit should raise ValueError."""
        from xpcsviewer.schemas.validators import QMapSchema

        # Arrange - use correct shape first
        size = (256, 256)
        sqmap = np.zeros(size, dtype=np.int32)
        dqmap = np.zeros(size, dtype=np.int32)
        phis = np.zeros(size)

        # Act & Assert
        with pytest.raises(ValueError, match="sqmap_unit"):
            QMapSchema(
                sqmap=sqmap,
                dqmap=dqmap,
                phis=phis,
                sqmap_unit="invalid_unit",  # Invalid
                dqmap_unit="nm^-1",
                phis_unit="deg",
            )

    def test_qmap_schema_invalid_phis_unit(self):
        """Invalid phis unit should raise ValueError."""
        from xpcsviewer.schemas.validators import QMapSchema

        # Arrange
        size = (256, 256)
        sqmap = np.zeros(size, dtype=np.int32)
        dqmap = np.zeros(size, dtype=np.int32)
        phis = np.zeros(size)

        # Act & Assert
        with pytest.raises(ValueError, match="phis_unit"):
            QMapSchema(
                sqmap=sqmap,
                dqmap=dqmap,
                phis=phis,
                sqmap_unit="nm^-1",
                dqmap_unit="nm^-1",
                phis_unit="invalid_unit",  # Invalid
            )


class TestQMapSchemaShapeMismatch:
    """Test QMapSchema shape validation."""

    def test_qmap_schema_shape_mismatch(self):
        """Mismatched sqmap/dqmap shapes should raise ValueError."""
        from xpcsviewer.schemas.validators import QMapSchema

        # Arrange
        sqmap = np.zeros((256, 256), dtype=np.int32)
        dqmap = np.zeros((512, 512), dtype=np.int32)  # Different shape
        phis = np.zeros((256, 256))

        # Act & Assert
        with pytest.raises(ValueError, match="shape"):
            QMapSchema(
                sqmap=sqmap,
                dqmap=dqmap,
                phis=phis,
                sqmap_unit="nm^-1",
                dqmap_unit="nm^-1",
                phis_unit="deg",
            )

    def test_qmap_schema_mask_shape_mismatch(self):
        """Mismatched mask shape should raise ValueError."""
        from xpcsviewer.schemas.validators import QMapSchema

        # Arrange
        size = (256, 256)
        sqmap = np.zeros(size, dtype=np.int32)
        dqmap = np.zeros(size, dtype=np.int32)
        phis = np.zeros(size)
        mask = np.ones((512, 512), dtype=np.int32)  # Different shape

        # Act & Assert
        with pytest.raises(ValueError, match="[Mm]ask"):
            QMapSchema(
                sqmap=sqmap,
                dqmap=dqmap,
                phis=phis,
                sqmap_unit="nm^-1",
                dqmap_unit="nm^-1",
                phis_unit="deg",
                mask=mask,
            )


class TestG2DataSchemaValidation:
    """Test G2Data schema validation."""

    def test_g2_data_schema_validation(self):
        """Valid G2Data instantiation."""
        from xpcsviewer.schemas.validators import G2Data

        # Arrange
        n_q = 10
        n_delay = 50
        g2 = 1.0 + 0.5 * np.random.rand(n_q, n_delay)
        g2_err = 0.01 * np.ones((n_q, n_delay))
        delay_times = np.logspace(-6, 2, n_delay)
        q_values = list(np.linspace(0.001, 0.1, n_q))

        # Act
        schema = G2Data(
            g2=g2,
            g2_err=g2_err,
            delay_times=delay_times,
            q_values=q_values,
        )

        # Assert
        assert schema is not None
        assert schema.g2.shape == (n_q, n_delay)
        assert len(schema.delay_times) == n_delay

    def test_g2_data_shape_mismatch(self):
        """Mismatched g2/g2_err shapes should raise ValueError."""
        from xpcsviewer.schemas.validators import G2Data

        # Arrange
        g2 = np.ones((10, 50))
        g2_err = np.ones((5, 50))  # Different shape
        delay_times = np.logspace(-6, 2, 50)
        q_values = list(np.linspace(0.001, 0.1, 10))

        # Act & Assert
        with pytest.raises(ValueError, match="shape"):
            G2Data(
                g2=g2,
                g2_err=g2_err,
                delay_times=delay_times,
                q_values=q_values,
            )


class TestMaskSchemaValidation:
    """Test MaskSchema schema validation."""

    def test_mask_schema_validation(self):
        """Valid MaskSchema instantiation."""
        from xpcsviewer.schemas.validators import GeometryMetadata, MaskSchema

        # Arrange
        size = (256, 256)
        mask = np.random.randint(0, 2, size=size, dtype=np.int32)

        geometry = GeometryMetadata(
            bcx=128.0,
            bcy=128.0,
            det_dist=5.0,
            lambda_=1.54e-10,
            pix_dim=75e-6,
            shape=size,
        )

        # Act
        schema = MaskSchema(
            mask=mask,
            metadata=geometry,
        )

        # Assert
        assert schema is not None
        assert schema.mask.shape == size

    def test_mask_schema_invalid_values(self):
        """Mask with values other than 0/1 should raise ValueError."""
        from xpcsviewer.schemas.validators import GeometryMetadata, MaskSchema

        # Arrange
        size = (256, 256)
        mask = np.full(size, 5, dtype=np.int32)  # Invalid values

        geometry = GeometryMetadata(
            bcx=128.0,
            bcy=128.0,
            det_dist=5.0,
            lambda_=1.54e-10,
            pix_dim=75e-6,
            shape=size,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="[Mm]ask"):
            MaskSchema(
                mask=mask,
                metadata=geometry,
            )


class TestPartitionSchemaValidation:
    """Test PartitionSchema schema validation."""

    def test_partition_schema_validation(self):
        """Valid PartitionSchema instantiation."""
        from xpcsviewer.schemas.validators import GeometryMetadata, PartitionSchema

        # Arrange
        size = (256, 256)
        n_pts = 10
        partition_map = np.random.randint(0, n_pts, size=size, dtype=np.int32)
        val_list = list(np.linspace(0.001, 0.1, n_pts))
        num_list = list(np.random.randint(100, 1000, n_pts))

        geometry = GeometryMetadata(
            bcx=128.0,
            bcy=128.0,
            det_dist=5.0,
            lambda_=1.54e-10,
            pix_dim=75e-6,
            shape=size,
        )

        # Act
        schema = PartitionSchema(
            partition_map=partition_map,
            num_pts=n_pts,
            val_list=val_list,
            num_list=num_list,
            metadata=geometry,
        )

        # Assert
        assert schema is not None
        assert schema.num_pts == n_pts

    def test_partition_schema_num_pts_mismatch(self):
        """Mismatched num_pts and val_list length should raise ValueError."""
        from xpcsviewer.schemas.validators import GeometryMetadata, PartitionSchema

        # Arrange
        size = (256, 256)
        partition_map = np.zeros(size, dtype=np.int32)
        val_list = list(np.linspace(0.001, 0.1, 5))  # Length 5
        num_list = [0] * 10  # Length 10 - mismatch

        geometry = GeometryMetadata(
            bcx=128.0,
            bcy=128.0,
            det_dist=5.0,
            lambda_=1.54e-10,
            pix_dim=75e-6,
            shape=size,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="num_pts|val_list|num_list"):
            PartitionSchema(
                partition_map=partition_map,
                num_pts=10,  # Says 10 but val_list has 5
                val_list=val_list,
                num_list=num_list,
                metadata=geometry,
            )


class TestSchemaConversions:
    """Test schema to_dict and from_dict conversions."""

    def test_qmap_schema_to_dict(self):
        """QMapSchema to_dict returns dictionary."""
        from xpcsviewer.schemas.validators import QMapSchema

        # Arrange
        size = (256, 256)
        sqmap = np.zeros(size, dtype=np.int32)
        dqmap = np.zeros(size, dtype=np.int32)
        phis = np.zeros(size)

        schema = QMapSchema(
            sqmap=sqmap,
            dqmap=dqmap,
            phis=phis,
            sqmap_unit="nm^-1",
            dqmap_unit="nm^-1",
            phis_unit="deg",
        )

        # Act
        result = schema.to_dict()

        # Assert
        assert isinstance(result, dict)
        assert "sqmap" in result
        assert "dqmap" in result

    def test_g2_data_to_dict(self):
        """G2Data to_dict returns dictionary."""
        from xpcsviewer.schemas.validators import G2Data

        # Arrange
        g2 = np.ones((10, 50))
        g2_err = np.ones((10, 50))
        delay_times = np.logspace(-6, 2, 50)
        q_values = list(np.linspace(0.001, 0.1, 10))

        schema = G2Data(
            g2=g2,
            g2_err=g2_err,
            delay_times=delay_times,
            q_values=q_values,
        )

        # Act
        result = schema.to_dict()

        # Assert
        assert isinstance(result, dict)
        assert "g2" in result
        assert "delay_times" in result
