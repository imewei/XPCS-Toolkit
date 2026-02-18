Mask and Q-Map Internals
=========================

This page explains how XPCS Viewer computes Q-maps from detector geometry
and how the mask system manages pixel validity through an undo/redo stack.

From Pixels to Q-Space
------------------------

Each pixel on a 2D area detector corresponds to a specific scattering vector
:math:`\mathbf{q}` determined by the experimental geometry. The **Q-map**
is the mapping from pixel coordinates :math:`(i, j)` to the scattering
wavevector magnitude :math:`q` and azimuthal angle :math:`\phi`.

Computing this mapping is the foundation of all XPCS analysis: it determines
which pixels belong to which Q-bin for correlation function computation.

Geometry Parameters
^^^^^^^^^^^^^^^^^^^^

The Q-map computation requires six geometry parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Unit
     - Description
   * - ``energy``
     - keV
     - X-ray energy, converted to wavevector via :math:`k_0 = 2\pi / \lambda`
   * - ``bcx``, ``bcy``
     - pixels
     - Beam center position (column, row) on the detector
   * - ``det_dist``
     - mm
     - Sample-to-detector distance
   * - ``pix_dim``
     - mm
     - Physical size of one detector pixel
   * - ``shape``
     - (H, W)
     - Detector dimensions in pixels

Defaults (used when geometry is not available in the HDF5 file):
``pix_dim=0.075``, ``energy=10.0``, ``det_dist=5000.0``, beam center at
the image center.

Transmission Geometry
^^^^^^^^^^^^^^^^^^^^^^

For small-angle X-ray scattering (SAXS) in transmission, the Q-map
computation proceeds as follows:

1. **Pixel grid**: Create arrays ``v`` and ``h`` of pixel distances from
   the beam center, then form a 2D meshgrid.

2. **Radial distance**: :math:`r = \sqrt{v^2 + h^2} \times \text{pix\_dim}`
   gives the physical distance from the beam center in mm.

3. **Scattering angle**: :math:`\alpha = \arctan(r / d)` where :math:`d` is
   the detector distance.

4. **Q magnitude**: The scattering wavevector magnitude is:

   .. math::

      q = k_0 \sin(\alpha) = \frac{2\pi}{\lambda} \sin\!\left[\arctan\!\left(\frac{r}{d}\right)\right]

   where the wavelength is computed from energy:
   :math:`\lambda [\text{\AA}] = 12.398 / E [\text{keV}]`.

   For small angles (SAXS regime), this simplifies to:

   .. math::

      q \approx \frac{4\pi}{\lambda} \sin\!\left(\frac{\theta}{2}\right)

   where :math:`2\theta = \alpha` is the full scattering angle.

5. **Azimuthal angle**: :math:`\phi = -\arctan2(v, h)` gives the azimuthal
   angle in the detector plane (negated by convention).

6. **Q components**: :math:`q_x = q \cos\phi`, :math:`q_y = q \sin\phi`.

The output is a dictionary containing arrays ``sqmap`` (Q magnitude),
``phis`` (azimuthal angle), ``qr``, ``qx``, ``qy``, and angular maps.

Reflection Geometry
^^^^^^^^^^^^^^^^^^^^

For grazing-incidence SAXS (GI-SAXS), the geometry is more complex because
the incident beam hits the sample at a shallow angle :math:`\alpha_i`.
The Q-map must account for refraction effects and the tilted scattering
geometry. The reflection Q-map additionally requires:

- ``alpha_i_deg``: Incident angle in degrees (default 0.14)
- ``orientation``: Detector orientation relative to the sample surface

JIT Compilation of Q-Map
^^^^^^^^^^^^^^^^^^^^^^^^^^

Q-map computation is one of the most performance-critical operations because
it runs on every geometry change (beam center adjustment, detector distance
update, etc.) during interactive mask editing.

When the JAX backend is active, the core computation is JIT-compiled:

.. code-block:: text

   compute_qmap(stype, metadata)
       |
       v
   compute_transmission_qmap(energy, beam_center, shape, pix_dim, det_dist)
       |
       v
   _compute_transmission_qmap_cached(...)   <-- dict-based cache check
       |
       v (cache miss)
   _get_transmission_qmap_jit()             <-- creates @jax.jit function
       |
       v
   _transmission_qmap_core(k0, v, h, ...)   <-- JIT-compiled, runs on XLA

The dict-based caching (``_JIT_CACHE``) is necessary because JAX arrays
are not hashable, so ``functools.lru_cache`` cannot be used. The first call
with a new scattering type creates and caches the compiled function;
subsequent calls with different numerical inputs reuse the compiled trace
(XLA recompiles only if array shapes change).

Typical performance: ~200 ms first call (compilation), ~5 ms subsequent calls
for 1024x1024 detector.

Q-Bin Partition
----------------

After computing the Q-map, pixels are grouped into **Q-bins** for correlation
analysis. The partition assigns each pixel to a bin based on its Q value.

The partition generation (``simplemask/utils.py``) creates:

- ``partition_map``: An integer array of shape ``(H, W)`` where each value is
  the Q-bin index (0 = unassigned/masked)
- ``val_list``: Q-bin center values
- ``num_list``: Number of pixels per bin
- ``num_pts``: Total number of Q-bins

Binning can be linear or logarithmic in Q-space. Logarithmic binning is
preferred for SAXS data because Q resolution varies across the detector
(finer at small Q, coarser at large Q).

Mask System Architecture
-------------------------

The mask is a boolean array (stored as int32 with values 0/1) that marks
which detector pixels are valid for analysis. Masked pixels (value 0) are
excluded from correlation computation.

Common reasons for masking:

- Dead or hot pixels on the detector
- Beamstop shadow
- Gaps between detector modules
- Regions with parasitic scattering
- User-defined exclusion zones

MaskAssemble and the History Stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``MaskAssemble`` class manages the composite mask through a layered
system with undo/redo capability.

**Mask layers** (applied in order):

1. **Default masks**: Loaded from file or computed from detector properties
   (dead pixels, gaps)
2. **Threshold mask**: Pixels above/below intensity thresholds
3. **Parameter masks**: Based on Q-range or angle constraints
4. **Drawing masks**: User-drawn regions (rectangles, circles, polygons,
   lines, ellipses)

Each mask layer implements a ``MaskBase`` subclass with an ``evaluate()``
method that returns a boolean array. The final mask is the logical AND of
all layers.

**Undo/redo mechanism**:

.. code-block:: text

   mask_record:  [state_0, state_1, state_2, state_3]
                                       ^
   mask_ptr:                           |
                           mask_ptr = 2

   Undo: mask_ptr -= 1, apply state_1
   Redo: mask_ptr += 1, apply state_3

The history stack (``mask_record``) stores mask states at each modification
point. The pointer (``mask_ptr``) tracks the current position.
``mask_ptr_min`` marks the boundary of the default mask -- undo cannot go
below this point (the default mask is always present).

Operations:

- ``mask_action(key, arr)``: Pushes a new mask state, truncating any
  redo history beyond the current pointer
- ``redo_undo(step)``: Moves the pointer by ``step`` (+1 for redo, -1
  for undo), then applies the state at the new position
- ``mask_apply(key)``: Syncs the kernel's mask with the current state

Drawing Tools
^^^^^^^^^^^^^^

XPCS Viewer provides six drawing tools for interactive mask editing:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Tool
     - Description
   * - Rectangle
     - Axis-aligned rectangular region
   * - Circle
     - Circular region defined by center and radius
   * - Ellipse
     - Elliptical region with adjustable axes
   * - Polygon
     - Arbitrary polygon defined by vertex clicks
   * - Line
     - Linear selection (custom ``LineROI`` from ``pyqtgraph_mod.py``)
   * - Eraser
     - Removes masked regions (restores pixels)

Drawing is performed through PyQtGraph ROI (Region of Interest) objects.
When the user completes a drawing operation:

1. ``apply_drawing()`` converts the ROI to a boolean mask array
2. The mask is inverted (``~mask``) and passed to
   ``evaluate("mask_draw", arr=~mask)``
3. ``mask_apply("mask_draw")`` records the state in the history stack

Signal Export
^^^^^^^^^^^^^^

When the user exports the mask and partition from SimpleMask to the main
XPCS Viewer, two Qt signals carry the data:

- ``mask_exported(np.ndarray)``: The final mask array (int32, 0/1)
- ``qmap_exported(dict)``: The partition data (partition_map, val_list,
  num_list, num_pts)

This signal-based communication maintains loose coupling between the
SimpleMask subsystem and the main viewer -- neither module imports or
directly calls the other. The main viewer connects to these signals and
applies the received data to update its analysis state.
