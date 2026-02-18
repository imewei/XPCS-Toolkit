Two-Time Correlation Analysis
==============================

This page explains the concept and interpretation of two-time correlation
analysis in XPCS, which extends the standard :math:`g_2(\tau)` analysis to
detect non-equilibrium and aging dynamics.

Motivation
-----------

The standard G2 autocorrelation function (see :doc:`g2_theory`) assumes
**time-translation invariance** -- that the sample dynamics are stationary,
meaning the same at all measurement times. This assumption breaks down for:

- **Aging systems**: Glasses, gels, and other out-of-equilibrium materials
  where dynamics slow down over time
- **Phase transitions**: Systems undergoing structural or ordering
  transformations during the measurement
- **Beam-induced dynamics**: X-ray damage that progressively changes
  sample behavior
- **Intermittent dynamics**: Sudden, discrete rearrangement events
  (avalanches, earthquakes in colloidal crystals)

The two-time correlation function captures these non-stationary effects by
retaining the explicit dependence on both time points.

Definition
-----------

The two-time correlation function is defined as:

.. math::

   C(t_1, t_2) = \frac{\langle I(\mathbf{q}, t_1) \cdot I(\mathbf{q}, t_2) \rangle_{\mathbf{q}}}
                      {\langle I(\mathbf{q}, t_1) \rangle_{\mathbf{q}} \cdot \langle I(\mathbf{q}, t_2) \rangle_{\mathbf{q}}}

where:

- :math:`I(\mathbf{q}, t)` is the scattered intensity at wavevector
  :math:`\mathbf{q}` and time :math:`t`
- :math:`\langle \cdot \rangle_{\mathbf{q}}` denotes averaging over pixels
  within a Q-bin
- :math:`t_1` and :math:`t_2` are two frame times (not a time and a lag)

The result is a symmetric matrix :math:`C(t_1, t_2) = C(t_2, t_1)` of
dimension :math:`N_{\text{frames}} \times N_{\text{frames}}`.

Interpretation
---------------

Relationship to G2
^^^^^^^^^^^^^^^^^^^

For a **stationary** system, :math:`C(t_1, t_2)` depends only on the
time difference :math:`\tau = |t_1 - t_2|`, and averaging along the
diagonals recovers the standard :math:`g_2(\tau)`:

.. math::

   g_2(\tau) = \frac{1}{N - k} \sum_{i=1}^{N-k} C(t_i, t_i + \tau)

where :math:`k = \tau / \Delta t` is the lag index and :math:`N` is the
number of frames.

Non-Stationary Features
^^^^^^^^^^^^^^^^^^^^^^^^

Deviations from uniform diagonal structure reveal dynamics that change
over the measurement:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pattern
     - Physical Interpretation
   * - Uniform diagonals
     - Stationary dynamics (standard G2 analysis is valid)
   * - Diagonals broaden with time
     - **Aging**: dynamics slow down (e.g., glass formation)
   * - Diagonals narrow with time
     - **Speeding up**: dynamics accelerate (e.g., melting)
   * - Sharp off-diagonal boundaries
     - **Discrete events**: sudden rearrangements at specific times
   * - Stripe patterns
     - **Intermittent dynamics**: alternating fast and slow periods
   * - Bright corners / dark center
     - **Beam damage**: sample changes due to X-ray exposure

Extracting :math:`g_2(\tau, t_{\text{age}})` from Cuts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For aging systems, one can extract a time-dependent correlation function
by taking cuts along diagonals at different mean ages
:math:`t_{\text{age}} = (t_1 + t_2) / 2`:

.. math::

   g_2(\tau, t_{\text{age}}) = C\!\left(t_{\text{age}} - \frac{\tau}{2},\;
                                         t_{\text{age}} + \frac{\tau}{2}\right)

Fitting these cuts with the single exponential model at each age gives
:math:`\tau_c(t_{\text{age}})`, revealing how the relaxation time evolves.

Visualization in XPCS Viewer
------------------------------

XPCS Viewer provides two visualization modes for two-time data:

**Two-Time Correlation Map**

The full :math:`C(t_1, t_2)` matrix is displayed as a color-coded image.
The visualization applies:

- Outlier clipping to handle spurious extreme values
- Percentile-based color scaling for robust contrast
- Symmetric colormap centered on the diagonal

**Diagonal G2 Extraction**

Averaging along diagonals of the two-time matrix produces the standard
:math:`g_2(\tau)` curve. This is useful for comparing with the directly
computed G2 data and for verifying stationarity.

Computational Considerations
-----------------------------

Two-time correlation matrices are large: an experiment with :math:`N` frames
produces an :math:`N \times N` matrix per Q-bin. For 10,000 frames, this is
:math:`10^8` elements (800 MB in float64), making two-time analysis the most
memory-intensive operation in XPCS Viewer.

The implementation addresses this through:

- **Lazy loading**: Two-time data is read from HDF5 only when the user
  navigates to the two-time analysis tab.
- **Slicing**: Only the requested Q-bin's data is loaded, not the full
  multi-Q dataset.
- **Backend acceleration**: When the JAX backend is active, element-wise
  operations on the correlation matrix benefit from JIT compilation.
- **Memory monitoring**: The ``MemoryMonitor`` tracks system memory
  usage and warns before allocating matrices that would exceed available
  RAM.
