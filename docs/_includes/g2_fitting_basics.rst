The G2 autocorrelation for a single relaxation process is:

.. math::

   g_2(\tau) = \text{baseline} + \text{contrast} \cdot \exp(-2\tau / \tau_c)

where :math:`\tau_c` is the characteristic relaxation time, baseline is
typically near 1.0, and contrast is the Siegert factor.

Parameter bounds for single-exponential fitting:

.. list-table::
   :widths: 20 15 15 30
   :header-rows: 1

   * - Parameter
     - Lower
     - Upper
     - Description
   * - baseline
     - 0.9
     - 1.1
     - Ergodic intercept (Siegert relation)
   * - tau
     - 1e-6
     - 1e3
     - Relaxation time (seconds)
   * - contrast
     - 0.0
     - 0.5
     - Speckle contrast
   * - beta
     - 0.5
     - 1.5
     - Stretching exponent (1.0 = simple diffusion)
