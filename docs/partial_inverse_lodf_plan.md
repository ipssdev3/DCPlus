# Partial-Inverse LODF Implementation Plan

This document proposes an implementation path for scaling DC+ contingency
analysis to large systems where materializing the full dense Jacobian inverse is
too expensive.

The current implementation stores:

```text
Jinv = J^-1
```

as a dense `n_eq x n_eq` array. That gives excellent JAX/GPU throughput, but
for 25k to 100k+ bus systems the memory cost can dominate. The proposed design
keeps the mathematical low-rank Woodbury solve, but replaces the full inverse
with partial inverse panels:

```text
X_G = J^-1[rows_needed, K_G]
```

where `K_G` is the set of affected Jacobian columns for a batch of
contingencies and `rows_needed` contains monitored output rows plus affected
rows needed by the Woodbury correction.

## Recommended Strategy

Use a hybrid CPU/GPU workflow:

1. Build and factor the sparse AC Jacobian on CPU.
2. Solve selected inverse columns on CPU in panels.
3. Transfer each panel to GPU once.
4. Run many fixed-shape JAX outage solves against that panel.
5. Transfer only final or reduced results back to CPU.

This avoids per-outage CPU/GPU transfers and avoids storing full `J^-1`.

```text
CPU:
  sparse J construction
  sparse factorization
  selected RHS solves J X = E_K
  panel row slicing

GPU/JAX:
  gather A = J^-1[K_o, K_o]
  gather G = J^-1[R, K_o]
  batched 4x4 Woodbury solves
  monitored theta/Vm and branch flow reconstruction
```

## Memory Model

Full dense inverse:

```text
bytes_full = n_eq * n_eq * dtype_size
```

Partial panel:

```text
bytes_panel = rows_needed_count * k_panel_count * dtype_size
```

For float64, `dtype_size = 8`.

Example for a 25k-bus subsystem:

```text
n_eq ~= 25k to 50k
K_all ~= 10k to 20k affected states
R ~= monitored states
```

If `n_eq = 50k`, full dense inverse costs:

```text
50k * 50k * 8 ~= 20 GB
```

A partial panel with `R = 50k` and `K_all = 20k` costs:

```text
50k * 20k * 8 ~= 8 GB
```

Use a VRAM target below the physical limit:

```text
panel bytes + JAX temporaries + output arrays < 60% to 75% of GPU VRAM
```

If the whole partial panel does not fit, split contingencies into outage groups
and solve one panel per group.

## Required Index Sets

For each branch outage `o`, define:

```text
K_o = [theta_from, theta_to, Vm_from, Vm_to]
```

after removing invalid entries such as slack-bus angle rows and PV-bus voltage
rows.

For monitored outputs:

```text
R = [theta_i for monitored buses] union [Vm_i for monitored PQ buses]
```

For an outage group `G`:

```text
K_G = unique union of K_o for all outages o in G
rows_needed_G = R union K_G
```

Then the CPU sparse solve produces:

```text
X_G = J^-1[rows_needed_G, K_G]
```

The JAX solve for outage `o` gathers:

```text
A_o = J^-1[K_o, K_o]
G_o = J^-1[R, K_o]
```

from `X_G`.

## Proposed Files

Add the following files:

```text
src/dc_plus/interfaces/partial_inverse_cache.py
src/dc_plus/preprocess/partial_inverse_lodf.py
src/dc_plus/jax/lodf_partial_inverse.py
tests/preprocess/test_partial_inverse_lodf.py
tests/jax/test_lodf_partial_inverse.py
```

Keep the current dense-inverse implementation intact. The partial-inverse path
should be an additional backend.

## Proposed Data Classes

### `OutageIndexSet`

Location:

```text
src/dc_plus/interfaces/partial_inverse_cache.py
```

Purpose: store per-outage affected rows and masks.

```python
from dataclasses import dataclass
import numpy as np
from jaxtyping import Bool, Int


@dataclass(slots=True)
class OutageIndexSet:
    outage_branch_indices: Int[np.ndarray, " n_outages"]
    k_indices: Int[np.ndarray, " n_outages 4"]
    k_mask: Bool[np.ndarray, " n_outages 4"]
```

`k_indices[o]` uses the same fixed order as the current JAX implementation:

```text
[theta_from, theta_to, Vm_from, Vm_to]
```

Invalid entries are set to `0` and masked with `k_mask`.

### `MonitorIndexSet`

Purpose: store monitored state rows.

```python
@dataclass(slots=True)
class MonitorIndexSet:
    monitor_bus_indices: Int[np.ndarray, " n_monitor_buses"]
    r_indices: Int[np.ndarray, " n_r"]
    r_kind: Int[np.ndarray, " n_r"]
    r_bus: Int[np.ndarray, " n_r"]
```

Suggested `r_kind` convention:

```text
0 = theta
1 = Vm
```

This makes it easy to reconstruct `theta_all` and `vm_all` after the partial
panel solve.

### `OutageGroup`

Purpose: group outages so one inverse panel supports many contingencies.

```python
@dataclass(slots=True)
class OutageGroup:
    group_outage_positions: Int[np.ndarray, " n_group_outages"]
    k_panel_indices: Int[np.ndarray, " n_k_panel"]
    rows_needed_indices: Int[np.ndarray, " n_rows_needed"]
    k_local_positions: Int[np.ndarray, " n_group_outages 4"]
    r_local_positions: Int[np.ndarray, " n_r"]
```

`k_local_positions` maps each outage's `K_o` into columns of `k_panel_indices`.
`r_local_positions` maps monitored rows into rows of the panel.

### `PartialInversePanel`

Purpose: hold one solved panel before transfer to JAX.

```python
@dataclass(slots=True)
class PartialInversePanel:
    group: OutageGroup
    panel: np.ndarray  # shape: [n_rows_needed, n_k_panel]
```

Mathematically:

```text
panel = J^-1[rows_needed_indices, k_panel_indices]
```

### `PreparedPartialLODFInputs`

Purpose: device-ready arrays for the JAX kernel.

```python
@dataclass(slots=True)
class PreparedPartialLODFInputs:
    panel: object
    k_local_positions: object
    r_local_positions: object
    k_mask: object
    outage_delta: object
    outage_mismatch: object
    base_theta0: object
    base_vm0: object
    monitor_theta_rows_in_r: object
    monitor_vm_rows_in_r: object
```

Use `object` in the high-level type sketch because these are `jax.Array` at
runtime.

## CPU Preprocessing API

Location:

```text
src/dc_plus/preprocess/partial_inverse_lodf.py
```

### Build Outage Indices

```python
def build_outage_index_set(
    outage_branch_indices: np.ndarray,
    branch_from: np.ndarray,
    branch_to: np.ndarray,
    angle_component_indices: np.ndarray,
    magnitude_component_indices: np.ndarray,
) -> OutageIndexSet:
    ...
```

This should mirror the existing logic in:

```text
src/dc_plus/jax/low_rank_helper.py::_branch_state_indices
```

but run once on CPU for all outages.

### Build Monitor Indices

```python
def build_monitor_index_set(
    monitor_bus_indices: np.ndarray,
    angle_component_indices: np.ndarray,
    magnitude_component_indices: np.ndarray,
) -> MonitorIndexSet:
    ...
```

This should mirror:

```text
src/dc_plus/jax/lodf_voltages.py::build_monitor_rows
```

but produce a compact unique `R` vector.

### Group Outages

```python
def group_outages_for_panels(
    outage_index_set: OutageIndexSet,
    monitor_index_set: MonitorIndexSet,
    max_panel_bytes: int,
    dtype_size: int = 8,
) -> list[OutageGroup]:
    ...
```

Grouping policy:

1. Accumulate outages into a group.
2. Track `K_G = unique(K_o)`.
3. Track `rows_needed = unique(R union K_G)`.
4. Estimate `len(rows_needed) * len(K_G) * dtype_size`.
5. Start a new group before exceeding `max_panel_bytes`.

For better locality, sort or cluster outages by from/to bus area before
grouping. A simple first implementation can use input order.

### Sparse Factorization

```python
def factor_jacobian_cpu(jacobian: sp.sparray):
    return scipy.sparse.linalg.splu(jacobian.tocsc())
```

For production, allow a solver backend abstraction:

```python
class SparseLinearSolver:
    def solve_columns(self, column_indices: np.ndarray) -> np.ndarray:
        ...
```

Initial implementation can wrap SciPy `splu`. Later backends can wrap KLU,
UMFPACK, PARDISO, MUMPS, or a distributed solver.

### Solve a Panel

```python
def solve_partial_inverse_panel(
    solver: SparseLinearSolver,
    group: OutageGroup,
    n_eq: int,
) -> PartialInversePanel:
    ...
```

Implementation:

1. Build sparse or dense RHS identity columns `E_K`.
2. Solve:

   ```text
   X = J^-1[:, K_G]
   ```

3. Keep only:

   ```text
   X[rows_needed_indices, :]
   ```

4. Return `PartialInversePanel`.

Important: avoid forming a full dense `n_eq x n_k_panel` matrix if memory is
tight. If the CPU solver returns full columns, slice immediately and release the
full array.

## JAX Kernel API

Location:

```text
src/dc_plus/jax/lodf_partial_inverse.py
```

### One Outage Kernel

```python
import jax
import jax.numpy as jnp


@jax.jit
def solve_one_partial_inverse_outage(
    panel,
    k_local,
    r_local,
    k_mask,
    d_mat,
    mismatch,
):
    k_mask_f = k_mask.astype(panel.dtype)

    # A = J^-1[K_o, K_o]
    a_rows = k_local
    a_cols = k_local
    a = panel[a_rows[:, None], a_cols[None, :]]
    a = a * k_mask_f[:, None] * k_mask_f[None, :]

    # G = J^-1[R, K_o]
    g = panel[r_local[:, None], k_local[None, :]]
    g = g * k_mask_f[None, :]

    d = d_mat * k_mask_f[:, None] * k_mask_f[None, :]
    m = mismatch * k_mask_f

    base_k = a @ m
    small = jnp.eye(4, dtype=panel.dtype) + d @ a
    corr = jnp.linalg.solve(small, d @ base_k) * k_mask_f

    dx_r = -(g @ m) + (g @ corr)
    return dx_r
```

### Batched Panel Kernel

```python
@jax.jit
def solve_partial_inverse_panel_outages(
    panel,
    k_local_positions,
    r_local_positions,
    k_masks,
    outage_delta,
    outage_mismatch,
):
    return jax.vmap(
        solve_one_partial_inverse_outage,
        in_axes=(None, 0, None, 0, 0, 0),
    )(
        panel,
        k_local_positions,
        r_local_positions,
        k_masks,
        outage_delta,
        outage_mismatch,
    )
```

Output:

```text
dx_r_all: [n_group_outages, n_r]
```

Then reconstruct monitored arrays:

```text
theta_post = theta_hat[monitor_bus_indices] + dx_r[theta monitor rows]
vm_post    = vm_hat[monitor_bus_indices]    + dx_r[Vm monitor rows]
```

Branch current and power reconstruction can reuse logic from:

```text
src/dc_plus/jax/lodf_branches.py
```

## Branch Delta and Mismatch Precompute

Precompute these once per outage group or once globally:

```text
outage_delta[o]    = 4x4 local branch Jacobian delta
outage_mismatch[o] = -[P_from, P_to, Q_from, Q_to]
```

The delta logic should reuse or mirror:

```text
src/dc_plus/jax/low_rank_helper.py::_compute_branch_delta_submatrix_from_admittance
```

For CPU preprocessing, either:

1. call a NumPy equivalent from `src/dc_plus/numpy/low_rank_helper.py`, or
2. add a shared tested helper for branch delta generation.

The branch mismatch construction currently appears in tests. It should become a
library helper:

```python
def build_branch_pq_base(
    branch_from: np.ndarray,
    branch_to: np.ndarray,
    v_mag_hat: np.ndarray,
    theta_hat: np.ndarray,
    y_ff: np.ndarray,
    y_ft: np.ndarray,
    y_tf: np.ndarray,
    y_tt: np.ndarray,
    branch_connected: np.ndarray,
) -> np.ndarray:
    ...
```

Suggested location:

```text
src/dc_plus/preprocess/partial_inverse_lodf.py
```

or a more general:

```text
src/dc_plus/preprocess/branch_quantities.py
```

## Integration With Existing Solver

Do not replace the current dense-inverse API immediately. Add a backend option:

```python
class LODFBackend:
    DENSE_INVERSE = "dense_inverse"
    PARTIAL_INVERSE = "partial_inverse"
```

Possible public API:

```python
def line_outage_post_contingency_monitored_partial_inverse(
    jacobian,
    outage_branch_idx,
    dynamic_network_data,
    jacobian_data,
    monitor_bus_indices,
    monitor_branch_indices,
    max_panel_bytes,
    sparse_solver_backend="scipy_splu",
) -> SolverLoadflowResults:
    ...
```

This can internally:

1. build `OutageIndexSet`,
2. build `MonitorIndexSet`,
3. group outages,
4. factor `jacobian`,
5. loop over panels,
6. transfer each panel to JAX,
7. evaluate JAX kernels,
8. concatenate results in original outage order.

## Tests

### Unit Tests

Add:

```text
tests/preprocess/test_partial_inverse_lodf.py
```

Test:

- `build_outage_index_set()` matches `_branch_state_indices()`.
- invalid slack/PV rows are masked correctly.
- `build_monitor_index_set()` matches current monitor-row logic.
- grouping respects `max_panel_bytes`.
- panel local mappings gather the same values as dense `Jinv`.

### Numerical Tests

Add:

```text
tests/jax/test_lodf_partial_inverse.py
```

Test against the existing dense-inverse JAX path:

1. Build a small test grid.
2. Compute dense `Jinv`.
3. Build partial inverse panels from dense `Jinv` first, as a simple fixture.
4. Run partial-inverse JAX kernel.
5. Assert equality with `line_outage_post_contingency_monitored()`.

Then add a second test:

1. Factor sparse `J` with SciPy.
2. Solve partial panels.
3. Assert equality against dense inverse path within tolerance.

Finally, keep the existing PowSyBl one-step comparison as the end-to-end
reference.

## Performance Benchmarks

Add benchmark timing around:

```text
1. Jacobian build
2. CPU sparse factorization
3. partial panel solve
4. host-to-device transfer
5. JAX compile time
6. steady-state JAX panel solve
7. branch flow reconstruction
```

Report:

```text
outages per second
power flows per second
peak CPU RAM
peak GPU VRAM
panel size
number of panels
```

This is important because the partial-inverse approach trades GPU memory for
CPU sparse solves and transfer time.

## Implementation Order

1. Add data classes in `partial_inverse_cache.py`.
2. Add CPU index-building and grouping helpers.
3. Add branch mismatch helper.
4. Add a partial-panel builder that initially slices from dense `Jinv`.
5. Add JAX partial-inverse outage kernel.
6. Verify exact agreement with current dense-inverse JAX implementation.
7. Add SciPy `splu` partial-panel builder.
8. Verify agreement with dense inverse on small/medium grids.
9. Add panel batching and result concatenation.
10. Benchmark on realistic subsystem cases.
11. Add optional production solver backends beyond SciPy if needed.

## Design Decision Summary

- Factor sparse Jacobian on CPU, not in JAX.
- Use JAX/GPU for fixed-shape batched Woodbury outage solves.
- Transfer one large panel per outage group, not tiny per-outage blocks.
- Store panels, not duplicated `[n_outages, n_R, 4]` monitor blocks.
- Keep the existing dense-inverse backend for small and medium systems where it
  is fastest and fits in VRAM.
- Use partial inverse panels for large systems where dense `J^-1` is too large
  or leaves too little VRAM for JAX temporaries and outputs.
