"""Coordinator for chunked sparse calibration matrix building.

Extracted from ``UnifiedMatrixBuilder.build_matrix_chunked`` so per-chunk
work, final assembly, and any future parallel dispatch share one
well-tested seam. Phase-1 scope: in-process serial execution and
streaming CSR assembly. A later phase will add a Modal dispatch
function that constructs ``ChunkedMatrixAssembler`` on each worker and
calls ``run_chunks`` with its assigned chunk ids.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


@dataclass
class ChunkPlan:
    """Identity and output paths for one column chunk."""

    chunk_id: int
    col_start: int
    col_end: int
    coo_path: Path
    h5_path: Path


@dataclass
class ChunkResult:
    """Per-chunk summary returned after materialization.

    ``nnz`` is the number of nonzero entries written to the COO shard.
    ``cached`` is true when the shard already existed and was reused
    under resume semantics (in which case no kernel work ran).
    """

    chunk_id: int
    nnz: int
    cached: bool = False
    n_households: Optional[int] = None
    n_persons: Optional[int] = None
    unique_states: Optional[int] = None
    unique_counties: Optional[int] = None
    unique_cds: Optional[int] = None


@dataclass
class SharedBuildState:
    """Read-only state every chunk consumes.

    Pickle-clean: only data, no bound instance methods. A Modal worker
    can unpickle this and reconstruct a ``ChunkedMatrixAssembler``
    without access to the originating ``UnifiedMatrixBuilder``.
    """

    source_dataset_path: str
    time_period: int
    rerandomize_takeup: bool
    n_records: int
    n_clones: int
    n_targets: int
    target_variables: List[str]
    target_reform_ids: List[int]
    target_geo_info: List[Tuple[str, str]]
    non_geo_constraints_list: List[List[dict]]
    unique_variables: Set[str]
    unique_constraint_vars: Set[str]
    reform_variables: Set[str]
    target_names: List[str]
    base_entity_maps: object
    block_geoid: np.ndarray
    cd_geoid: np.ndarray
    county_fips: np.ndarray
    state_fips: np.ndarray

    @property
    def n_total(self) -> int:
        return self.n_records * self.n_clones


def partition_chunks(
    n_total: int, chunk_size: int, coo_dir: Path, h5_dir: Path
) -> List[ChunkPlan]:
    """Split ``n_total`` columns into ``ChunkPlan`` objects of ``chunk_size``.

    The last chunk may be smaller. ``chunk_size`` must be positive and
    ``n_total`` must be non-negative.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if n_total < 0:
        raise ValueError("n_total must be non-negative")
    plans: List[ChunkPlan] = []
    chunk_id = 0
    for col_start in range(0, n_total, chunk_size):
        col_end = min(col_start + chunk_size, n_total)
        plans.append(
            ChunkPlan(
                chunk_id=chunk_id,
                col_start=col_start,
                col_end=col_end,
                coo_path=coo_dir / f"chunk_{chunk_id:06d}.npz",
                h5_path=h5_dir / f"chunk_{chunk_id:06d}.h5",
            )
        )
        chunk_id += 1
    return plans


def stream_csr_from_shards(
    shard_dir: Path,
    n_chunks: int,
    n_targets: int,
    n_total: int,
) -> sparse.csr_matrix:
    """Assemble a CSR matrix from per-chunk COO ``.npz`` shards without
    materializing a full COO triple or scipy's internal COO->CSR copy.

    Two passes over shards: pass 1 counts per-row nonzeros across all
    shards to compute ``indptr``; pass 2 scatters each shard's entries
    into preallocated ``data``/``indices`` arrays at the right offsets.

    Peak memory during pass 2 is one shard plus the final CSR arrays.
    """
    row_nnz = np.zeros(n_targets, dtype=np.int64)
    shard_paths: List[Path] = []
    for chunk_id in range(n_chunks):
        path = shard_dir / f"chunk_{chunk_id:06d}.npz"
        shard_paths.append(path)
        with np.load(str(path)) as shard:
            rows = shard["rows"]
            if rows.size == 0:
                continue
            counts = np.bincount(rows.astype(np.int64), minlength=n_targets)
            row_nnz += counts

    total_nnz = int(row_nnz.sum())
    indptr = np.empty(n_targets + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_nnz, out=indptr[1:])

    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int32)
    row_cursor = indptr[:-1].copy()

    for path in shard_paths:
        with np.load(str(path)) as shard:
            rows = shard["rows"]
            if rows.size == 0:
                continue
            cols = shard["cols"]
            vals = shard["vals"]
            # Group entries by row within the shard so we can write
            # contiguous slices per row instead of looping entry-by-entry.
            order = np.argsort(rows, kind="stable")
            rows_sorted = rows[order]
            cols_sorted = cols[order]
            vals_sorted = vals[order]
            unique_rows, starts, counts = np.unique(
                rows_sorted, return_index=True, return_counts=True
            )
            for row, start, count in zip(unique_rows, starts, counts):
                offset = int(row_cursor[row])
                end = start + count
                data[offset : offset + count] = vals_sorted[start:end]
                indices[offset : offset + count] = cols_sorted[start:end]
                row_cursor[row] += count

    # scipy requires indptr/indices to be int32 for canonical CSR; cast
    # once at the end. indices are already int32; indptr may need to be
    # downcast if total_nnz fits.
    if indptr[-1] <= np.iinfo(np.int32).max:
        indptr_final = indptr.astype(np.int32)
    else:
        indptr_final = indptr
    X = sparse.csr_matrix(
        (data, indices, indptr_final),
        shape=(n_targets, n_total),
    )
    X.sort_indices()
    return X


class ChunkedMatrixAssembler:
    """Coordinate partitioning, per-chunk execution, and streaming assembly.

    Serial execution today; a Modal dispatch function can construct one
    of these per worker container and call ``run_chunks`` with the
    worker's assigned chunk ids.

    This class is a deliberate precursor to the ``MatrixAssembler``
    extraction described in ``US Data Pipeline Refactor.md`` Phase 4.
    It owns chunking/assembly today and will absorb target repository,
    simulation batching, and constraint evaluation responsibilities as
    that refactor lands.
    """

    def __init__(
        self,
        shared_state: SharedBuildState,
        chunk_root: Path,
        chunk_size: int,
        resume: bool,
        keep_chunks: bool,
        base_sim=None,
    ):
        self.shared_state = shared_state
        self.chunk_root = Path(chunk_root)
        self.coo_dir = self.chunk_root / "coo"
        self.h5_dir = self.chunk_root / "h5"
        self.coo_dir.mkdir(parents=True, exist_ok=True)
        self.h5_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.resume = resume
        self.keep_chunks = keep_chunks
        self.plans: List[ChunkPlan] = partition_chunks(
            shared_state.n_total, chunk_size, self.coo_dir, self.h5_dir
        )
        self.n_chunks: int = len(self.plans)
        # ``base_sim`` is the source ``Microsimulation`` whose household
        # arrays are sliced by ``materialize_clone_household_chunk``. The
        # facade builds it once and passes it in; a Modal worker in
        # phase 2 would construct it from ``source_dataset_path`` on the
        # volume (unpicklable, so not part of ``SharedBuildState``).
        self._base_sim = base_sim

    def run_chunks(self, chunk_ids: Iterable[int]) -> List[ChunkResult]:
        """Materialize the given chunks serially, honoring resume skip."""
        ids = list(chunk_ids)
        results: List[ChunkResult] = []
        t_build = time.time()
        processed_times: List[float] = []
        cached_chunks = 0
        for i, chunk_id in enumerate(ids):
            t0 = time.time()
            result = self.run_single_chunk(chunk_id)
            results.append(result)
            if result.cached:
                cached_chunks += 1
            else:
                processed_times.append(time.time() - t0)
            remaining = len(ids) - (i + 1)
            if processed_times:
                avg = float(np.mean(processed_times))
                eta = avg * remaining
            else:
                avg = 0.0
                eta = 0.0
            elapsed = time.time() - t_build
            plan = self.plans[chunk_id]
            if result.cached:
                logger.info(
                    "Chunk %d/%d cached: cols %d-%d, cached=%d",
                    chunk_id + 1,
                    self.n_chunks,
                    plan.col_start,
                    plan.col_end - 1,
                    cached_chunks,
                )
            else:
                from policyengine_us_data.calibration.unified_matrix_builder import (
                    _current_rss_mb,
                    _format_duration,
                )

                rss = _current_rss_mb()
                rss_part = f", rss={rss:,.0f} MB" if rss is not None else ""
                logger.info(
                    "Chunk %d/%d: cols %d-%d, hh=%s, persons=%s, "
                    "states=%s, counties=%s, cds=%s, nnz=%d, "
                    "chunk=%s, avg=%s, elapsed=%s, eta=%s%s",
                    chunk_id + 1,
                    self.n_chunks,
                    plan.col_start,
                    plan.col_end - 1,
                    result.n_households,
                    result.n_persons,
                    result.unique_states,
                    result.unique_counties,
                    result.unique_cds,
                    result.nnz,
                    _format_duration(time.time() - t0),
                    _format_duration(avg),
                    _format_duration(elapsed),
                    _format_duration(eta),
                    rss_part,
                )
        return results

    def run_single_chunk(self, chunk_id: int) -> ChunkResult:
        """Run one chunk's kernel: materialize H5, simulate, write shard.

        If ``resume=True`` and a valid shard already exists at the
        expected ``coo_path``, the kernel is skipped and a cached
        ``ChunkResult`` is returned.
        """
        plan = self.plans[chunk_id]
        state = self.shared_state

        if self.resume and plan.coo_path.exists():
            with np.load(str(plan.coo_path)) as cached_chunk:
                if "col_start" not in cached_chunk or "col_end" not in cached_chunk:
                    raise ValueError(
                        f"Cached chunk {plan.coo_path} is missing "
                        "col_start/col_end metadata"
                    )
                cached_col_start = int(np.asarray(cached_chunk["col_start"]).item())
                cached_col_end = int(np.asarray(cached_chunk["col_end"]).item())
                cached_nnz = int(cached_chunk["rows"].shape[0])
            if cached_col_start != plan.col_start or cached_col_end != plan.col_end:
                raise ValueError(
                    f"Cached chunk {plan.coo_path} covers cols "
                    f"{cached_col_start}-{cached_col_end - 1}, "
                    f"expected {plan.col_start}-{plan.col_end - 1}"
                )
            return ChunkResult(chunk_id=chunk_id, nnz=cached_nnz, cached=True)

        # Imports are local so the module is import-safe in lightweight
        # environments (e.g., cold Modal containers that haven't yet
        # run ``uv sync`` for the heavy deps).
        from policyengine_us import Microsimulation

        from policyengine_us_data.calibration.entity_clone import (
            materialize_clone_household_chunk,
        )
        from policyengine_us_data.calibration.unified_matrix_builder import (
            _build_entity_index_maps,
            _calculate_target_values_standalone,
            _make_neutralize_variable_reform,
            build_entity_relationship,
        )

        global_cols = np.arange(plan.col_start, plan.col_end, dtype=np.int64)
        active_hh = global_cols % state.n_records
        active_clone_indices = global_cols // state.n_records
        active_blocks = np.asarray(state.block_geoid)[global_cols]
        active_cd_geoids = np.asarray(state.cd_geoid, dtype=str)[global_cols]
        active_states = np.asarray(state.state_fips)[global_cols]
        active_counties = np.asarray(state.county_fips, dtype=str)[global_cols]

        if self._base_sim is None:
            self._base_sim = Microsimulation(dataset=state.source_dataset_path)
        summary = materialize_clone_household_chunk(
            sim=self._base_sim,
            entity_maps=state.base_entity_maps,
            active_hh=active_hh,
            active_blocks=active_blocks,
            active_cd_geoids=active_cd_geoids,
            active_clone_indices=active_clone_indices,
            output_path=plan.h5_path,
            apply_takeup=state.rerandomize_takeup,
        )

        chunk_sim = Microsimulation(dataset=str(plan.h5_path))
        chunk_n = len(global_cols)
        entity_rel = build_entity_relationship(chunk_sim)
        household_ids = chunk_sim.calculate("household_id", map_to="household").values
        entity_hh_idx_map, person_to_entity_idx_map = _build_entity_index_maps(
            entity_rel, household_ids, chunk_sim
        )

        variable_entity_map: Dict[str, str] = {}
        hh_vars: Dict[str, np.ndarray] = {}
        target_entity_vars: Dict[str, np.ndarray] = {}
        for variable in sorted(state.unique_variables):
            if variable in chunk_sim.tax_benefit_system.variables:
                variable_entity_map[variable] = chunk_sim.tax_benefit_system.variables[
                    variable
                ].entity.key
            if variable.endswith("_count"):
                continue
            try:
                hh_vars[variable] = chunk_sim.calculate(
                    variable, state.time_period, map_to="household"
                ).values.astype(np.float32)
            except Exception as exc:
                logger.warning(
                    "Chunk %d cannot calculate target '%s': %s",
                    chunk_id,
                    variable,
                    exc,
                )
            entity_key = variable_entity_map.get(variable, "household")
            if entity_key == "household":
                continue
            try:
                target_entity_vars[variable] = chunk_sim.calculate(
                    variable, state.time_period, map_to=entity_key
                ).values.astype(np.float32)
            except Exception as exc:
                logger.warning(
                    "Chunk %d cannot calculate entity-level target '%s' "
                    "(map_to=%s): %s",
                    chunk_id,
                    variable,
                    entity_key,
                    exc,
                )

        person_vars: Dict[str, np.ndarray] = {}
        for variable in sorted(state.unique_constraint_vars):
            try:
                raw = chunk_sim.calculate(
                    variable, state.time_period, map_to="person"
                ).values
                try:
                    person_vars[variable] = raw.astype(np.float32)
                except (ValueError, TypeError):
                    person_vars[variable] = raw
            except Exception as exc:
                logger.warning(
                    "Chunk %d cannot calculate constraint '%s': %s",
                    chunk_id,
                    variable,
                    exc,
                )

        reform_hh_vars: Dict[str, np.ndarray] = {}
        if state.reform_variables:
            baseline_income_tax = chunk_sim.calculate(
                "income_tax", state.time_period, map_to="household"
            ).values.astype(np.float32)
            for variable in sorted(state.reform_variables):
                try:
                    reform_sim = Microsimulation(
                        dataset=str(plan.h5_path),
                        reform=_make_neutralize_variable_reform(variable),
                    )
                    reform_income_tax = reform_sim.calculate(
                        "income_tax", state.time_period, map_to="household"
                    ).values.astype(np.float32)
                    reform_hh_vars[variable] = reform_income_tax - baseline_income_tax
                except Exception as exc:
                    logger.warning(
                        "Chunk %d cannot calculate reform target '%s': %s",
                        chunk_id,
                        variable,
                        exc,
                    )

        target_value_cache: Dict[tuple, np.ndarray] = {}
        rows_list: List[np.ndarray] = []
        cols_list: List[np.ndarray] = []
        vals_list: List[np.ndarray] = []

        for row_idx in range(state.n_targets):
            variable = state.target_variables[row_idx]
            reform_id = state.target_reform_ids[row_idx]
            geo_level, geo_id = state.target_geo_info[row_idx]
            non_geo = state.non_geo_constraints_list[row_idx]

            if geo_level == "district":
                geo_mask = active_cd_geoids == str(geo_id)
            elif geo_level == "state":
                geo_mask = active_states.astype(np.int64) == int(geo_id)
            elif geo_level == "county":
                geo_mask = active_counties == str(geo_id).zfill(5)
            else:
                geo_mask = np.ones(chunk_n, dtype=bool)

            if not geo_mask.any():
                continue

            constraint_key = tuple(
                sorted((c["variable"], c["operation"], c["value"]) for c in non_geo)
            )
            value_key = (variable, constraint_key, reform_id)
            if value_key not in target_value_cache:
                target_value_cache[value_key] = _calculate_target_values_standalone(
                    target_variable=variable,
                    non_geo_constraints=non_geo,
                    n_households=chunk_n,
                    hh_vars=hh_vars,
                    reform_hh_vars=reform_hh_vars,
                    target_entity_vars=target_entity_vars,
                    person_vars=person_vars,
                    entity_rel=entity_rel,
                    household_ids=household_ids,
                    variable_entity_map=variable_entity_map,
                    entity_hh_idx_map=entity_hh_idx_map,
                    person_to_entity_idx_map=person_to_entity_idx_map,
                    reform_id=reform_id,
                )
            values = target_value_cache[value_key]

            vals = values[geo_mask]
            nonzero = vals != 0
            if nonzero.any():
                rows_list.append(np.full(nonzero.sum(), row_idx, dtype=np.int32))
                cols_list.append(global_cols[geo_mask][nonzero].astype(np.int32))
                vals_list.append(vals[nonzero].astype(np.float32, copy=False))

        if rows_list:
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
            vals = np.concatenate(vals_list)
        else:
            rows = np.array([], dtype=np.int32)
            cols = np.array([], dtype=np.int32)
            vals = np.array([], dtype=np.float32)

        np.savez_compressed(
            str(plan.coo_path),
            rows=rows,
            cols=cols,
            vals=vals,
            col_start=np.array([plan.col_start], dtype=np.int64),
            col_end=np.array([plan.col_end], dtype=np.int64),
        )

        if not self.keep_chunks and plan.h5_path.exists():
            plan.h5_path.unlink()

        return ChunkResult(
            chunk_id=chunk_id,
            nnz=int(vals.shape[0]),
            cached=False,
            n_households=getattr(summary, "n_households", None),
            n_persons=getattr(summary, "n_persons", None),
            unique_states=getattr(summary, "unique_states", None),
            unique_counties=getattr(summary, "unique_counties", None),
            unique_cds=getattr(summary, "unique_cds", None),
        )

    def assemble_final(self) -> sparse.csr_matrix:
        """Stream-assemble the final CSR matrix from all shards on disk."""
        logger.info("Assembling matrix from %d chunk files...", self.n_chunks)
        X_csr = stream_csr_from_shards(
            shard_dir=self.coo_dir,
            n_chunks=self.n_chunks,
            n_targets=self.shared_state.n_targets,
            n_total=self.shared_state.n_total,
        )
        logger.info(
            "Chunked matrix: %d targets x %d cols, %d nnz",
            X_csr.shape[0],
            X_csr.shape[1],
            X_csr.nnz,
        )
        return X_csr
