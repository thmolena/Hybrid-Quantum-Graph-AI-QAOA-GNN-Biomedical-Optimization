"""Concentration-aware trust-region projection for QAOA angle predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class ConcentrationStats:
    center: np.ndarray
    radius_mean: float
    radius_rms: float
    radius_max: float
    num_instances: int


@dataclass(frozen=True)
class ProjectionResult:
    raw_angles: np.ndarray
    projected_angles: np.ndarray
    center_distance: float
    projected_distance: float
    radius: float
    projection_active: bool
    shrinkage: float


def concentration_stats_from_angle_array(angle_array: np.ndarray) -> ConcentrationStats:
    angles = np.asarray(angle_array, dtype=np.float64)
    if angles.ndim != 2:
        raise ValueError(f"Expected a 2D angle array, received shape {angles.shape}")
    center = angles.mean(axis=0)
    distances = np.linalg.norm(angles - center, axis=1)
    return ConcentrationStats(
        center=center,
        radius_mean=float(distances.mean()),
        radius_rms=float(np.sqrt(np.mean(distances**2))),
        radius_max=float(distances.max()),
        num_instances=int(len(angles)),
    )


def concentration_stats_from_instances(instances: Sequence[Dict[str, object]]) -> ConcentrationStats:
    if not instances:
        raise ValueError("Cannot compute concentration stats from an empty instance list")
    angles = np.vstack([np.asarray(instance["target_angles"], dtype=np.float64) for instance in instances])
    return concentration_stats_from_angle_array(angles)


def project_onto_ball(raw_angles: np.ndarray, center: np.ndarray, radius: float) -> ProjectionResult:
    raw = np.asarray(raw_angles, dtype=np.float64).reshape(-1)
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    if raw.shape != center.shape:
        raise ValueError(f"Angle shape mismatch: {raw.shape} vs {center.shape}")
    if radius < 0.0:
        raise ValueError("radius must be nonnegative")

    displacement = raw - center
    distance = float(np.linalg.norm(displacement))
    if distance == 0.0 or distance <= radius:
        return ProjectionResult(
            raw_angles=raw,
            projected_angles=raw.copy(),
            center_distance=distance,
            projected_distance=distance,
            radius=float(radius),
            projection_active=False,
            shrinkage=0.0,
        )

    scale = float(radius / distance)
    projected = center + scale * displacement
    return ProjectionResult(
        raw_angles=raw,
        projected_angles=projected,
        center_distance=distance,
        projected_distance=float(np.linalg.norm(projected - center)),
        radius=float(radius),
        projection_active=True,
        shrinkage=1.0 - scale,
    )


def project_with_trust_factor(raw_angles: np.ndarray, stats: ConcentrationStats, trust_factor: float = 1.0) -> ProjectionResult:
    if trust_factor <= 0.0:
        raise ValueError("trust_factor must be positive")
    return project_onto_ball(raw_angles, stats.center, trust_factor * stats.radius_max)
