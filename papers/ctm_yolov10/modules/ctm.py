"""Clustering–Template Matching (CTM) module for wafer die segmentation.

Implements the CTM preprocessing strategy described in:

    "Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network"
    (Section 2.1, Figure 2)

The CTM module:
    1. Takes a wafer grayscale image containing multiple dies in the field of view.
    2. Applies template matching (normalized cross-correlation) to find
       candidate die positions.
    3. Uses Affinity Propagation (AP) clustering to group redundant matches
       and select the best match per cluster.
    4. Extracts individual die images for downstream defect detection.

This module is used as a preprocessing step BEFORE the improved YOLOv10
detector, not as a network layer.

Reference:
    Frey & Dueck, "Clustering by Passing Messages Between Data Points",
    Science 2007.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def normalized_cross_correlation(
    image: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """Compute normalized cross-correlation (NCC) between image and template.

    Uses ``cv2.TM_CCOEFF_NORMED`` which implements Equation (1) from the paper:

        R(i,j) = sum(S^{i,j}(m,n) * T(m,n)) /
                 (sqrt(sum(S^{i,j}(i,j)^2)) * sqrt(sum(T(i,j)^2)))

    Args:
        image: Grayscale search image [H, W].
        template: Grayscale template image [M, M].

    Returns:
        Correlation map of shape [H-M+1, W-M+1] with values in [-1, 1].
    """
    import cv2
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return result


def affinity_propagation_clustering(
    matches: np.ndarray,
    preference: float | None = None,
    damping: float = 0.5,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster template matching results using Affinity Propagation.

    AP clustering does not require a pre-specified number of clusters,
    making it suitable for variable numbers of dies per field of view.

    Args:
        matches: Array of match bounding boxes [N, 4] in (x1, y1, x2, y2) format.
        preference: Input preference for AP (lower → fewer clusters).
            If ``None``, uses the median of similarities.
        damping: Damping factor for AP (0.5-1.0).
        max_iter: Maximum number of AP iterations.

    Returns:
        Tuple of (cluster_labels, exemplar_indices):
            - cluster_labels: [N] array of cluster assignments.
            - exemplar_indices: [K] indices of exemplar (best) matches.
    """
    try:
        from sklearn.cluster import AffinityPropagation
    except ImportError:
        raise ImportError(
            "scikit-learn is required for Affinity Propagation clustering. "
            "Install it with: pip install scikit-learn"
        )

    if len(matches) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Use bounding box centers as features for clustering
    centers = np.column_stack([
        (matches[:, 0] + matches[:, 2]) / 2,  # cx
        (matches[:, 1] + matches[:, 3]) / 2,  # cy
    ])

    # Compute similarity matrix (negative squared Euclidean distance)
    # as described in Equations (2)-(3) of the paper
    ap = AffinityPropagation(
        preference=preference,
        damping=damping,
        max_iter=max_iter,
        random_state=42,
    )
    labels = ap.fit_predict(centers)

    # Select exemplar (best match) per cluster: the one with highest
    # correlation score (closest to cluster center)
    exemplar_indices = []
    for cluster_id in range(labels.max() + 1):
        cluster_mask = labels == cluster_id
        cluster_centers = centers[cluster_mask]
        # Find the point closest to the cluster center (exemplar)
        cluster_center = cluster_centers.mean(axis=0)
        distances = np.linalg.norm(cluster_centers - cluster_center, axis=1)
        local_idx = int(np.argmin(distances))
        # Map back to global index
        global_indices = np.where(cluster_mask)[0]
        exemplar_indices.append(global_indices[local_idx])

    return labels, np.array(exemplar_indices)


def match_template_with_clustering(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.7,
    preference: float | None = None,
    damping: float = 0.5,
) -> list[dict[str, Any]]:
    """Apply clustering–template matching to locate dies in a wafer image.

    This implements the full CTM algorithm from Section 2.1:
        1. Compute NCC between image and template.
        2. Find all matches above ``threshold``.
        3. Cluster matches using AP to eliminate redundancy.
        4. Return one best match per cluster.

    Args:
        image: Grayscale wafer image [H, W].
        template: Grayscale die template [M, M].
        threshold: NCC threshold for candidate matches (default 0.7).
        preference: AP preference parameter (``None`` = auto).
        damping: AP damping factor.

    Returns:
        List of match dicts, each with:
            - ``bbox``: (x1, y1, x2, y2) bounding box of the matched die.
            - ``score``: NCC correlation score.
            - ``die_image``: Extracted die image patch.
    """
    # Step 1: Compute NCC
    correlation = normalized_cross_correlation(image, template)
    h, w = correlation.shape
    th, tw = template.shape

    # Step 2: Find all matches above threshold
    match_locations = np.where(correlation >= threshold)
    if len(match_locations[0]) == 0:
        return []

    boxes = []
    scores = []
    for y, x in zip(*match_locations):
        x1, y1 = x, y
        x2, y2 = x + tw, y + th
        boxes.append([x1, y1, x2, y2])
        scores.append(correlation[y, x])

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Step 3: Cluster matches using AP
    labels, exemplar_idx = affinity_propagation_clustering(
        boxes, preference=preference, damping=damping,
    )

    # Step 4: Return one best match per cluster
    results = []
    for idx in exemplar_idx:
        x1, y1, x2, y2 = boxes[idx]
        die_image = image[y1:y2, x1:x2]
        results.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "score": float(scores[idx]),
            "die_image": die_image,
        })

    return results


class CTM:
    """Clustering–Template Matching module for wafer die segmentation.

    This is a preprocessing module that extracts individual die images
    from a wafer field-of-view image before feeding them to the improved
    YOLOv10 detector.

    Args:
        template: Die template image (grayscale). If ``None``, must be
            provided at runtime via ``set_template()``.
        threshold: NCC threshold for candidate matches.
        preference: AP preference parameter.
        damping: AP damping factor.
    """

    def __init__(
        self,
        template: np.ndarray | None = None,
        threshold: float = 0.7,
        preference: float | None = None,
        damping: float = 0.5,
    ) -> None:
        self.template = template
        self.threshold = threshold
        self.preference = preference
        self.damping = damping

    def set_template(self, template: np.ndarray) -> None:
        """Set or update the die template image."""
        self.template = template

    def __call__(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run CTM preprocessing on a wafer image.

        Args:
            image: Grayscale wafer image [H, W] or RGB [H, W, 3].

        Returns:
            List of matched die dicts with ``bbox``, ``score``, ``die_image``.
        """
        import cv2

        if self.template is None:
            raise ValueError("CTM template not set. Call set_template() first.")

        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Convert template to grayscale if needed
        templ = self.template
        if templ.ndim == 3:
            templ = cv2.cvtColor(templ, cv2.COLOR_RGB2GRAY)

        return match_template_with_clustering(
            gray, templ,
            threshold=self.threshold,
            preference=self.preference,
            damping=self.damping,
        )