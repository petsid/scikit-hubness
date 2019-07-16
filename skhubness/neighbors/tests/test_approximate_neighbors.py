# SPDX-License-Identifier: BSD-3-Clause

import pytest
from sklearn.utils.estimator_checks import check_estimator

from skhubness.neighbors.approximate_neighbors import ApproximateNearestNeighbor, UnavailableANN
from skhubness.neighbors import LSH, HNSW


@pytest.mark.parametrize('ann', [UnavailableANN, LSH])  # HNSW
def test_sklearn_baseestimator(ann):
    # TODO fix LSH, HNSW
    check_estimator(ann)
