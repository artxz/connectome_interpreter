import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import inspect
import tempfile
from unittest import mock

from connectome_interpreter.external_paths import (
    compute_flow_hitting_time,
    find_instance_flow,
    layered_el,
)


class TestLayeredEl(unittest.TestCase):
    def test_percentile_is_forwarded_to_group_paths(self):
        matrix = csr_matrix(
            (
                np.array([1.0, 2.0, 3.0, 4.0]),
                (np.zeros(4, dtype=int), np.arange(1, 5)),
            ),
            shape=(5, 5),
        )
        idx_to_group = {0: "A", 1: "B", 2: "B", 3: "B", 4: "B"}
        flow = pd.DataFrame(
            {"cell_group": ["A", "B"], "hitting_time": [0.0, 1.0]}
        )

        result, _, _, _ = layered_el(
            matrix,
            inidx=[0],
            outidx=[1, 2, 3, 4],
            n=1,
            idx_to_group=idx_to_group,
            combining_method="percentile",
            percentile=75,
            flow=flow,
        )

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["weight"], 3.25)


def _navis_has_method():
    """Return True if navis' BayesianTraversalModel takes `method`."""
    try:
        from navis.models import BayesianTraversalModel
    except ImportError:
        return False
    return "method" in inspect.signature(BayesianTraversalModel.__init__).parameters


# Chain 0 -> 1 -> 2 -> 3; weight 0.05 with flow_thre 0.1 gives p = 0.5 per edge.
_CHAIN = csr_matrix(
    (np.full(3, 0.05), (np.arange(3), np.arange(1, 4))), shape=(4, 4)
)


@unittest.skipUnless(_navis_has_method(), "navis without BayesianTraversalModel(method=)")
class TestFlowMethod(unittest.TestCase):
    def _hit(self, method):
        df = compute_flow_hitting_time(_CHAIN, np.array([0]), 10, 0.1, method=method)
        return df.set_index("idx")["hitting_time"]

    def test_default_is_navis_default(self):
        np.testing.assert_allclose(self._hit(None), self._hit("conditioned"))

    def test_legacy_differs_from_conditioned(self):
        legacy, cond = self._hit("legacy"), self._hit("conditioned")
        np.testing.assert_allclose(legacy[1], cond[1])  # one edge: same model
        self.assertLess(legacy[3], cond[3])  # legacy is biased low on chains

    def test_find_instance_flow_passes_method(self):
        idx_to_group = {0: "A", 1: "B", 2: "C", 3: "D"}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "connectome_interpreter.external_paths.compute_flow_hitting_time",
                wraps=compute_flow_hitting_time,
            ) as spy:
                find_instance_flow(
                    _CHAIN, idx_to_group, flow_seed_groups=["A"], file_path=tmp,
                    flow_steps=10, method="legacy",
                )
        self.assertEqual(spy.call_args.kwargs["method"], "legacy")


class TestFlowMethodUnsupported(unittest.TestCase):
    def test_clear_error_when_navis_lacks_method(self):
        class _OldModel:
            def __init__(self, *args, traversal_func=None, **kwargs):
                pass

        with mock.patch("navis.models.BayesianTraversalModel", _OldModel):
            with self.assertRaisesRegex(ValueError, "navis-org/navis#202"):
                compute_flow_hitting_time(_CHAIN, np.array([0]), 10, 0.1, method="legacy")
