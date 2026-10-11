"""Unit tests for the Kolibri 1 router."""

import unittest

import torch

from sglang.srt.models.kolibri1 import kolibri1_routing
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKolibri1Routing(CustomTestCase):
    def test_selects_on_biased_logits_and_weights_unbiased_sigmoid(self):
        """Matches torchtitan's sigmoid_logit_add router: top-k over logits + bias,
        weights sigmoid(logits) of the chosen experts; sigmoid + bias picks differently."""
        torch.manual_seed(0)
        logits = torch.randn(64, 384) * 3
        # Bias magnitudes seen in the released checkpoint reach about 20.
        bias = torch.randn(384) * 5

        weights, ids = kolibri1_routing(
            hidden_states=torch.empty(64, 8),
            gating_output=logits,
            topk=6,
            renormalize=False,
            correction_bias=bias,
        )

        ref_ids = torch.topk(logits + bias, k=6, dim=-1)[1]
        self.assertTrue(torch.equal(ids.long(), ref_ids))
        torch.testing.assert_close(weights, torch.sigmoid(logits).gather(1, ref_ids))
        score_add_ids = torch.topk(torch.sigmoid(logits) + bias, k=6, dim=-1)[1]
        self.assertFalse(
            torch.equal(ref_ids.sort(-1).values, score_add_ids.sort(-1).values)
        )


if __name__ == "__main__":
    unittest.main()
