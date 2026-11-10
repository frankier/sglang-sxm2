"""V100 regression for the 256 MiB FP32 MHC prefill temporary."""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.elementwise.sm70_glm_hc import glm_hc_pre
from sglang.kernels.ops.layernorm.mhc import hc_pre
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0),
    "Requires a V100 (SM70)",
)
class TestSm70MhcPrefill(CustomTestCase):
    @torch.inference_mode()
    def test_prefill_matches_fp32_reference_with_bounded_workspace(self):
        # A full prefill chunk plus one row also exercises the projection tail.
        torch.manual_seed(1)
        x = torch.randn(4097, 16384, dtype=torch.float16, device="cuda")
        fn = torch.randn(24, 16384, device="cuda") * 0.02
        scale = torch.rand(3, device="cuda") + 0.5
        base = torch.randn(24, device="cuda")
        eps = 1e-6

        x32 = x.float()
        mixes = F.linear(x32, fn) * torch.rsqrt(
            x32.square().mean(dim=-1, keepdim=True) + eps
        )
        del x32
        post_ref, comb_ref, layer_ref = glm_hc_pre(
            x.view(-1, 4, 4096), mixes, scale, base, 20, eps
        )
        with envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False):
            for tokens in (65, x.shape[0]):
                with self.subTest(tokens=tokens):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    allocated = torch.cuda.memory_allocated()
                    layer, comb, post, _ = hc_pre(
                        x[:tokens], fn, scale, base, 4, eps, eps, 20
                    )
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() - allocated

                    # The old large-batch path held over 512 MiB of FP32 buffers.
                    self.assertLess(peak, 96 * 1024**2)
                    torch.testing.assert_close(
                        post, post_ref[:tokens].flatten(1), atol=5e-6, rtol=2e-5
                    )
                    torch.testing.assert_close(
                        comb, comb_ref[:tokens].flatten(1), atol=5e-6, rtol=2e-5
                    )
                    torch.testing.assert_close(
                        layer, layer_ref[:tokens], atol=2e-5, rtol=1e-3
                    )


if __name__ == "__main__":
    unittest.main()
