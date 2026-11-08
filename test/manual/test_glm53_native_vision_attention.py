"""Native GLM-5.3 page attention memory regression. Run on an idle V100.

The original dense mask/SDPA path exceeds the 5% allocator limit even when
the GPU is otherwise empty. No checkpoint or language model is needed.
"""

import json
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.vision import VisionSdpaAttention
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0),
    "Requires a V100 (SM70)",
)
class TestNativePageAttention(CustomTestCase):
    def test_packed_unequal_frames_match_dense_reference(self):
        torch.manual_seed(10)
        q, k, v = [
            torch.randn(450, 2, 64, device="cuda", dtype=torch.float16)
            for _ in range(3)
        ]
        frame = torch.repeat_interleave(
            torch.arange(2, device="cuda"), torch.tensor([117, 333], device="cuda")
        )
        attention = VisionSdpaAttention(
            head_dim=64,
            num_heads=2,
            num_kv_heads=2,
            flatten_batch=True,
            softmax_scale=0.37,
        )
        attention._MAX_SCORE_BYTES = 32 * 1024
        with torch.inference_mode():
            expected = F.scaled_dot_product_attention(
                q.transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1),
                attn_mask=frame[:, None] == frame[None, :],
                scale=0.37,
            ).transpose(0, 1)
            actual = attention(
                q, k, v, bsz=1, cu_seqlens=torch.tensor([0, 117, 450], device="cuda")
            )
            changed = v.clone()
            changed[117:] += 3
            other = attention(
                q,
                k,
                changed,
                bsz=1,
                cu_seqlens=torch.tensor([0, 117, 450], device="cuda"),
            )
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(actual[:117], other[:117], atol=0, rtol=0)
        self.assertEqual(actual.dtype, q.dtype)
        print(
            json.dumps(
                {"packed_max_abs_error": (actual - expected).abs().max().item()}
            ),
            flush=True,
        )

    def test_31800_patches_fit_without_a_square_attention_allocation(self):
        torch.manual_seed(9)
        torch.cuda.set_per_process_memory_fraction(0.05)
        self.addCleanup(torch.cuda.set_per_process_memory_fraction, 1.0)
        torch.cuda.reset_peak_memory_stats()
        length = 31800  # The issue's [1, 150, 212] native image grid.
        q, k, v = [
            torch.randn(length, 2, 64, device="cuda", dtype=torch.float16)
            for _ in range(3)
        ]
        attention = VisionSdpaAttention(
            head_dim=64, num_heads=2, num_kv_heads=2, flatten_batch=True
        )
        with torch.inference_mode():
            actual = attention(
                q, k, v, bsz=1, cu_seqlens=torch.tensor([0, length], device="cuda")
            )
            # Full keys for selected queries provide a reference without
            # requiring the very square allocation this test forbids.
            selected = torch.linspace(0, length - 1, 17, device="cuda").long()
            expected = F.scaled_dot_product_attention(
                q[selected].transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            ).transpose(0, 1)
        torch.testing.assert_close(actual[selected], expected, atol=1e-3, rtol=1e-3)
        peak = torch.cuda.max_memory_allocated()
        self.assertLess(peak, 512 * 1024**2)
        print(
            json.dumps(
                {
                    "patches": length,
                    "peak_allocated_bytes": peak,
                    "max_abs_error": (actual[selected] - expected).abs().max().item(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
