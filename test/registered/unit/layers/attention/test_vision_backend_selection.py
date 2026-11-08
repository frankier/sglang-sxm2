import sys
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.layers.attention import vision
from sglang.test.ci.ci_register import register_cpu_ci, register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_npu_ci(est_time=2, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=2, suite="nightly-1-npu-a3", nightly=True)


class TestPackedSdpaMemory(CustomTestCase):
    def test_chunked_frames_match_dense_attention_and_remain_isolated(self):
        """Native document frames must not allocate a square mask or score matrix."""
        torch.manual_seed(7)
        lengths = (5, 9)
        heads, dim = 2, 8
        q, k, v = [torch.randn(sum(lengths), heads, dim) for _ in range(3)]
        frame_ids = torch.repeat_interleave(torch.arange(2), torch.tensor(lengths))
        mask = frame_ids[:, None] == frame_ids[None, :]
        scale = 0.37
        expected = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
                scale=scale,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        layer = vision.VisionSdpaAttention(
            head_dim=dim,
            num_heads=heads,
            num_kv_heads=heads,
            flatten_batch=True,
            softmax_scale=scale,
        )
        layer._MAX_SCORE_BYTES = 128
        sdpa = F.scaled_dot_product_attention

        def bounded_sdpa(queries, keys, values, **kwargs):
            self.assertIsNone(kwargs.get("attn_mask"))
            self.assertLessEqual(
                queries.shape[1] * queries.shape[2] * keys.shape[2] * 4,
                layer._MAX_SCORE_BYTES,
            )
            return sdpa(queries, keys, values, **kwargs)

        with patch.object(vision.F, "scaled_dot_product_attention", bounded_sdpa):
            actual = layer(q, k, v, bsz=1, cu_seqlens=torch.tensor([0, 5, 14]))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        changed = v.clone()
        changed[5:] += 100
        other = layer(q, k, changed, bsz=1, cu_seqlens=torch.tensor([0, 5, 14]))
        torch.testing.assert_close(other[:5], actual[:5], atol=0, rtol=0)
        torch.testing.assert_close(other[5:], actual[5:] + 100, atol=2e-5, rtol=1e-6)

    def test_single_frame_matches_unmasked_sdpa_and_rejects_invalid_partition(self):
        torch.manual_seed(8)
        q, k, v = [torch.randn(11, 2, 8) for _ in range(3)]
        layer = vision.VisionSdpaAttention(
            head_dim=8,
            num_heads=2,
            num_kv_heads=2,
            flatten_batch=True,
        )
        layer._MAX_SCORE_BYTES = 128
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        ).transpose(0, 1)
        sdpa = F.scaled_dot_product_attention

        def bounded_sdpa(queries, keys, values, **kwargs):
            self.assertIsNone(kwargs.get("attn_mask"))
            self.assertLessEqual(
                queries.shape[1] * queries.shape[2] * keys.shape[2] * 4,
                layer._MAX_SCORE_BYTES,
            )
            return sdpa(queries, keys, values, **kwargs)

        with patch.object(vision.F, "scaled_dot_product_attention", bounded_sdpa):
            for cu in (None, torch.tensor([0, 11])):
                actual = layer(q, k, v, bsz=1, cu_seqlens=cu)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "partition"):
            layer(q, k, v, bsz=1, cu_seqlens=torch.tensor([0, 12]))

        valid = torch.arange(11) < 10
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            attn_mask=valid[:, None] & valid[None, :],
        ).transpose(0, 1)
        actual = layer(q, k, v, bsz=1, cu_seqlens=torch.tensor([0, 10]))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.fixture
def npu_platform(monkeypatch):
    monkeypatch.setattr(vision, "is_cuda", lambda: False)
    monkeypatch.setattr(vision, "_is_npu", True)
    monkeypatch.setattr(vision, "_is_musa", False)
    monkeypatch.setattr(vision, "_is_hip", False)
    monkeypatch.setattr(vision, "_is_cpu", False)
    monkeypatch.setattr(vision, "_is_xpu", False)


@pytest.mark.parametrize(
    ("server_backend", "passed_backend", "expected"),
    [
        (None, None, "ascend_attn"),
        (None, "sdpa", "sdpa"),
        ("sdpa", None, "sdpa"),
        ("sdpa", "ascend_attn", "sdpa"),
    ],
)
def test_npu_backend_selection_priority(
    monkeypatch,
    npu_platform,
    server_backend,
    passed_backend,
    expected,
):
    monkeypatch.setattr(
        vision,
        "get_mm",
        lambda: SimpleNamespace(mm_attention_backend=server_backend),
    )

    backend = vision.VisionAttention._determine_attention_backend(None, passed_backend)

    assert backend == expected


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((9, 0), "fa3"),
        ((10, 0), "fa4"),
        ((10, 3), "fa4"),
        ((12, 0), "triton_attn"),
    ],
)
def test_cuda_default_backend_by_capability(monkeypatch, capability, expected):
    monkeypatch.setattr(vision, "is_cuda", lambda: True)
    monkeypatch.setattr(vision, "get_device_capability", lambda: capability)
    monkeypatch.setattr(
        vision, "get_platform", lambda: SimpleNamespace(is_blackwell=False)
    )
    monkeypatch.setattr(
        vision, "get_mm", lambda: SimpleNamespace(mm_attention_backend=None)
    )

    backend = vision.VisionAttention._determine_attention_backend(None, None)

    assert backend == expected


@pytest.mark.parametrize("graph_env", ["0", "1"])
def test_fa4_seqlens_from_vit_cuda_graph_list(monkeypatch, graph_env):
    # ViT CUDA graph runners pass [cu_seqlens, max_seqlen]; tensors must keep
    # working even when SGLANG_VIT_ENABLE_CUDA_GRAPH is set for another model.
    monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", graph_env)
    monkeypatch.setattr(vision, "_is_cuda", True)
    calls = []

    def fake_flash_attn_func(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    monkeypatch.setattr(vision, "flash_attn_func", fake_flash_attn_func, raising=False)
    q = torch.randn(6, 2, 8)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)
    backend = vision.VisionFlash4Attention()

    backend(q, q, q, cu_seqlens=[cu_seqlens, 4], bsz=1, seq_len=6)
    backend(q, q, q, cu_seqlens=cu_seqlens, bsz=1, seq_len=6, max_seqlen=4)

    for kwargs in calls:
        assert kwargs["ver"] == 4
        assert torch.equal(kwargs["cu_seqlens_q"], cu_seqlens)
        assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 4
    assert calls[0]["cu_seqlens_q"] is cu_seqlens


def test_explicit_backend_without_published_mm_context(monkeypatch, npu_platform):
    monkeypatch.setattr(
        vision,
        "get_mm",
        Mock(side_effect=ValueError("config namespace 'mm' not published")),
    )
    monkeypatch.setattr(
        vision,
        "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda namespace: False),
    )

    backend = vision.VisionAttention._determine_attention_backend(None, "sdpa")

    assert backend == "sdpa"


def test_explicit_backend_keeps_published_context_errors(monkeypatch, npu_platform):
    monkeypatch.setattr(
        vision,
        "get_mm",
        Mock(side_effect=ValueError("mm namespace is not available for this role")),
    )
    monkeypatch.setattr(
        vision,
        "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda namespace: True),
    )

    with pytest.raises(ValueError, match="not available for this role"):
        vision.VisionAttention._determine_attention_backend(None, "sdpa")


def test_sdpa_preserves_flattened_batch_layout():
    torch.manual_seed(0)
    bsz, seq_len, num_heads, head_dim = 3, 5, 2, 8
    q, k, v = [torch.randn(bsz * seq_len, num_heads, head_dim) for _ in range(3)]
    backend = vision.VisionSdpaAttention(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )

    output = backend(q=q, k=k, v=v, bsz=bsz)
    q_ref, k_ref, v_ref = [
        x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2) for x in (q, k, v)
    ]
    expected = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        scale=backend.scale,
    )
    expected = expected.transpose(1, 2).reshape(bsz * seq_len, num_heads, head_dim)

    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("mask_kind", ["causal", "padding"])
def test_ascend_attention_masked_inputs_fall_back_to_sdpa(
    monkeypatch,
    npu_platform,
    mask_kind,
):
    torch.manual_seed(0)
    bsz, seq_len, num_heads, head_dim = 2, 4, 2, 8
    softmax_scale = 0.37
    q, k, v = [torch.randn(bsz * seq_len, num_heads, head_dim) for _ in range(3)]
    mask = torch.zeros(bsz, 1, seq_len, seq_len)
    if mask_kind == "causal":
        masked_positions = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
        mask.masked_fill_(masked_positions, torch.finfo(mask.dtype).min)
    else:
        mask[:, :, :, -1] = torch.finfo(mask.dtype).min

    fused_attention = Mock(
        side_effect=AssertionError("masked inputs must not use Ascend fused attention")
    )
    monkeypatch.setattr(
        vision,
        "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=fused_attention),
        raising=False,
    )
    backend = vision.VisionAscendAttention(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        softmax_scale=softmax_scale,
    )

    output = backend(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.arange(0, (bsz + 1) * seq_len, seq_len),
        bsz=bsz,
        seq_len=seq_len,
        attention_mask=mask,
    )
    q_ref, k_ref, v_ref = [
        rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in (q, k, v)
    ]
    expected = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=mask,
        scale=softmax_scale,
    )
    expected = rearrange(expected, "b h s d -> (b s) h d")

    torch.testing.assert_close(output, expected)
    fused_attention.assert_not_called()


def test_ascend_attention_unmasked_inputs_keep_fused_path(
    monkeypatch,
    npu_platform,
):
    bsz, seq_len, num_heads, head_dim = 1, 2, 2, 8
    q, k, v = [torch.randn(bsz * seq_len, num_heads, head_dim) for _ in range(3)]
    expected = torch.randn_like(q)
    fused_attention = Mock(return_value=(expected, None))
    monkeypatch.setattr(
        vision,
        "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=fused_attention),
        raising=False,
    )
    backend = vision.VisionAscendAttention(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )
    sdpa_forward = Mock(
        side_effect=AssertionError("unmasked inputs must keep Ascend fused attention")
    )
    monkeypatch.setattr(backend.sdpa_fallback, "forward", sdpa_forward)

    output = backend(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32),
        bsz=bsz,
        seq_len=seq_len,
    )

    torch.testing.assert_close(output, expected)
    fused_attention.assert_called_once()
    sdpa_forward.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
