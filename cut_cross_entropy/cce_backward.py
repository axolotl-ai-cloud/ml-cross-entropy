# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import cce_backward_autotune
from cut_cross_entropy.tl_utils import (
    b_bin_fn,
    is_triton_greater_or_equal_3_2_0,
    tl_and_reduce_fn,
    tl_lock_add,
    tl_lock_kahan_sum,
    tl_softcapping,
    tl_softcapping_grad,
)
from cut_cross_entropy.vocab_parallel.utils import vp_reduce_e_grad


@triton.jit
def _mm_backward(
    do,
    dA,
    dAC,
    a_rows,
    stride_ab,
    stride_ad,
    partial_mask_a,
    da_lock_ptr,
    n_locks,
    Bp,
    b_rows,
    stride_bb,
    stride_bd,
    partial_mask_b,
    D,
    dA2,
    dA2C,
    a2_rows,
    stride_a2b,
    stride_a2d,
    B2p,
    b2_rows,
    stride_b2b,
    stride_b2d,
    D2,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    USE_KAHAN: tl.constexpr,
    HAS_MAIN: tl.constexpr,
    HAS_SECOND: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """dA += do @ B over BLOCK_D column tiles, then the same for the second operand pair.

    The second pair (D2 padded to whole tiles) rides on the tail of the same loop instead
    of a separate helper instance: three or more inlined copies of this loop trip an MLIR
    dominance error in Triton's layout-conversion pass.
    """
    d_inds = tl.arange(0, BLOCK_D).to(tl.int64)

    if HAS_MAIN:
        n_main = tl.cdiv(D, BLOCK_D)
    else:
        n_main = 0
    if HAS_SECOND:
        n_total = n_main + D2 // BLOCK_D
    else:
        n_total = n_main

    for d in range(0, n_total):
        if HAS_MAIN and HAS_SECOND:
            is_main = d < n_main
            d_local = tl.where(is_main, d, d - n_main)
            cols = d_local * BLOCK_D + d_inds
            b_ptrs = (
                tl.where(is_main, Bp, B2p)
                + tl.where(is_main, b_rows, b2_rows)[:, None]
                * tl.where(is_main, stride_bb, stride_b2b)
                + cols[None, :] * tl.where(is_main, stride_bd, stride_b2d)
            )
            a_off = tl.where(is_main, a_rows, a2_rows)[:, None] * tl.where(
                is_main, stride_ab, stride_a2b
            ) + cols[None, :] * tl.where(is_main, stride_ad, stride_a2d)
            da_ptrs = tl.where(is_main, dA, dA2) + a_off
            if USE_KAHAN:
                dac_ptrs = tl.where(is_main, dAC, dA2C) + a_off
            if EVEN_D:
                col_mask = d_inds[None, :] < BLOCK_D
            else:
                col_mask = d_inds[None, :] < tl.where(is_main, D - d * BLOCK_D, BLOCK_D)
            lock_offset = tl.where(d < n_main, d // tl.cdiv(D, BLOCK_D * n_locks), n_locks - 1)
        elif HAS_MAIN:
            cols = d * BLOCK_D + d_inds
            b_ptrs = Bp + b_rows[:, None] * stride_bb + cols[None, :] * stride_bd
            a_off = a_rows[:, None] * stride_ab + cols[None, :] * stride_ad
            da_ptrs = dA + a_off
            if USE_KAHAN:
                dac_ptrs = dAC + a_off
            if EVEN_D:
                col_mask = d_inds[None, :] < BLOCK_D
            else:
                col_mask = d_inds[None, :] < (D - d * BLOCK_D)
            lock_offset = d // tl.cdiv(D, BLOCK_D * n_locks)
        else:
            cols = d * BLOCK_D + d_inds
            b_ptrs = B2p + b2_rows[:, None] * stride_b2b + cols[None, :] * stride_b2d
            a_off = a2_rows[:, None] * stride_a2b + cols[None, :] * stride_a2d
            da_ptrs = dA2 + a_off
            if USE_KAHAN:
                dac_ptrs = dA2C + a_off
            col_mask = d_inds[None, :] < BLOCK_D
            lock_offset = d // tl.cdiv(D2, BLOCK_D * n_locks)

        b = tl.load(b_ptrs, mask=partial_mask_b & col_mask, other=0.0)

        da_i = tl.dot(do, b, input_precision=DOT_PRECISION).to(da_ptrs.dtype.element_ty)

        this_da_lock_ptr = da_lock_ptr + lock_offset

        if USE_KAHAN:
            tl_lock_kahan_sum(da_ptrs, dac_ptrs, da_i, partial_mask_a & col_mask, this_da_lock_ptr)
        else:
            tl_lock_add(da_ptrs, da_i, partial_mask_a & col_mask, this_da_lock_ptr)


@triton.jit
def _block_is_filtered(check_val: tl.tensor, filter_eps: tl.tensor) -> tl.tensor:
    return tl.reduce(check_val < filter_eps, None, tl_and_reduce_fn)


def _cce_backward_kernel(
    E,
    C,
    E2,
    C2,
    Bias,
    LSE,
    dOut,
    grad_scale,
    Valids,
    VocabOrdering,
    softcap,
    Targets,
    dE,
    dEC,
    dELocks,
    dC,
    dCC,
    dCLocks,
    dBias,
    dE2,
    dE2C,
    dC2,
    dC2C,
    B,
    D,
    D2,
    V,
    BMax,
    VStart,
    VCount,
    n_de_locks_0,
    n_de_locks_1,
    n_dc_locks_0,
    n_dc_locks_1,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_dcv,
    stride_dcd,
    stride_e2b,
    stride_e2d,
    stride_c2v,
    stride_c2d,
    stride_dc2v,
    stride_dc2d,
    stride_biasv,
    stride_vb,
    filter_eps,
    shift,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MM_BACK_BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    MM_BACK_EVEN_D: tl.constexpr,
    ITEM_DO: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_VOCAB_ORDERING: tl.constexpr,
    FILTER_E_GRAD: tl.constexpr,
    FILTER_C_GRAD: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    KAHAN_E: tl.constexpr,
    KAHAN_C: tl.constexpr,
    COMPUTE_DC: tl.constexpr,
    COMPUTE_DE: tl.constexpr,
    COMPUTE_DBIAS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    CHUNKED_C: tl.constexpr,
    HAS_E2: tl.constexpr,
    EVEN_D2: tl.constexpr,
    COMPUTE_DE2: tl.constexpr,
    COMPUTE_DC2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(VCount, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_v_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_v_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(tl.int64)

    local_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_v = local_v + VStart
    if HAS_VOCAB_ORDERING:
        offs_v = tl.load(VocabOrdering + offs_v, mask=offs_v < V, other=V).to(tl.int64)

    if CHUNKED_C:
        offs_v = tl.where(local_v < VCount, offs_v, V)

    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))

        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    if HAS_E2:
        e2_ptrs = E2 + (offs_b[:, None] * stride_e2b + offs_d[None, :] * stride_e2d)
        c2_ptrs = C2 + (offs_v[None, :] * stride_c2v + offs_d[:, None] * stride_c2d)
        for d in range(0, tl.cdiv(D2, BLOCK_D)):
            e_mask = offs_b[:, None] < BMax
            if not EVEN_D2:
                e_mask = e_mask & (offs_d[None, :] < (D2 - d * BLOCK_D))

            e2 = tl.load(e2_ptrs, mask=e_mask, other=0.0)

            c_mask = offs_v[None, :] < V
            if not EVEN_D2:
                c_mask = c_mask & (offs_d[:, None] < (D2 - d * BLOCK_D))

            c2 = tl.load(c2_ptrs, mask=c_mask, other=0.0)

            accum = tl.dot(e2, c2, accum, input_precision=DOT_PRECISION)

            e2_ptrs += BLOCK_D * stride_e2d
            c2_ptrs += BLOCK_D * stride_c2d

    tl.debug_barrier()

    if HAS_BIAS:
        bias = tl.load(Bias + offs_v * stride_biasv, mask=offs_v < V, other=0.0)
        bias = bias.to(dtype=accum.dtype)
        accum += bias[None, :]

    if HAS_SOFTCAP:
        accum = tl_softcapping(accum, softcap)

    if HAS_VALIDS:
        direct_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
        lse = tl.load(LSE + direct_offs_b, mask=direct_offs_b < B, other=float("inf"))
    else:
        lse = tl.load(LSE + offs_b, mask=offs_b < B, other=float("inf"))

    d_accum = tl.exp(accum - lse[:, None])
    d_accum = tl.where(offs_v[None, :] < V, d_accum, 0.0)

    if HAS_TARGETS:
        if HAS_SHIFT:
            target_offs_b = offs_b + shift
        else:
            target_offs_b = offs_b

        targets = tl.load(Targets + target_offs_b, mask=target_offs_b < BMax, other=V + 1)
        is_target = targets[:, None] == offs_v[None, :]
        d_accum += tl.where(is_target, -1.0, 0.0)
    else:
        is_target = None

    should_skip = False
    if (FILTER_E_GRAD and (COMPUTE_DE or COMPUTE_DE2)) and (
        FILTER_C_GRAD and (COMPUTE_DC or COMPUTE_DC2)
    ):
        if _block_is_filtered(tl.abs(d_accum), filter_eps):
            return
    elif (FILTER_E_GRAD and (COMPUTE_DE or COMPUTE_DE2)) or (
        FILTER_C_GRAD and (COMPUTE_DC or COMPUTE_DC2)
    ):
        should_skip = _block_is_filtered(tl.abs(d_accum), filter_eps)

    if HAS_SOFTCAP:
        d_accum = tl_softcapping_grad(d_accum, accum, softcap)

    if ITEM_DO:
        d_out = tl.load(dOut)
    else:
        if HAS_SHIFT:
            d_out_offs_b = offs_b + shift
        else:
            d_out_offs_b = offs_b

        d_out = tl.load(dOut + d_out_offs_b, mask=d_out_offs_b < BMax, other=0.0)[:, None]

    d_out = grad_scale * d_out

    d_accum = d_accum * d_out

    if COMPUTE_DBIAS:
        tl.atomic_add(dBias + offs_v * stride_biasv, tl.sum(d_accum, 0), mask=offs_v < V)

    d_accum = d_accum.to(e_ptrs.dtype.element_ty)

    if COMPUTE_DE or COMPUTE_DE2:
        if FILTER_E_GRAD:
            should_skip_e = should_skip
        else:
            should_skip_e = False

        if not should_skip_e:
            lock_offset = (pid_b // tl.cdiv(B, BLOCK_B * n_de_locks_0)) * n_de_locks_1

            _mm_backward(
                d_accum,
                dE,
                dEC,
                offs_b,
                stride_eb,
                stride_ed,
                offs_b[:, None] < BMax,
                dELocks + lock_offset,
                n_de_locks_1,
                C,
                offs_v,
                stride_cv,
                stride_cd,
                offs_v[:, None] < V,
                D,
                dE2,
                dE2C,
                offs_b,
                stride_e2b,
                stride_e2d,
                C2,
                offs_v,
                stride_c2v,
                stride_c2d,
                D2,
                MM_BACK_BLOCK_D,
                MM_BACK_EVEN_D,
                KAHAN_E,
                COMPUTE_DE,
                COMPUTE_DE2,
                DOT_PRECISION,
            )

    if COMPUTE_DC or COMPUTE_DC2:
        if FILTER_C_GRAD:
            should_skip_c = should_skip
        else:
            should_skip_c = False

        if not should_skip_c:
            lock_offset = (pid_v // tl.cdiv(VCount, BLOCK_V * n_dc_locks_0)) * n_dc_locks_1
            dc_v = local_v if CHUNKED_C else offs_v

            # dC2 rows are global vocab rows even when dC is chunked.
            _mm_backward(
                tl.trans(d_accum),
                dC,
                dCC,
                dc_v,
                stride_dcv,
                stride_dcd,
                offs_v[:, None] < V,
                dCLocks + lock_offset,
                n_dc_locks_1,
                E,
                offs_b,
                stride_eb,
                stride_ed,
                offs_b[:, None] < BMax,
                D,
                dC2,
                dC2C,
                offs_v,
                stride_dc2v,
                stride_dc2d,
                E2,
                offs_b,
                stride_e2b,
                stride_e2d,
                D2,
                MM_BACK_BLOCK_D,
                MM_BACK_EVEN_D,
                KAHAN_C,
                COMPUTE_DC,
                COMPUTE_DC2,
                DOT_PRECISION,
            )


def _cce_back_block_d(args) -> int:
    block_d = args["BLOCK_D"]
    return 2 * block_d


try:
    USE_TF32 = torch.get_float32_matmul_precision() == "high"
except RuntimeError:
    USE_TF32 = (
        torch.backends.cuda.matmul.fp32_precision == "tf32"
        or torch.backends.fp32_precision == "tf32"
    )

_cce_backward_kernel = triton.jit(_cce_backward_kernel)
_cce_backward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "MM_BACK_BLOCK_D": lambda args: _cce_back_block_d(args),
        "MM_BACK_EVEN_D": lambda args: (args["D"] % _cce_back_block_d(args)) == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VOCAB_ORDERING": lambda args: args["VocabOrdering"] is not None,
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_SHIFT": lambda args: args["shift"] != 0,
        "ITEM_DO": lambda args: args["dOut"].numel() == 1,
        "GROUP_B": lambda args: 8,
        "COMPUTE_DC": lambda args: args["dC"] is not None,
        "COMPUTE_DE": lambda args: args["dE"] is not None,
        "COMPUTE_DBIAS": lambda args: args["dBias"] is not None,
        "DOT_PRECISION": lambda args: "tf32" if USE_TF32 else "ieee",
        "HAS_E2": lambda args: args["E2"] is not None,
        "EVEN_D2": lambda args: (args["D2"] % args["BLOCK_D"]) == 0,
        "COMPUTE_DE2": lambda args: args["dE2"] is not None,
        "COMPUTE_DC2": lambda args: args["dC2"] is not None,
        "KAHAN_E": lambda args: args["dEC"] is not None or args["dE2C"] is not None,
        "KAHAN_C": lambda args: args["dCC"] is not None or args["dC2C"] is not None,
    }
)(_cce_backward_kernel)
_cce_backward_kernel = cce_backward_autotune()(_cce_backward_kernel)  # type: ignore


def _second_operand_multiple() -> int:
    # The backward tile is 2 * BLOCK_D: 64 for the default config, up to 256 under autotune.
    from cut_cross_entropy.tl_autotune import _AUTOTUNE

    return 256 if _AUTOTUNE else 64


def _pad_second_operand(e2: torch.Tensor, c2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad the second operand pair to a whole backward tile.

    ``dE2`` / ``dC2`` use a mask-free tiled helper, so the column count must be a multiple
    of the tile. Zero columns do not change the logits.
    """
    multiple = _second_operand_multiple()
    rem = e2.size(-1) % multiple
    if rem == 0:
        return e2, c2
    pad = multiple - rem
    return torch.nn.functional.pad(e2, (0, pad)), torch.nn.functional.pad(c2, (0, pad))


def cce_backward_kernel(
    do: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    lse: torch.Tensor,
    valids: torch.Tensor | None,
    softcap: float | None,
    filter_eps: float | None,
    targets: torch.Tensor | None = None,
    shift: int = 0,
    vocab_ordering: torch.Tensor | None = None,
    grad_scale: float = 1.0,
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    reduce_e_grad: bool = False,
    pg: torch.distributed.ProcessGroup | None = None,
    c_grad_chunk_size: int = 0,
    e2: torch.Tensor | None = None,
    c2: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    assert do.numel() in (e.size(0), 1)
    assert c.size(1) == e.size(1)
    if e2 is not None:
        assert c2 is not None
        assert e2.size(0) == e.size(0) and c2.size(0) == c.size(0)
        assert e2.size(1) == c2.size(1)
        assert e2.dtype == e.dtype and c2.dtype == c.dtype
        d2 = e2.size(1)
        e2_requires_grad, c2_requires_grad = e2.requires_grad, c2.requires_grad
        e2, c2 = _pad_second_operand(e2, c2)
        e2 = e2.contiguous()
    else:
        assert c2 is None
    assert lse.size(0) == e.size(0) or (valids is not None and lse.size(0) == valids.size(0))
    assert e.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "Backwards requires embeddings to be bf16 or fp16"
    assert c.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "Backwards requires classifier to be bf16 or fp16"

    do = do.contiguous()
    lse = lse.contiguous()

    can_use_fp32_accum = is_triton_greater_or_equal_3_2_0()

    de_dtype = torch.float32 if (accum_e_fp32 and can_use_fp32_accum) else None
    de = torch.zeros_like(e, dtype=de_dtype) if e.requires_grad else None

    assert c_grad_chunk_size >= 0 and c_grad_chunk_size % 128 == 0, (
        "c_grad_chunk_size must be zero or a positive multiple of 128"
    )
    chunked_c = c_grad_chunk_size > 0 and c.requires_grad and c_grad_chunk_size < c.size(0)
    chunk_size = c_grad_chunk_size if chunked_c else c.size(0)
    dc_dtype = torch.float32 if (accum_c_fp32 and can_use_fp32_accum) else None
    if chunked_c:
        assert dc_dtype is torch.float32, (
            "Chunked classifier accumulation requires accum_c_fp32=True and Triton >= 3.2"
        )
        # Reusable fp32 scratch covering one vocabulary chunk. The final gradient
        # is assembled chunk by chunk in the classifier's dtype and layout.
        dc = torch.empty((chunk_size, c.size(1)), device=c.device, dtype=dc_dtype)
        dc_output = torch.empty_like(c)
    else:
        dc = torch.zeros_like(c, dtype=dc_dtype) if c.requires_grad else None

    accum_e_fp32 = accum_e_fp32 and de is not None
    accum_c_fp32 = accum_c_fp32 and dc is not None

    if bias is not None:
        dbias = torch.zeros_like(bias, dtype=torch.float32) if bias.requires_grad else None
    else:
        dbias = None

    if de is not None:
        assert de.stride() == e.stride()

    if dc is not None:
        if chunked_c:
            assert dc.is_contiguous()
        else:
            assert dc.stride() == c.stride()

    if dbias is not None:
        assert bias is not None
        assert dbias.stride() == bias.stride()

    # The second operand pair is small (LoRA rank), so its gradients follow the same
    # accumulation policy as e / c but are never chunked.
    de2 = dc2 = de2c = dc2c = None
    if e2 is not None:
        assert c2 is not None
        if e2_requires_grad:
            de2 = torch.zeros_like(e2, dtype=de_dtype)
            if accum_e_fp32 and not can_use_fp32_accum:
                de2c = torch.zeros_like(e2)
        if c2_requires_grad:
            dc2 = torch.zeros_like(c2, dtype=dc_dtype)
            if accum_c_fp32 and not can_use_fp32_accum:
                dc2c = torch.zeros_like(c2)

    if accum_e_fp32 and not can_use_fp32_accum:
        dec = torch.zeros_like(e) if de is not None else None
    else:
        dec = None

    if accum_c_fp32 and not can_use_fp32_accum:
        dcc = torch.zeros_like(c) if dc is not None else None
    else:
        dcc = None

    if dec is not None:
        assert dec.stride() == e.stride()

    if dcc is not None:
        assert dcc.stride() == c.stride()

    if valids is not None:
        assert valids.ndim == 1
        B = valids.size(0)
    else:
        B = e.size(0)

    if do.numel() > 1:
        do = do.contiguous()
        lse = lse.contiguous()
        assert do.stride(0) == lse.stride(0), f"{do.stride()=}, {lse.stride()=}"

    dc_strides = (dc.stride(0), dc.stride(1)) if dc is not None else (c.stride(0), c.stride(1))

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(v_count, META["BLOCK_V"]),)

    if vocab_ordering is not None:
        assert vocab_ordering.ndim == 1
        assert vocab_ordering.numel() == c.size(0)
        assert vocab_ordering.stride(0) == 1

    nd_locks = triton.cdiv(c.size(1), 64)
    if de is not None or de2 is not None:
        de_locks = e.new_zeros((triton.cdiv(B, 128), nd_locks), dtype=torch.int32)
        de_lock_sizes = de_locks.size()
    else:
        de_locks = None
        de_lock_sizes = (None, None)

    if dc is not None or dc2 is not None:
        dc_locks = c.new_zeros((triton.cdiv(chunk_size, 128), nd_locks), dtype=torch.int32)
        dc_lock_sizes = dc_locks.size()
    else:
        dc_locks = None
        dc_lock_sizes = (None, None)

    D2 = 0 if c2 is None else c2.size(1)
    e2_strides = (e2.stride(0), e2.stride(1)) if e2 is not None else (1, 1)
    c2_strides = (c2.stride(0), c2.stride(1)) if c2 is not None else (1, 1)
    dc2_strides = (dc2.stride(0), dc2.stride(1)) if dc2 is not None else c2_strides

    for v_start in range(0, c.size(0), chunk_size):
        v_count = min(chunk_size, c.size(0) - v_start)
        if chunked_c:
            dc.zero_()
        _cce_backward_kernel[grid](
            e,
            c,
            e2,
            c2,
            bias,
            lse,
            do,
            grad_scale,
            valids,
            vocab_ordering,
            softcap,
            targets,
            de,
            dec,
            de_locks,
            dc,
            dcc,
            dc_locks,
            dbias,
            de2,
            de2c,
            dc2,
            dc2c,
            B,
            e.size(1),
            D2,
            c.size(0),
            e.size(0),
            v_start,
            v_count,
            *de_lock_sizes,
            *dc_lock_sizes,
            e.stride(0),
            e.stride(1),
            c.stride(0),
            c.stride(1),
            *dc_strides,
            *e2_strides,
            *c2_strides,
            *dc2_strides,
            1 if bias is None else bias.stride(0),
            1 if valids is None else valids.stride(0),
            filter_eps,
            shift=shift,
            B_BIN=b_bin_fn(B),
            FILTER_E_GRAD=filter_e_grad and (de is not None or de2 is not None),
            FILTER_C_GRAD=filter_c_grad and (dc is not None or dc2 is not None),
            CHUNKED_C=chunked_c,
        )
        if chunked_c:
            # Cast this chunk into its final rows. With a vocab ordering, scratch row i
            # holds the gradient for classifier row vocab_ordering[v_start + i].
            rows = slice(v_start, v_start + v_count)
            if vocab_ordering is None:
                dc_output[rows].copy_(dc[:v_count])
            else:
                dc_output.index_copy_(
                    0, vocab_ordering[rows].to(torch.int64), dc[:v_count].to(c.dtype)
                )
    if chunked_c:
        dc = dc_output

    if reduce_e_grad and de is not None:
        de = vp_reduce_e_grad(de, pg)
    if reduce_e_grad and de2 is not None:
        de2 = vp_reduce_e_grad(de2, pg)

    if dbias is not None:
        assert bias is not None
        dbias = dbias.to(dtype=bias.dtype)

    if dc is not None:
        dc = dc.to(dtype=c.dtype)

    if de is not None:
        de = de.to(dtype=e.dtype)

    if de2 is not None:
        assert e2 is not None
        de2 = de2[:, :d2].to(dtype=e2.dtype)

    if dc2 is not None:
        assert c2 is not None
        dc2 = dc2[:, :d2].to(dtype=c2.dtype)

    return de, dc, dbias, de2, dc2
