# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

def attention_backward(ctx, dout, _dlse):
    from fla.ops.phba.backward_kernels import compute_delta, direct_sparse_backward, merge_gradient
    from fla.ops.phba.contracts import prepare_grad_output
    from fla.ops.phba.flash_attention import local_backward
    from fla.ops.phba.metadata import BranchMetadata, DirectMetadata

    qr, qw, kr, v, pk, pv, out, lse, bp, valid, sc, so, sr, *token = ctx.saved_tensors
    c, bc, bq = ctx.settings
    rtok, rstate = token[-2:]
    token = token[:-2]
    state = BranchMetadata(sc, so, sr, sr.numel())
    tok = BranchMetadata(*token, token[-1].numel()) if ctx.has_token else None
    metadata = DirectMetadata(tok, state, int(ctx.has_token))
    dout, _copy_bytes = prepare_grad_output(dout, out)
    dout = dout.contiguous()
    delta = compute_delta(dout, out)
    local = local_backward(dout, qr, kr, v, out, lse, bp, ctx.batches, deterministic=True)
    sparse = direct_sparse_backward(qr, qw, kr, v, pk, pv, dout, lse, delta, metadata, bp, valid,
                                    chunk_size=c, bc=bc, bq=bq, rtok=rtok, rstate=rstate,
                                    tape_plan=getattr(ctx, 'owner_tape_plan', None))
    dq = merge_gradient(local[0], sparse.dq_rope, output_dtype=qr.dtype)
    dk = merge_gradient(local[1], sparse.dk_rope, output_dtype=kr.dtype)
    dv = merge_gradient(local[2], sparse.dv, output_dtype=v.dtype)
    return dq.to(qr.dtype), sparse.dq_raw.to(qw.dtype), dk.to(kr.dtype), dv.to(v.dtype), \
        sparse.dpk.to(pk.dtype), sparse.dpv.to(pv.dtype), \
        None, None, None, None, None
