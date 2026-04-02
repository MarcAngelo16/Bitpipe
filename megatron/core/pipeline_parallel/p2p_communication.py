# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import operator
import os
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import torch

from megatron import core
from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
)

# Types
Shape = Union[List[int], torch.Size]


def _communicate_shapes(tensor_send_next, tensor_send_prev, recv_prev, recv_next, config):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )

    if config.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(
            tensor_send_prev=send_prev_shape_tensor,
            tensor_recv_prev=recv_prev_shape_tensor,
            tensor_send_next=send_next_shape_tensor,
            tensor_recv_next=recv_next_shape_tensor,
            group=get_pipeline_model_parallel_group(),
        )
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup
):
    # DEBUG: Log actual P2P operations with ACTUAL ranks if CHIMERA_DEBUG is set
    import os
    if os.environ.get('CHIMERA_DEBUG', '0') == '1':
        my_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        prev_rank = get_pipeline_model_parallel_prev_rank()
        next_rank = get_pipeline_model_parallel_next_rank()
        ops_desc = []
        if tensor_send_prev is not None:
            ops_desc.append(f"send→R{prev_rank}")
        if tensor_send_next is not None:
            ops_desc.append(f"send→R{next_rank}")
        if tensor_recv_prev is not None:
            ops_desc.append(f"recv←R{prev_rank}")
        if tensor_recv_next is not None:
            ops_desc.append(f"recv←R{next_rank}")
        if ops_desc:
            print(f"[P2P ACTUAL] R{my_rank}: {', '.join(ops_desc)}", flush=True)

    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_next,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_next,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup
):
    reqs = []
    rank = get_pipeline_model_parallel_rank()
    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(send_prev_req)
    return reqs


def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    wait_on_reqs: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Arguments:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if not config.variable_seq_lengths:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
    else:
        recv_prev_shape, recv_next_shape = _communicate_shapes(
            tensor_send_next, tensor_send_prev, recv_prev, recv_next, config
        )

    if recv_prev:
        if config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev = torch.empty(
            recv_prev_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )
    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next = torch.empty(
            recv_next_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        assert wait_on_reqs
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    reqs = p2p_func(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=get_pipeline_model_parallel_group(),
    )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs


def _communicate2F(*, tensor_send_next: Optional[torch.Tensor],
                 tensor_send_prev: Optional[torch.Tensor],
                 recv_prev: bool,
                 recv_next: bool,
                 tensor_shape: Shape,
                 tensor_shape_g: Shape,
                 config: ModelParallelConfig,
                 wait_on_reqs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Arguments:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if not config.variable_seq_lengths:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape_g
    else:
        recv_prev_shape, recv_next_shape = \
            _communicate_shapes(tensor_send_next, tensor_send_prev,
                                recv_prev, recv_next, config)

    if recv_prev:
        if config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev = torch.empty(recv_prev_shape,
                                       requires_grad=True,
                                       device=torch.cuda.current_device(),
                                       dtype=config.pipeline_dtype)
    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape_g is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next = torch.empty(recv_next_shape,
                                       requires_grad=True,
                                       device=torch.cuda.current_device(),
                                       dtype=config.pipeline_dtype)

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:
        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []
        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        assert wait_on_reqs
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    reqs = p2p_func(tensor_send_prev=tensor_send_prev,
                    tensor_recv_prev=tensor_recv_prev,
                    tensor_send_next=tensor_send_next,
                    tensor_recv_next=tensor_recv_next,
                    group=get_pipeline_model_parallel_group())

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs


def recv_forward(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).


    See _communicate for argument details.
    """

    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-recv').stop()
    return input_tensor


def recv_backward(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-recv').stop()
    return output_tensor_grad


def recv_backward_bd(recv_prev: bool,tensor_shape: Shape,
                  config: ModelParallelConfig) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config)
        if config.timers is not None:
            config.timers('backward-recv').stop()
    return output_tensor_grad


def send_forward(output_tensor: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    if not core.parallel_state.is_pipeline_last_stage():
        if config.timers is not None:
            config.timers('forward-send', log_level=2).start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-send').stop()


def send_backward(input_tensor_grad: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not core.parallel_state.is_pipeline_first_stage():
        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-send').stop()


def send_forward_recv_backward(
    output_tensor: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('forward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('backward-send-forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-send-forward-recv', log_level=2).start()
    input_tensor, _, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config,
    )
    if config.timers is not None:
        config.timers('forward-send-forward-recv').stop()
    if overlap_p2p_comm:
        return input_tensor, wait_handles
    return input_tensor


def send_forward_recv_forward_bd0(output_tensor: torch.Tensor,
                              recv_next: bool,
                              tensor_shape: Shape,
                              config: ModelParallelConfig,
                              overlap_p2p_comm: bool = False) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-send-forward-recv', log_level=2).start()
    _, input_tensor, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config)
    if config.timers is not None:
        config.timers('forward-send-forward-recv').stop()
    if overlap_p2p_comm:
        return input_tensor, wait_handles
    return input_tensor


def send_forward_recv_forward_bd(output_tensor: torch.Tensor,
                              recv_prev: bool,
                              recv_next: bool,
                              tensor_shape: Shape,
                              config: ModelParallelConfig,
                              overlap_p2p_comm: bool = False) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-send-forward-recv', log_level=2).start()
    input_tensor,input_tensor_n, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config)
    if config.timers is not None:
        config.timers('forward-send-forward-recv').stop()
    if overlap_p2p_comm:
        return input_tensor,input_tensor_n, wait_handles
    return input_tensor,input_tensor_n


def send_forward_recv_forward_bd1(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers("forward-send-forward-recv", log_level=2).start()
    input_tensor, _, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=output_tensor,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config,
    )
    if config.timers is not None:
        config.timers("forward-send-forward-recv").stop()
    if overlap_p2p_comm:
        return input_tensor, wait_handles
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('backward-send-backward-recv', log_level=2).start()
    _, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config,
    )
    if config.timers is not None:
        config.timers('backward-send-backward-recv').stop()
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles
    return output_tensor_grad


def send_backward_recv_backward_bd(input_tensor_grad: torch.Tensor,
                                recv_prev: bool,
                                tensor_shape: Shape,
                                config: ModelParallelConfig,
                                overlap_p2p_comm: bool = False) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('backward-send-backward-recv', log_level=2).start()
    output_tensor_grad, _, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config)
    if config.timers is not None:
        config.timers('backward-send-backward-recv').stop()
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles
    return output_tensor_grad


def send_backward_recv_backward_bd1(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers("backward-send-backward-recv", log_level=2).start()
    _, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=input_tensor_grad,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config,
    )
    if config.timers is not None:
        config.timers("backward-send-backward-recv").stop()
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
    input_tensor, output_tensor_grad, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad


def send_forward_backward_recv_forward_backward2F(
        output_tensor: torch.Tensor,
        input_tensor_grad: torch.Tensor,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        tensor_shape_g: Shape,
        config: ModelParallelConfig) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv',
               log_level=2).start()
    input_tensor, output_tensor_grad, _ = _communicate2F(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        tensor_shape_g=tensor_shape_g,
        config=config)
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad


def send_forward_backward_recv_forward_backward_bd(
        output_tensor: torch.Tensor,
        input_tensor_grad: torch.Tensor,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        config: ModelParallelConfig) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv',
               log_level=2).start()
    output_tensor_grad,input_tensor,  _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        config=config)
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad


def send_forward_backward_recv_forward_backward_bd2F(
        output_tensor: torch.Tensor,
        input_tensor_grad: torch.Tensor,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        tensor_shape_g: Shape,
        config: ModelParallelConfig) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv',
               log_level=2).start()
    output_tensor_grad,input_tensor,  _ = _communicate2F(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        tensor_shape_g=tensor_shape_g,
        config=config)
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad


def send_forward_backward_recv_forward_backward_bd1(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers(
            "forward-backward-send-forward-backward-recv", log_level=2
        ).start()
    input_tensor, output_tensor_grad, _ = _communicate(
        tensor_send_next=input_tensor_grad,
        tensor_send_prev=output_tensor,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers("forward-backward-send-forward-backward-recv").stop()
    return input_tensor, output_tensor_grad


def send_forward_backward_recv_forward_backward_bd2(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers(
            "forward-backward-send-forward-backward-recv", log_level=2
        ).start()
    output_tensor_grad, input_tensor, _ = _communicate(
        tensor_send_next=input_tensor_grad,
        tensor_send_prev=output_tensor,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers("forward-backward-send-forward-backward-recv").stop()
    return input_tensor, output_tensor_grad


# =============================================================================
# Chimera 2-VR P2P Communication Functions
# =============================================================================
#
# These functions are designed for Chimera's bidirectional pipeline where:
# - VR0 flows 0→1→2→3 (send to next, recv from prev)
# - VR1 flows 3→2→1→0 (send to prev, recv from next)
#
# Unlike standard P2P functions, these do NOT have built-in stage checks
# because Chimera's stage boundaries are VR-dependent:
# - VR0 first stage: Rank 0 (creates embedding)
# - VR0 last stage: Rank N-1 (creates output)
# - VR1 first stage: Rank N-1 (creates embedding)
# - VR1 last stage: Rank 0 (creates output)
#
# The caller (chimera_2vr.py) is responsible for checking stage boundaries.
# =============================================================================


def chimera_send_next_recv_prev(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera: Send to next rank AND receive from prev rank.

    Used for VR0→VR0 transitions (standard forward direction).
    No stage checks - caller handles boundary conditions.

    Args:
        output_tensor: Tensor to send to next rank
        tensor_shape: Shape for receiving tensor
        config: Model config

    Returns:
        Received tensor from prev rank
    """
    if config.timers is not None:
        config.timers('chimera-send-next-recv-prev', log_level=2).start()
    input_tensor, _, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-send-next-recv-prev').stop()
    return input_tensor


def chimera_send_next_recv_next(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera: Send to next rank AND receive from next rank.

    Used for VR0→VR1 transitions (bidirectional crossover).
    - Send VR0 activation to next rank
    - Receive VR1 activation from next rank

    No stage checks - caller handles boundary conditions.

    Args:
        output_tensor: Tensor to send to next rank (VR0 output)
        tensor_shape: Shape for receiving tensor (VR1 input)
        config: Model config

    Returns:
        Received tensor from next rank (VR1 activation)
    """
    if config.timers is not None:
        config.timers('chimera-send-next-recv-next', log_level=2).start()
    _, input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-send-next-recv-next').stop()
    return input_tensor


def chimera_send_prev_recv_prev(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera: Send to prev rank AND receive from prev rank.

    Used for VR1→VR0 transitions (bidirectional crossover).
    - Send VR1 activation to prev rank
    - Receive VR0 activation from prev rank

    No stage checks - caller handles boundary conditions.

    Args:
        output_tensor: Tensor to send to prev rank (VR1 output)
        tensor_shape: Shape for receiving tensor (VR0 input)
        config: Model config

    Returns:
        Received tensor from prev rank (VR0 activation)
    """
    if config.timers is not None:
        config.timers('chimera-send-prev-recv-prev', log_level=2).start()
    input_tensor, _, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=output_tensor,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-send-prev-recv-prev').stop()
    return input_tensor


def chimera_send_prev_recv_next(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera: Send to prev rank AND receive from next rank.

    Used for VR1→VR1 transitions (backward pipeline continuation).
    No stage checks - caller handles boundary conditions.

    Args:
        output_tensor: Tensor to send to prev rank (VR1 output)
        tensor_shape: Shape for receiving tensor (VR1 input)
        config: Model config

    Returns:
        Received tensor from next rank (VR1 activation)
    """
    if config.timers is not None:
        config.timers('chimera-send-prev-recv-next', log_level=2).start()
    _, input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=output_tensor,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-send-prev-recv-next').stop()
    return input_tensor


def chimera_send_next_only(
    output_tensor: torch.Tensor,
    config: ModelParallelConfig,
) -> None:
    """
    Chimera: Send to next rank only (no receive).

    Used by first stage of VR0 (Rank 0) when it doesn't need to receive.
    No stage checks - caller handles boundary conditions.

    Args:
        output_tensor: Tensor to send to next rank
        config: Model config
    """
    if config.timers is not None:
        config.timers('chimera-send-next-only', log_level=2).start()
    _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=None,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-send-next-only').stop()


def chimera_send_prev_only(
    output_tensor: torch.Tensor,
    config: ModelParallelConfig,
) -> None:
    """
    Chimera: Send to prev rank only (no receive).

    Used by first stage of VR1 (Rank N-1) when it doesn't need to receive.
    No stage checks - caller handles boundary conditions.

    Args:
        output_tensor: Tensor to send to prev rank
        config: Model config
    """
    if config.timers is not None:
        config.timers('chimera-send-prev-only', log_level=2).start()
    _communicate(
        tensor_send_next=None,
        tensor_send_prev=output_tensor,
        recv_prev=False,
        recv_next=False,
        tensor_shape=None,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-send-prev-only').stop()


def chimera_recv_prev_only(
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera: Receive from prev rank only (no send).

    Used for pre-receiving VR0 input at the start of execution.
    No stage checks - caller handles boundary conditions.

    Args:
        tensor_shape: Shape for receiving tensor
        config: Model config

    Returns:
        Received tensor from prev rank
    """
    if config.timers is not None:
        config.timers('chimera-recv-prev-only', log_level=2).start()
    input_tensor, _, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-recv-prev-only').stop()
    return input_tensor


def chimera_recv_next_only(
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera: Receive from next rank only (no send).

    Used for pre-receiving VR1 input at the start of execution.
    No stage checks - caller handles boundary conditions.

    Args:
        tensor_shape: Shape for receiving tensor
        config: Model config

    Returns:
        Received tensor from next rank
    """
    if config.timers is not None:
        config.timers('chimera-recv-next-only', log_level=2).start()
    _, input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-recv-next-only').stop()
    return input_tensor


# Backward gradient versions for Chimera
# Gradients flow opposite to activations:
# - VR0 activations: 0→1→2→3, gradients: 3→2→1→0
# - VR1 activations: 3→2→1→0, gradients: 0→1→2→3

def chimera_grad_send_prev_recv_next(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera gradient: Send to prev rank AND receive from next rank.

    Used for VR0 backward pass (gradients flow 3→2→1→0).
    No stage checks - caller handles boundary conditions.

    Args:
        input_tensor_grad: Gradient to send to prev rank
        tensor_shape: Shape for receiving gradient
        config: Model config

    Returns:
        Received gradient from next rank
    """
    if config.timers is not None:
        config.timers('chimera-grad-send-prev-recv-next', log_level=2).start()
    _, output_tensor_grad, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-grad-send-prev-recv-next').stop()
    return output_tensor_grad


def chimera_grad_send_next_recv_prev(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera gradient: Send to next rank AND receive from prev rank.

    Used for VR1 backward pass (gradients flow 0→1→2→3).
    No stage checks - caller handles boundary conditions.

    Args:
        input_tensor_grad: Gradient to send to next rank
        tensor_shape: Shape for receiving gradient
        config: Model config

    Returns:
        Received gradient from prev rank
    """
    if config.timers is not None:
        config.timers('chimera-grad-send-next-recv-prev', log_level=2).start()
    output_tensor_grad, _, _ = _communicate(
        tensor_send_next=input_tensor_grad,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-grad-send-next-recv-prev').stop()
    return output_tensor_grad


def chimera_grad_send_prev_recv_prev(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera gradient: Send to prev rank AND receive from prev rank.

    Used for VR0→VR1 transition in cooldown (send VR0 grad to prev, recv VR1 grad from prev).
    No stage checks - caller handles boundary conditions.

    Args:
        input_tensor_grad: Gradient to send to prev rank
        tensor_shape: Shape for receiving gradient
        config: Model config

    Returns:
        Received gradient from prev rank
    """
    if config.timers is not None:
        config.timers('chimera-grad-send-prev-recv-prev', log_level=2).start()
    output_tensor_grad, _, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-grad-send-prev-recv-prev').stop()
    return output_tensor_grad


def chimera_grad_send_next_recv_next(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera gradient: Send to next rank AND receive from next rank.

    Used for VR1→VR0 transition in cooldown (send VR1 grad to next, recv VR0 grad from next).
    No stage checks - caller handles boundary conditions.

    Args:
        input_tensor_grad: Gradient to send to next rank
        tensor_shape: Shape for receiving gradient
        config: Model config

    Returns:
        Received gradient from next rank
    """
    if config.timers is not None:
        config.timers('chimera-grad-send-next-recv-next', log_level=2).start()
    _, output_tensor_grad, _ = _communicate(
        tensor_send_next=input_tensor_grad,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-grad-send-next-recv-next').stop()
    return output_tensor_grad


def chimera_grad_send_prev_only(
    input_tensor_grad: torch.Tensor,
    config: ModelParallelConfig,
) -> None:
    """
    Chimera gradient: Send to prev rank only (no recv).

    Used for VR0 gradient send without receiving.
    No stage checks - caller handles boundary conditions.

    Args:
        input_tensor_grad: Gradient to send to prev rank
        config: Model config
    """
    if config.timers is not None:
        config.timers('chimera-grad-send-prev-only', log_level=2).start()
    _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=False,
        tensor_shape=None,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-grad-send-prev-only').stop()


def chimera_grad_send_next_only(
    input_tensor_grad: torch.Tensor,
    config: ModelParallelConfig,
) -> None:
    """
    Chimera gradient: Send to next rank only (no recv).

    Used for VR1 gradient send without receiving.
    No stage checks - caller handles boundary conditions.

    Args:
        input_tensor_grad: Gradient to send to next rank
        config: Model config
    """
    if config.timers is not None:
        config.timers('chimera-grad-send-next-only', log_level=2).start()
    _communicate(
        tensor_send_next=input_tensor_grad,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=None,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-grad-send-next-only').stop()


def chimera_grad_recv_next_only(
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera gradient: Receive from next rank only (no send).

    Used for VR0 gradient recv without sending.
    No stage checks - caller handles boundary conditions.

    Args:
        tensor_shape: Shape for receiving gradient
        config: Model config

    Returns:
        Received gradient from next rank
    """
    if config.timers is not None:
        config.timers('chimera-grad-recv-next-only', log_level=2).start()
    _, output_tensor_grad, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=True,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-grad-recv-next-only').stop()
    return output_tensor_grad


def chimera_grad_recv_prev_only(
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """
    Chimera gradient: Receive from prev rank only (no send).

    Used for VR1 gradient recv without sending.
    No stage checks - caller handles boundary conditions.

    Args:
        tensor_shape: Shape for receiving gradient
        config: Model config

    Returns:
        Received gradient from prev rank
    """
    if config.timers is not None:
        config.timers('chimera-grad-recv-prev-only', log_level=2).start()
    output_tensor_grad, _, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('chimera-grad-recv-prev-only').stop()
    return output_tensor_grad


def chimera_communicate(
    tensor_send_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Unified Chimera communication with flags (BitPipe-style).

    Handles all combinations of send/recv in a single function.
    Uses _communicate() directly to bypass stage checks.

    Args:
        tensor_send_prev: Tensor to send to previous rank (None = no send)
        tensor_send_next: Tensor to send to next rank (None = no send)
        recv_prev: Whether to receive from previous rank
        recv_next: Whether to receive from next rank
        tensor_shape: Shape for receiving tensors
        config: Model config
        dtype: Data type for received tensors (optional)

    Returns:
        (recv_prev_tensor, recv_next_tensor)
        - recv_prev_tensor: Tensor received from prev (None if recv_prev=False)
        - recv_next_tensor: Tensor received from next (None if recv_next=False)

    Examples:
        # VR0 grad: send_prev + recv_next
        _, grad_in = chimera_communicate(
            tensor_send_prev=grad_out, tensor_send_next=None,
            recv_prev=False, recv_next=True, ...
        )

        # VR1 grad: send_next + recv_prev
        grad_in, _ = chimera_communicate(
            tensor_send_prev=None, tensor_send_next=grad_out,
            recv_prev=True, recv_next=False, ...
        )

        # Send only (no recv)
        _, _ = chimera_communicate(
            tensor_send_prev=grad_out, tensor_send_next=None,
            recv_prev=False, recv_next=False, ...
        )

        # Recv only (no send)
        _, grad_in = chimera_communicate(
            tensor_send_prev=None, tensor_send_next=None,
            recv_prev=False, recv_next=True, ...
        )
    """
    # Debug logging
    if os.environ.get('CHIMERA_DEBUG', '0') == '1':
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        send_str = []
        if tensor_send_prev is not None:
            send_str.append("send_prev")
        if tensor_send_next is not None:
            send_str.append("send_next")

        recv_str = []
        if recv_prev:
            recv_str.append("recv_prev")
        if recv_next:
            recv_str.append("recv_next")

        ops = " + ".join(send_str + recv_str) if (send_str or recv_str) else "NO-OP"
        print(f"[Chimera P2P Rank {rank}] chimera_communicate: {ops}", flush=True)

    if config.timers is not None:
        config.timers('chimera-communicate', log_level=2).start()

    # Use _communicate directly (bypasses stage checks)
    recv_prev_tensor, recv_next_tensor, _ = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=True,
        config=config,
    )

    if config.timers is not None:
        config.timers('chimera-communicate').stop()

    # Debug logging for results
    if os.environ.get('CHIMERA_DEBUG', '0') == '1':
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        result_str = []
        if recv_prev_tensor is not None:
            result_str.append(f"recv_prev: shape={recv_prev_tensor.shape}")
        if recv_next_tensor is not None:
            result_str.append(f"recv_next: shape={recv_next_tensor.shape}")

        if result_str:
            print(f"[Chimera P2P Rank {rank}] chimera_communicate result: {', '.join(result_str)}", flush=True)

    return recv_prev_tensor, recv_next_tensor