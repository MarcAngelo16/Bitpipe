# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""
Iteration Context Manager for BitPipe Profiling

This module manages the current training/evaluation iteration context
to enable selective profiling of specific iterations.
"""

from megatron import get_args
from megatron import print_rank_0

# Global iteration state
_current_iteration = 1
_current_type = 'train'
_validation_done = False

def set_iteration_context(iteration, iter_type):
    """Set the current iteration context.
    
    Args:
        iteration (int): Current iteration number (1-indexed)
        iter_type (str): 'train' or 'eval'
    """
    global _current_iteration, _current_type
    _current_iteration = iteration
    _current_type = iter_type

def get_iteration_context():
    """Get the current iteration context.
    
    Returns:
        tuple: (iteration, iter_type)
    """
    return _current_iteration, _current_type

def _validate_iteration_ranges():
    """Validate that requested iterations don't exceed limits.
    
    Returns:
        tuple: (valid_train_iters, valid_eval_iters, has_warnings)
    """
    args = get_args()
    
    train_iters = getattr(args, 'profile_train_iters', [])
    eval_iters = getattr(args, 'profile_eval_iters', [])
    max_train_iters = getattr(args, 'train_iters', 1)
    max_eval_iters = getattr(args, 'eval_iters', 1)
    
    valid_train_iters = []
    valid_eval_iters = []
    has_warnings = False
    
    # Validate training iterations
    for iter_num in train_iters:
        if iter_num > max_train_iters or iter_num < 1:
            print_rank_0(f"WARNING: Requested train iteration {iter_num} is invalid "
                         f"(valid range: 1-{max_train_iters}). Skipping.")
            has_warnings = True
        else:
            valid_train_iters.append(iter_num)
    
    # Validate evaluation iterations  
    for iter_num in eval_iters:
        if iter_num > max_eval_iters or iter_num < 1:
            print_rank_0(f"WARNING: Requested eval iteration {iter_num} is invalid "
                         f"(valid range: 1-{max_eval_iters}). Skipping.")
            has_warnings = True
        else:
            valid_eval_iters.append(iter_num)
    
    return valid_train_iters, valid_eval_iters, has_warnings

def should_profile_current_iteration():
    """Check if the current iteration should be profiled.
    
    Returns:
        bool: True if current iteration should be profiled
    """
    args = get_args()
    
    # If BitPipe profiling is not enabled, don't profile
    if not getattr(args, 'enable_profiling', False):
        return False
    
    iteration, iter_type = get_iteration_context()
    
    # Get validated iteration lists
    valid_train_iters, valid_eval_iters, has_warnings = _validate_iteration_ranges()
    
    # Get original requested lists for default behavior
    train_iters = getattr(args, 'profile_train_iters', [])
    eval_iters = getattr(args, 'profile_eval_iters', [])
    
    # Determine what to profile
    should_profile = False
    
    if iter_type == 'train':
        if train_iters:  # User specified train iterations
            if valid_train_iters:  # Some valid iterations exist
                should_profile = iteration in valid_train_iters
            else:  # All iterations were invalid, fall back to default
                should_profile = iteration == 1
                if has_warnings and iteration == 1:
                    print_rank_0("WARNING: All requested train iterations were invalid. "
                                "Falling back to default (iteration 1).")
        elif not train_iters and not eval_iters:
            # Neither specified, use default
            should_profile = iteration == 1
        # If only eval_iters specified but not train_iters, don't profile training
    
    elif iter_type == 'eval':
        if eval_iters:  # User specified eval iterations
            if valid_eval_iters:  # Some valid iterations exist
                should_profile = iteration in valid_eval_iters
            else:  # All iterations were invalid, fall back to default
                should_profile = iteration == 1
                if has_warnings and iteration == 1:
                    print_rank_0("WARNING: All requested eval iterations were invalid. "
                                "Falling back to default (iteration 1).")
        elif not train_iters and not eval_iters:
            # Neither specified, use default
            should_profile = iteration == 1
        # If only train_iters specified but not eval_iters, don't profile evaluation
    
    if should_profile:
        print_rank_0(f"BitPipe profiling ENABLED for {iter_type} iteration {iteration}")
    
    return should_profile