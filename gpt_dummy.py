#!/usr/bin/env python
"""
BitPipe Pipeline Parallelism example for 4 GPUs using Megatron-LM
Uses BitPipe bidirectional schedule with NullTokenizer
Simplified GPT-style transformer for demonstration
"""

import os
import sys
import torch
import torch.nn as nn
from functools import partial

# Add Megatron to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from megatron import get_args, print_rank_0, get_tokenizer
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.training import pretrain
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.model.gpt_model import GPTModel
from megatron.utils import get_ltor_masks_and_position_ids, average_losses_across_data_parallel_group
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.arguments import core_transformer_config_from_args




def model_provider(pre_process=True, post_process=True):
    """Build the model for BitPipe."""
    print_rank_0('Building BitPipe GPT model ...')
    args = get_args()
    
    # Use the core transformer config from args (similar to pretrain_gpt.py)
    config = core_transformer_config_from_args(args)
    
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    
    return model


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset that generates random token sequences"""
    
    def __init__(self, seq_length, vocab_size, num_samples=10000):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate deterministic random tokens based on index
        # This ensures reproducibility
        torch.manual_seed(idx)
        tokens = torch.randint(0, self.vocab_size, 
                              (self.seq_length + 1,),  # +1 for labels
                              dtype=torch.long)
        return {'text': tokens}


def get_batch(data_iterator):
    """Generate a batch for BitPipe training."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    if data_b is not None and 'text' in data_b:
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
    else:
        # Generate synthetic data
        tokens = torch.randint(1, args.vocab_size, 
                              (args.micro_batch_size, args.seq_length), 
                              dtype=torch.long).cuda()
        labels = torch.randint(1, args.vocab_size, 
                              (args.micro_batch_size, args.seq_length), 
                              dtype=torch.long).cuda()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    """Loss function for language modeling."""
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step for BitPipe training."""
    
    # Get batch
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    
    # Forward pass through model
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    print_rank_0('> building dummy datasets for BitPipe...')
    
    args = get_args()
    
    # Create dummy datasets with appropriate sizes
    train_size = train_val_test_num_samples[0] if train_val_test_num_samples[0] is not None else 10000
    valid_size = train_val_test_num_samples[1] if train_val_test_num_samples[1] is not None else 1000
    test_size = train_val_test_num_samples[2] if train_val_test_num_samples[2] is not None else 1000
    
    train_dataset = DummyDataset(args.seq_length, args.vocab_size, train_size)
    valid_dataset = DummyDataset(args.seq_length, args.vocab_size, valid_size)
    test_dataset = DummyDataset(args.seq_length, args.vocab_size, test_size)
    
    print_rank_0(f'> created dummy datasets: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}')
    
    return train_dataset, valid_dataset, test_dataset


def extra_args_provider(parser):
    """Provide extra arguments specific to BitPipe demo."""
    group = parser.add_argument_group(title='bitpipe demo')
    group.add_argument('--train-synthetic', action='store_true',
                       help='Use synthetic data (default for this example)')
    return parser


if __name__ == '__main__':
    
    # Use pretrain wrapper which handles most initialization including BitPipe
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={
                 'train_synthetic': True,
                 'seq_length': 128,  # Reasonable sequence length
                 'max_position_embeddings': 128,
                 'hidden_size': 512,  # Reasonable model size
                 'num_layers': 16,     
                 'num_attention_heads': 8,
                 'vocab_size': 1000,  # Small vocab for dummy tokenizer
                 # BitPipe requires these to be set
                 'micro_batch_size': 2,
                 'global_batch_size': 16,  # Must be divisible by (num_microbatches * pipeline_size)
                 'enable_bitpipe_schedule': True,  # Enable BitPipe!
             },
             extra_args_provider=extra_args_provider)