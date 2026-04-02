#!/usr/bin/env python
"""
BERT Dummy Training Script for BitPipe Testing

This script provides a minimal BERT training setup for testing BitPipe
pipeline parallelism with synthetic data.
"""

from functools import partial
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.model import BertModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from megatron.data.dataset_utils import build_train_valid_test_datasets

def model_provider(pre_process=True, post_process=True):
    """Build the BERT model."""
    
    print_rank_0('building BERT model for BitPipe testing...')
    
    args = get_args()
    config = core_transformer_config_from_args(args)
    
    # BERT-specific configuration
    # Use binary head by default (set to False if --bert-no-binary-head is specified)
    has_binary_head = not getattr(args, 'bert_no_binary_head', False)
    num_tokentypes = 2 if has_binary_head else 0
    
    model = BertModel(
        config=config,
        num_tokentypes=num_tokentypes,
        add_binary_head=has_binary_head,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    
    return model

def get_batch(data_iterator):
    """Build the batch with synthetic data for testing."""
    
    args = get_args()
    
    # Generate synthetic BERT batch data
    batch_size = args.micro_batch_size
    seq_length = args.seq_length
    vocab_size = args.vocab_size
    
    # Create synthetic input tensors
    # text: input token IDs [batch_size, seq_length]
    text = torch.randint(0, vocab_size, (batch_size, seq_length), 
                        dtype=torch.long, device=torch.cuda.current_device())
    
    # types: token type IDs (0 for sentence A, 1 for sentence B) [batch_size, seq_length]
    types = torch.randint(0, 2, (batch_size, seq_length), 
                         dtype=torch.long, device=torch.cuda.current_device())
    
    # labels: masked language model labels [batch_size, seq_length]
    labels = torch.randint(0, vocab_size, (batch_size, seq_length), 
                          dtype=torch.long, device=torch.cuda.current_device())
    
    # is_random: next sentence prediction labels [batch_size]
    is_random = torch.randint(0, 2, (batch_size,), 
                             dtype=torch.long, device=torch.cuda.current_device())
    
    # loss_mask: mask for MLM loss computation [batch_size, seq_length]
    loss_mask = torch.ones((batch_size, seq_length), 
                          dtype=torch.float, device=torch.cuda.current_device())
    
    # padding_mask: attention mask [batch_size, seq_length]
    padding_mask = torch.ones((batch_size, seq_length), 
                             dtype=torch.long, device=torch.cuda.current_device())
    
    return {
        'text': text,
        'types': types,
        'labels': labels,
        'is_random': is_random,
        'loss_mask': loss_mask,
        'padding_mask': padding_mask
    }

def loss_func(loss_mask, sentence_order, output_tensor):
    """Loss function for BERT (matches pretrain_bert.py)"""
    lm_loss_, sop_logits = output_tensor

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}
    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step for BERT training (matches pretrain_bert.py)"""
    
    # Get the batch
    batch = get_batch(data_iterator)
    
    # Extract tensors
    tokens = batch['text']
    types = batch['types']
    sentence_order = batch['is_random']
    loss_mask = batch['loss_mask']
    lm_labels = batch['labels']
    padding_mask = batch['padding_mask']
    
    # Forward pass through the model
    output_tensor = model(tokens, padding_mask, tokentype_ids=types, lm_labels=lm_labels)
    
    return output_tensor, partial(loss_func, loss_mask, sentence_order)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build synthetic datasets for training, validation, and testing."""
    
    args = get_args()
    
    print_rank_0('> building synthetic BERT datasets for BitPipe testing...')
    
    # Create dummy dataset that just returns synthetic data
    class SyntheticBertDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Return synthetic BERT sample
            return {
                'text': torch.randint(0, args.vocab_size, (args.seq_length,)),
                'types': torch.randint(0, 2, (args.seq_length,)),
                'labels': torch.randint(0, args.vocab_size, (args.seq_length,)),
                'is_random': torch.randint(0, 2, (1,)),
                'loss_mask': torch.ones((args.seq_length,)),
                'padding_mask': torch.ones((args.seq_length,))
            }
    
    # Create datasets
    train_ds = SyntheticBertDataset(train_val_test_num_samples[0])
    valid_ds = SyntheticBertDataset(train_val_test_num_samples[1])
    test_ds = SyntheticBertDataset(train_val_test_num_samples[2])
    
    return train_ds, valid_ds, test_ds

def main():
    """Main training function."""
    
    # Start pretraining - args will be initialized by pretrain()
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={
            'tokenizer_type': 'NullTokenizer',
        }
    )

if __name__ == "__main__":
    main()