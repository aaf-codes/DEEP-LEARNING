#!/usr/bin/env python
# coding: utf-8

# # Bidirectional LSTM Manual Implementation
# 
# In this notebook, we implement a Bidirectional LSTM layer using raw PyTorch tensor operations. We manually compute the four LSTM gate equations (Input, Forget, Cell Candidate, and Output) and run the recurrence both forward and backward. Finally, we concatenate the hidden states and validate our implementation against PyTorch's `nn.LSTM(bidirectional=True)`.

# In[ ]:


import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)


# ## 1. Manual BiLSTM Implementation

# In[ ]:


def manual_bilstm(input_seq, w_ih, w_hh, b_ih, b_hh, w_ih_rev, w_hh_rev, b_ih_rev, b_hh_rev):
    """
    Manual implementation of a bidirectional LSTM.

    Args:
        input_seq: Tensor of shape (seq_len, batch_size, input_size)
        w_ih, w_hh, b_ih, b_hh: Weights and biases for the forward pass
        w_ih_rev, w_hh_rev, b_ih_rev, b_hh_rev: Weights and biases for the backward pass

    Returns:
        outputs: Tensor of shape (seq_len, batch_size, 2 * hidden_size)
    """
    seq_len, batch_size, input_size = input_seq.shape

    # PyTorch's weight_hh_l0 has shape (4*hidden_size, hidden_size)
    # So the second dimension is exactly the hidden size.
    hidden_size = w_hh.shape[1]

    # Initialize hidden and cell states for forward and backward passes
    h_f = torch.zeros(batch_size, hidden_size, device=input_seq.device)
    c_f = torch.zeros(batch_size, hidden_size, device=input_seq.device)

    h_b = torch.zeros(batch_size, hidden_size, device=input_seq.device)
    c_b = torch.zeros(batch_size, hidden_size, device=input_seq.device)

    outputs_f = []
    outputs_b = []

    # ------------------ Forward Pass ------------------
    for t in range(seq_len):
        x_t = input_seq[t]

        # Compute all gates at once
        gates_f = x_t @ w_ih.T + h_f @ w_hh.T + b_ih + b_hh

        # Split gates: input (i), forget (f), cell candidate (g), output (o)
        i_f, f_f, g_f, o_f = gates_f.chunk(4, dim=1)

        # Apply activations
        i_f = torch.sigmoid(i_f)
        f_f = torch.sigmoid(f_f)
        g_f = torch.tanh(g_f)
        o_f = torch.sigmoid(o_f)

        # Update cell state and hidden state
        c_f = f_f * c_f + i_f * g_f
        h_f = o_f * torch.tanh(c_f)

        outputs_f.append(h_f)

    # ------------------ Backward Pass ------------------
    for t in reversed(range(seq_len)):
        x_t = input_seq[t]

        # Compute all gates at once for the reversed sequence
        gates_b = x_t @ w_ih_rev.T + h_b @ w_hh_rev.T + b_ih_rev + b_hh_rev

        # Split gates
        i_b, f_b, g_b, o_b = gates_b.chunk(4, dim=1)

        # Apply activations
        i_b = torch.sigmoid(i_b)
        f_b = torch.sigmoid(f_b)
        g_b = torch.tanh(g_b)
        o_b = torch.sigmoid(o_b)

        # Update cell state and hidden state
        c_b = f_b * c_b + i_b * g_b
        h_b = o_b * torch.tanh(c_b)

        outputs_b.append(h_b)

    # Stack outputs
    outputs_f = torch.stack(outputs_f)  # Shape: (seq_len, batch_size, hidden_size)

    # The backward outputs were collected in reverse order, so we need to reverse the list
    # before stacking to align them chronologically with the forward outputs.
    outputs_b = torch.stack(outputs_b[::-1])  # Shape: (seq_len, batch_size, hidden_size)

    # Concatenate forward and backward hidden states along the feature dimension
    outputs = torch.cat([outputs_f, outputs_b], dim=2)

    return outputs


# ## 2. Validation against PyTorch's `nn.LSTM`

# In[ ]:


# Hyperparameters
seq_len = 10
batch_size = 32
input_size = 64
hidden_size = 128

# Create dummy input data
x = torch.randn(seq_len, batch_size, input_size)

# Initialize PyTorch BiLSTM
pytorch_bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)

# Run PyTorch BiLSTM
out_pytorch, _ = pytorch_bilstm(x)

# Extract weights and biases to use in our manual implementation
w_ih = pytorch_bilstm.weight_ih_l0
w_hh = pytorch_bilstm.weight_hh_l0
b_ih = pytorch_bilstm.bias_ih_l0
b_hh = pytorch_bilstm.bias_hh_l0

w_ih_rev = pytorch_bilstm.weight_ih_l0_reverse
w_hh_rev = pytorch_bilstm.weight_hh_l0_reverse
b_ih_rev = pytorch_bilstm.bias_ih_l0_reverse
b_hh_rev = pytorch_bilstm.bias_hh_l0_reverse

# Run manual BiLSTM
out_manual = manual_bilstm(
    x, 
    w_ih, w_hh, b_ih, b_hh, 
    w_ih_rev, w_hh_rev, b_ih_rev, b_hh_rev
)

# Validate Correctness
print(f"PyTorch Output Shape: {out_pytorch.shape}")
print(f"Manual Output Shape:  {out_manual.shape}")

# Use torch.allclose to verify that the tensors are identical (within a small numerical tolerance)
is_close = torch.allclose(out_manual, out_pytorch, atol=1e-5)
print(f"\nValues match PyTorch implementation: {is_close}")

