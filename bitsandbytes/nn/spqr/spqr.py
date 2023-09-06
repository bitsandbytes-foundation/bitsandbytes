"""
SpQR (Sparse Quantized Representation) implementation

Implements a specialized 3.5-bit weight quantization for neural networks, based on the SpQR (Sparse Quantized Representation) method. The goal is to achieve significant model compression with minimal performance degradation. Special focus is on creating a custom packed data format to hold 3-bit weights and additional state information (scales, zero-points, and outliers).
"""

import torch
import torch.nn as nn


class Params3bit(nn.Parameter):
    """
    Custom Parameter Class for 3-bit Quantization
    
    Custom Data Type (Params3bit): Implement a custom data type that can hold 3-bit quantized weights, along with scales and zero-points for dequantization.
    """
    # not sure if this is even still needed
    pass


class Linear3BitSpQR(nn.Linear):
    """
    Implements the custom linear layer for 3-bit quantization.
    
    For more information, read the paper: SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression (https://arxiv.org/abs/2306.03078).
    
    State Management: Extend PyTorch's existing layers or create new ones that can manage additional states, like the scales, zero-points, and sensitive weights.

    By customizing the methods `_save_to_state_dict()` and `_load_from_state_dict()`, we ensure that all the extra information we're tracking gets saved and loaded correctly, making our custom layer fully compatible with PyTorch's existing mechanisms for saving and loading models.
    """

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.weight = Params3bit(self.weight.data)

        # Initialize additional state variables
        self.first_level_stats = torch.zeros(self.weight.shape)
        self.second_level_stats = torch.zeros(self.weight.shape[0])
        # self.csr_outliers = CSRMatrixOutliers(...)  # TODO: Initialize CSRMatrixOutliers

    @property
    def packed_weights(self):
        """
        Quantized Weights: The 3-bit quantized weights can be packed 21 at a time into a 64-bit integer (since 3x21 = 63 < 64).
        """

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Override PyTorch method to save a `Linear3BitSpQR` layer to disk, including the custom `Params3bit` data type and the additional state:
        
        1. Quantized weights (3-bit)
        2. Scales            (3-bit)
        3. Zero-points       (3-bit)
        4. Outliers          (64-bit = 32bit + 32bit (index)), likely torch.sparse_coo_tensor
        5. Shape information
        6. Type information

        The .detach() method creates a new tensor that shares the same storage as the original tensor but with the computation graph detached. In other words, it creates a tensor that won't require gradients. This is useful when saving tensors to disk because you typically don't want to save the computation graph that was used to create the tensor—just the raw numeric values.
        """
        # super()._save_to_state_dict(destination, prefix, keep_vars)
        # destination[prefix + 'first_level_stats'] = self.first_level_stats
        # destination[prefix + 'second_level_stats'] = self.second_level_stats
        # # TODO: Add csr_outliers to destination

        super()._save_to_state_dict(destination, prefix, keep_vars)

        # Save 64-bit packed 3-bit quantized weights
        destination[prefix + 'packed_weights'] = self.packed_weights.detach()

        # Save scales and zero-points
        destination[prefix + 'scales'] = self.scales.detach()
        destination[prefix + 'zero_points'] = self.zero_points.detach()

        # Save any additional statistics or flags
        destination[prefix + 'additional_stats'] = self.additional_stats.detach()

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs):
        """
        Override PyTorch method to load this custom saved state information back into a new `Linear3BitSpQR` layer, and all the internal state is restored correctly.
        """
        self.first_level_stats = state_dict[prefix + 'first_level_stats']
        self.second_level_stats = state_dict[prefix + 'second_level_stats']
        # TODO: Load csr_outliers from state_dict
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)

    def update_weights(self, optimizer):
        """Update weights using a given PyTorch optimizer."""
        optimizer.step()


"""

### Why Are the Outliers in 32-bit?

In the case of SpQR, outliers are weights that are deemed too "sensitive" or important to be quantized down to 3-bits. These weights are stored in their original, higher-precision form to preserve their impact on the model's performance. According to the paper, each outlier is stored as two 16-bit values: one for the weight value and another for the column index, making it a total of 32 bits for each outlier.

#### For the Implementer:
In the context of neural networks, using a 32-bit representation for outliers makes sense for several reasons:

1. **Preservation of Information**: The 16-bit weight value allows for a high level of granularity, thus maintaining the weight's original value without much loss of information.

2. **Indexing**: The 16-bit column index is crucial for unstructured sparsity. It allows the model to know where to place these outlier weights during the forward and backward passes, ensuring that they contribute to the computation correctly.

3. **Efficient Access**: Storing the index alongside the weight makes accessing these values more cache-efficient, which can be crucial for computational performance.

4. **Compatibility**: 32-bit (or rather, 16-bit for each part of the outlier) is a commonly used size for floating-point numbers and integers in computing, making it a convenient choice from a hardware perspective.

### How Do We Know If It's Working:

When you implement the 32-bit storage for outliers, you should be able to:

1. Accurately identify which weights are outliers based on your chosen criteria.
2. Store these weights in 32-bit form (16 bits for the weight value and 16 bits for the index).
3. Successfully save and reload this information using PyTorch's serialization methods (`_save_to_state_dict` and `_load_from_state_dict`).
4. Verify that the "restored" outliers are identical to the original ones and that they are placed correctly in the weight matrix.

To sum it up, the outliers are in 32-bit to preserve their original value and to maintain their positional information, which is crucial for the accurate and efficient functioning of the model.
"""
