# Best Models

This directory is used to store the best performing models from our experiments. The models are saved with a specific naming convention to easily identify their configuration.

## Model Naming Convention

The filenames follow this structure:
`CharGPT_{context_length}-{stride_length}C{embedding_dimension}E{num_layers}L{num_heads}H`

### How to Read the Filename

Here is the breakdown of the parameters used in the filename:

- **`CharGPT`**: The prefix identifier for the Character GPT project.
- **`{context_length}`**: The size of the context window used during training (e.g., `128`).
- **`{stride_length}`**: The stride used for the sliding window (e.g., `1`).
- **`C`**: Separator indicating the end of Context/Stride parameters.
- **`{embedding_dimension}`**: The dimensionality of the embeddings (e.g., `384`).
- **`E`**: Indicator for Embedding dimension.
- **`{num_layers}`**: The number of Transformer encoder layers (e.g., `4`).
- **`L`**: Indicator for Layers.
- **`{num_heads}`**: The number of Multi-Head Attention heads (e.g., `4`).
- **`H`**: Indicator for Heads.

### Example

A file named **`CharGPT_128-1C384E4L4H`** corresponds to a model with:
- Context Length: 128
- Stride Length: 1
- Embedding Dimension: 384
- Layers: 4
- Heads: 4
