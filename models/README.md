# Model checkpoints

Training writes the best validation checkpoint to:

```text
models/efficientnet_b0_koniq10k.pt
```

The file contains the EfficientNet-B0 state, preprocessing configuration, normalized target range, selected epoch and phase, validation metrics, optimizer state, and seed. The inference pipeline reconstructs the model without downloading pretrained weights.

Weights are excluded from Git because they are large. Reproduce the local checkpoint with the training command in the repository README, or point the API to another compatible local checkpoint with `IQA_MODEL_PATH`.
