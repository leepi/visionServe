# Checkpoints

This directory holds trained model weights.

- `best.pt` — Starter checkpoint included with the repo. Real ResNet18 trained on a small synthetic dataset (~89% val accuracy on synthetic data). Useful for verifying the API works end-to-end before training on actual data.

To produce a CIFAR-10 checkpoint that replaces this one:

```bash
visionserve-train --config configs/cifar10_resnet18.yaml
```

The training script writes the best checkpoint to `best.pt` (overwriting the starter), along with `config.yaml` and `history.json` describing the run.

## Checkpoint format

Each `.pt` is a PyTorch dict containing:

- `model_state_dict` — the trained weights
- `epoch`, `val_acc`, `val_loss` — training-time metadata
- `class_names` — label list (in index order)
- `config` — the model + data config the checkpoint was trained with
