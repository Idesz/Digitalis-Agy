# Neuromorphic Brain Emulation Platform

Comprehensive documentation and instructions for running the neuromorphic agent and live dashboard.

Refer to:
- [Dashboard and Training Guide](docs/DASHBOARD_TRAINING_GUIDE.md)
- [Detailed Architecture Specification](docs/ARCHITECTURE.md)
- [API Specifications](docs/API_SPECIFICATION.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

### Training and Quick Run

Run the quick training with the default 160 training and 160 test samples:

```bash
python train.py
python dashboard_app.py
```

The trainer reports the source and actual sample counts. If processed N-MNIST is not available, it clearly reports the synthetic fallback. To require real N-MNIST, prepare the official train/test splits and run the quick measurement:

```bash
python prepare_n_mnist.py
python train.py --require-n-mnist
```

The preparation command downloads the full official archives (about 1.10 GiB total) and writes versioned tensors under `data/n_mnist/processed_nmnist.pt`. Raw and processed datasets are Git-ignored; copy `data/n_mnist/` to another machine or rerun the preparation command there.

Use the complete processed training and test splits with:

```bash
python train.py --require-n-mnist --all-train --all-test --epochs 3
```

This keeps the official test split out of training and reports its loss and accuracy after training.

### Held-out Benchmark
```bash
python benchmark.py --task xor --samples 256 --epochs 10 --seed 42
python benchmark.py --measure-forgetting --samples 256 --epochs 5 --seed 42
python benchmark.py --task classification --require-n-mnist --samples 1000 --epochs 10 --input-dim 64
```

The benchmark command reports train/validation metrics per epoch and a held-out test result; the forgetting mode trains XOR then parity and measures each task on its reserved validation/test split.
