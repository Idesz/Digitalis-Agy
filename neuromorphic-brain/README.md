# Neuromorphic Brain Emulation Platform

Comprehensive documentation and instructions for running the neuromorphic agent and live dashboard.

Refer to:
- [Dashboard and Training Guide](docs/DASHBOARD_TRAINING_GUIDE.md)
- [Detailed Architecture Specification](docs/ARCHITECTURE.md)
- [API Specifications](docs/API_SPECIFICATION.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

### Quick Run
```bash
python train.py
python dashboard_app.py
```

### Held-out Benchmark
```bash
python benchmark.py --task xor --samples 256 --epochs 10 --seed 42
python benchmark.py --measure-forgetting --samples 256 --epochs 5 --seed 42
```

The benchmark command reports train/validation metrics per epoch and a held-out test result; the forgetting mode trains XOR then parity and measures each task on its reserved validation/test split.
