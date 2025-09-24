# CIFAR-10 Image Classification (TensorFlow/Keras)

Keras/TensorFlow CNN trained on CIFAR-10 with BatchNorm + Dropout. Saves model and plots to `artifacts/`.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 15 --batch-size 128
```
