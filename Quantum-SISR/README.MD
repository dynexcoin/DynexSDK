# Quantum Single Image Super-Resolution on Dynex

Implementation of a Quantum Single Image Super-Resolution algorithm to use on the Dynex platform. One of the well-known classical approaches for SISR relies on the well-established patch-wise sparse modeling of the problem. Yet, this field’s current state of affairs is that deep neural networks (DNNs) have demonstrated far superior results than traditional approaches. Nevertheless, quantum computing is expected to become increasingly prominent for machine learning problems soon. Among the two paradigms of quantum computing, namely universal gate quantum computing and adiabatic quantum computing (AQC), the latter has been successfully applied to practical computer vision problems, in which quantum parallelism has been exploited to solve combinatorial optimization efficiently. 

This algorithm demonstrates formulating quantum SISR as a sparse coding optimization problem, which is solved using the Dynex Neuromorphic Computing Platform via the Dynex SDK. This AQC-based algorithm is demonstrated to achieve improved SISR accuracy.

## Requirements

In addition to having the Dynex SDK installed and configured, the following packages are required:

```
pip install spams-bin
pip install easydict
pip install qubovert
pip install -U scikit-image
```

## Run

```
python run.py
``` 

## Training

```
python dict_train.py
```

## References

"[Quantum Annealing for Single Image Super-Resolution](https://arxiv.org/abs/2304.08924)" by Han Yao Choong, Suryansh Kumar and Luc Van Gool (ETH Zürich)
