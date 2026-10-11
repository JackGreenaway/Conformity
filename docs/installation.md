# Installation

Install Conformity when you already have a scikit-learn model and need held-out uncertainty calibration:

```bash
uv add conformity-calib
# Or:
pip install conformity-calib
```

The distribution is named `conformity-calib`; import it as `conformity`. Python **3.9 or newer** is supported. Runtime requirements are NumPy >=2.0.2, scikit-learn >=1.6.1 and typing-extensions >=4.15.0. The lock resolves compatible dependencies for each Python version; newer dependency releases may require newer Python.

Plotting is optional and uses Matplotlib:

```bash
uv add matplotlib
# Or:
pip install matplotlib
```

```python
from conformity import ConformalClassifier, ConformalRegressor, __version__

assert isinstance(__version__, str)
assert callable(ConformalRegressor) and callable(ConformalClassifier)
print(__version__)
```

For a checkout, use the locked [development instructions](development.md). Continue with [getting started](getting-started.md) to fit your first calibrated model.
