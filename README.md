# dist matrices

A short reference implementation of a function for calculating pairwise distance
functions using only NumPy arrays and broadcasting. The function is most similar
to `scipy.spatial.distance.cdist` with the signature:

```python
import numpy as np
A = np.ndarray

def cdist(x1: A, x2: A) -> A:
    """Calculate the pairwise distance matrix between two sets of points.

    NOTES:
    ------
    - The batch dimensions can be a tuple (), (b,), or (b1, b2, ...).

    - The p dimension must be the same for both inputs and must explicitly exist
      (i.e., it cannot be implied when p=1).

    ARGS:
    -----
    x1: (*batch, n1, p)

    x2: (*batch, n2, p)

    RETURNS:
    --------
    distance_matrix (*batch, n1, n2)
    """
    ...
```

Common use cases for the `cdist` function arise in various clustering
algorithms, Gaussian process regressions, and some graph / network problems.
This implementation is primarily useful in settings where `cdist` is not already
available from another framework or library (`scipy`, `torch`, etc.); namely, it
is not supported in the `jax` ecosystem already.

Two versions of the implementation are provided. One uses a naive implementation
of Euclidean distance. The other relies on `np.linalg.norm` to calculate norm
automatically.

The `dist.py` file contains two implementations of the `cdist` function
available through `scipy`, defined as `dist` and `dist2` with the same call
signature.

## Run code

To run the code, create an environment with `numpy` and `scipy` installed. The
`dist.py` script can then be run to try the implementations using

```python
python dist.py
```

Or simply copy this script verbatim into a Python environment and use it there.

## Batch dimensions

The implementation supports broadcasting over batch dimensions explicitly as
long as `batch` is a tuple of shapes which are consistent between the two
inputs.

## Changing metrics

With minimal changes, the implementations can be changed to use other distance
functions. Euclidean distances are most common and simplest, so I've provided
them.

## Numerical Stability

Numerical stability becomes especially important when using single precision
floats, hardware accelerators (GPUs), or JIT compiled functions.

The implementations shown here are not optimized for numerical stability. One
way to improve the stability is to pre-center the inputs before using them.
Another step is to clamp the min values to a small positive number to avoid
problems with small negative numbers if one computes square distances directly.

## References

Some helpful references for more background

1. [scipy cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

2. [torch cdist](https://pytorch.org/docs/stable/generated/torch.cdist.html)

3. [manual implementation](https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065)
