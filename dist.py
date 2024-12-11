import itertools
from typing import Callable, NamedTuple

import numpy as np

from numpy.typing import NDArray, DTypeLike
from scipy.spatial.distance import cdist

A = NDArray[np.float32 | np.float64]
Fn = Callable[[A, A], A]


class Dims(NamedTuple):
    batch: int
    n1: int
    n2: int


def batch_centering(x1: A, x2: A) -> tuple[A, A]:
    n1, n2 = np.prod(x1.shape[-2]), np.prod(x2.shape[-2])
    m1: A = x1.mean(axis=tuple(_ for _ in range(x1.ndim - 1)))
    m2: A = x2.mean(axis=tuple(_ for _ in range(x2.ndim - 1)))
    m = (n1 * m1 + n2 * m2) / (n1 + n2)

    return x1 - m, x2 - m


def diffs(x1: A, x2: A) -> A:
    return x1[:, None] - x2[None, :]


def batched_diffs(x1: A, x2: A) -> A:
    return x1[..., None, :] - x2[..., None, :, :]


# NOTE: The batch argument can be dropped without loss of generality. It is
# provided in this version only to demonstrate the results are consistent with
# the non-batched version.
def dist(x1: A, x2: A, batch: bool = False) -> A:
    x1, x2 = batch_centering(x1, x2)
    d = batched_diffs(x1, x2) if batch else diffs(x1, x2)
    d = np.square(d)
    sq_diffs: A = np.sum(d, axis=-1)
    dists = np.sqrt(sq_diffs)

    return dists


# NOTE: As above
def dist2(x1: A, x2: A, batch: bool = False) -> A:
    x1, x2 = batch_centering(x1, x2)
    d = batched_diffs(x1, x2) if batch else diffs(x1, x2)
    dists: A = np.linalg.norm(d, axis=-1)

    return dists


def cmp_dists(
    x1: A,
    x2: A,
    fns: list[Fn],
    dims: Dims,
    dtype: DTypeLike,
    atol: float,
):
    def cmp(fn1: Fn, fn2: Fn, dtype: DTypeLike, atol: float):
        x1_: A = x1.astype(dtype)
        x2_: A = x2.astype(dtype)
        b = np.allclose(fn1(x1_, x2_), fn2(x1_, x2_), atol=atol)
        fail_msg = (
            f"failed for {fn1.__name__} and {fn2.__name__} at atol {atol} "
            f"and dtype {dtype} and dims {dims}"
        )
        if not b:
            print(fail_msg)

    for fn1, fn2 in itertools.combinations(fns, 2):
        cmp(fn1, fn2, dtype, atol)


def main():
    rng = np.random.default_rng(0)
    atol = 1e-3

    ns = [
        Dims(batch=1, n1=10, n2=10),
        Dims(batch=1, n1=50, n2=10),
        Dims(batch=1, n1=10, n2=50),
        Dims(batch=10, n1=10, n2=10),
        Dims(batch=10, n1=50, n2=10),
        Dims(batch=10, n1=10, n2=50),
        Dims(batch=100, n1=10, n2=10),
        Dims(batch=100, n1=50, n2=10),
        Dims(batch=100, n1=10, n2=50),
    ]
    d = [3, 5, 10, 50, 100]
    dtypes = [np.float32, np.float64]

    # First demo behavior with no batched diffs
    # Note this part compares to cdist
    fns: list[Fn] = [
        cdist,
        lambda x1, x2: dist(x1, x2, batch=False),
        lambda x1, x2: dist2(x1, x2, batch=False),
    ]
    for n, di, dtype in itertools.product(ns, d, dtypes):
        x1 = rng.random((n.n1, di)).astype(dtype)
        x2 = rng.random((n.n2, di)).astype(dtype)
        cmp_dists(x1, x2, fns, n, dtype, atol)

    # Now demo behavior with batched diffs
    # Here we skip cdist since it doesn't support batched inputs
    batched_fns: list[Fn] = [
        lambda x1, x2: (
            np.stack([cdist(x1[i], x2[i]) for i in range(x1.shape[0])])
            if x1.ndim == 3
            else cdist(x1, x2)
        ),
        lambda x1, x2: dist(x1, x2, batch=True),
        lambda x1, x2: dist2(x1, x2, batch=True),
    ]
    for n, di, dtype in itertools.product(ns, d, dtypes):
        x1 = rng.random((n.batch, n.n1, di)).astype(dtype)
        x2 = rng.random((n.batch, n.n2, di)).astype(dtype)
        cmp_dists(x1, x2, batched_fns, n, dtype, atol)

        # Also show this works with non-batched inputs by indexing into the batch dim
        cmp_dists(x1[0], x2[0], batched_fns, n, dtype, atol)


if __name__ == "__main__":
    main()
