import torch

def rbf_kernel(x, y, sigma=1.0):
    return torch.exp(-0.5 * (x - y) ** 2 / sigma ** 2)

def poly_kernel(x, y, degree=3, c=1.0):
    return (x * y + c) ** degree

def matern_kernel(x, y, ell=1.0):
    r = torch.abs(x - y)
    sqrt3 = torch.sqrt(torch.tensor(3.0))
    return (1 + sqrt3 * r / ell) * torch.exp(-sqrt3 * r / ell)


def G_matrix(
    V, W, S, T,
    kernel_F=None,
    kernel_K1=None,
    kernel_K2=None,
    flip_K2=True,
):
    """
    Generalized Gram matrix G((V,S),(W,T)) supporting arbitrary kernels.

    Parameters
    ----------
    V, W : torch.Tensor
        Shape (N, τ+1) and (M, τ+1). The 'v' and 'w' sequences.
    S, T : torch.Tensor
        Shape (N, τ+1) and (M, τ+1). The 's' and 't' sequences.
    kernel_F : callable
        Function kernel_F(x, y) -> scalar kernel matrix of shape (N, M).
        If None, defaults to RBF kernel with sigma=1.0.
    kernel_K1 : callable
        Function kernel_K1(s, t) -> (N, M, τ+1, τ+1).
    kernel_K2 : callable
        Function kernel_K2(v, w) -> (N, M, τ+1, τ+1).
    flip_K2 : bool
        If True, reverses v and w before computing K2 (default=True).

    Returns
    -------
    torch.Tensor
        Full Gram matrix G(V,S; W,T) of shape (N, M).
    """

    N, _ = V.shape
    M, _ = W.shape

    # Default RBF kernel
    def default_rbf(x, y, sigma=1.0):
        return torch.exp(-0.5 * (x - y) ** 2 / sigma ** 2)

    # Use defaults if not provided
    if kernel_F is None:
        kernel_F = lambda x, y: default_rbf(x, y, sigma=1.0)
    if kernel_K1 is None:
        kernel_K1 = lambda s, t: default_rbf(s.unsqueeze(-2), t.unsqueeze(-3), sigma=1.0)
    if kernel_K2 is None:
        kernel_K2 = lambda v, w: default_rbf(v.unsqueeze(-2), w.unsqueeze(-3), sigma=1.0)

    # Pairwise expansions
    v_exp, w_exp = V[:, None, :], W[None, :, :]
    s_exp, t_exp = S[:, None, :], T[None, :, :]

    # 1️⃣ First term: F-kernel on v0, w0
    G_F = kernel_F(v_exp[..., 0], w_exp[..., 0])  # shape (N, M)

    # Prepare for K2 term (flipped if requested)
    v_in = torch.flip(v_exp, dims=[-1]) if flip_K2 else v_exp
    w_in = torch.flip(w_exp, dims=[-1]) if flip_K2 else w_exp

    # Compute K1 and K2 (each shape: N × M × τ+1 × τ+1)
    K1 = kernel_K1(s_exp, t_exp)
    K2 = kernel_K2(v_in, w_in)

    # Combine and sum over temporal axes
    sum_term = torch.sum(K1 * K2, dim=(-2, -1))

    return G_F + sum_term


def kernel_ridge_regression(
    X_train, S_train, Y_train,
    X_test, S_test,
    kernel_fn,
    reg_lambda=1e-6,
):
    """
    Kernel Ridge Regression using a user-specified kernel function.

    Parameters
    ----------
    X_train, S_train : torch.Tensor
        Training data tensors of shape (N, τ+1).
    Y_train : torch.Tensor
        Training targets, shape (N,) or (N, d).
    X_test, S_test : torch.Tensor
        Test data tensors of shape (M, τ+1).
    kernel_fn : callable
        Function kernel_fn(V, W, S, T) -> (N, M) kernel matrix.
        This can be G_matrix (possibly with different F, K1, K2),
        or any other kernel function with the same signature.
    reg_lambda : float
        Regularization strength (λ) for ridge regression.

    Returns
    -------
    torch.Tensor
        Predictions of shape (M,) or (M, d).
    """
    N = X_train.shape[0]

    # 1️⃣ Compute kernel (Gram) matrices
    K_train = kernel_fn(X_train, X_train, S_train, S_train)  # (N, N)
    K_cross = kernel_fn(X_test, X_train, S_test, S_train)    # (M, N)

    # 2️⃣ Regularization
    I = torch.eye(N, device=X_train.device, dtype=X_train.dtype)
    A = K_train + reg_lambda * I

    # 3️⃣ Solve for alpha = A^{-1} Y_train
    alpha = torch.linalg.solve(A, Y_train)

    # 4️⃣ Predict
    Y_pred = K_cross @ alpha

    return Y_pred