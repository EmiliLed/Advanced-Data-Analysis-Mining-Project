import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel,polynomial_kernel,sigmoid_kernel
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error
from numpy.linalg import eigh, inv

class KPLS:
    """
    Kernel Partial Least Squares
    """

    def __init__(self, n_components=2, kernel='rbf', kernel_params=None, tol=1e-12, regularization=1e-8):
        self.n_components = n_components
        self.kernel_name = kernel if isinstance(kernel, str) else None
        self.kernel = kernel if callable(kernel) else None
        self.kernel_params = kernel_params or {}
        self.tol = tol
        self.regularization = regularization

        # fitted attributes
        self.X_train_ = None
        self.Y_train_ = None
        self.K_ = None
        self.K_orig_ = None
        self.col_mean_ = None
        self.K_mean_all_ = None
        self.T_ = None
        self.C_ = None
        self.B_ = None
        self.A_ = 0
        self.H_=None
        self.y_mean_=None

    def _compute_kernel(self, X1, X2=None):
        if self.kernel is not None:
            return self.kernel(X1, X2, **self.kernel_params) if X2 is not None else self.kernel(X1, X1,
                                                                                                **self.kernel_params)
        if self.kernel_name == 'rbf':
            return rbf_kernel(X1, X2, **self.kernel_params) if X2 is not None else rbf_kernel(X1, X1,
                                                                                              **self.kernel_params)
        elif self.kernel_name == 'linear':
            return linear_kernel(X1, X2) if X2 is not None else linear_kernel(X1, X1)

        elif self.kernel_name == 'poly':
            return polynomial_kernel(X1, X2) if X2 is not None else polynomial_kernel(X1, X1,
                                                                                              **self.kernel_params)
        elif self.kernel_name == 'sigmoid':
            return sigmoid_kernel(X1, X2,**self.kernel_params) if X2 is not None else sigmoid_kernel(X1,X1,
                                                                                              **self.kernel_params)
        else:
            raise ValueError("Unknown kernel. Pass a callable or choose 'rbf' / 'linear'.")

    def _center_kernel(self, K):
        n = K.shape[0]
        if self.H_ is None:
            self.H_=np.eye(n)-1/n*np.ones((n,n))
        Kc=self.H_@K@self.H_
        col_mean = K.mean(axis=0)  # shape (n,)
        #row_mean = K.mean(axis=1)[:, None]  # shape (n,1)
        K_mean_all = K.mean()
        #Kc = K - row_mean - col_mean + K_mean_all


        return Kc, col_mean, K_mean_all
    def _center_Y(self, Y):
        return self.H_@Y
    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n, q = Y.shape
        K = self._compute_kernel(X, X)
        Kc,  col_mean, K_mean_all = self._center_kernel(K)
        Yc=self._center_Y(Y)
        self.X_train_ = X.copy()
        self.Y_train_ = Y.copy()
        self.y_mean_=Y.copy()-self.H_@Y.copy()
        self.K_ = Kc.copy()
        self.K_orig_ = K.copy()
        self.col_mean_ = col_mean.copy()
        self.K_mean_all_ = K_mean_all

        Ka = Kc.copy()
        Ya = Yc.copy()
        T_list = []
        C_list = []

        for a in range(self.n_components):
            YYt = Ya @ Ya.T  # n x n
            Ma = Ka @ YYt @ Ka
            #Ma = 0.5 * (Ma + Ma.T)
            vals, vecs = eigh(Ma)
            w = vecs[:, -1].reshape(-1, 1)
            t = Ka @ w
            t_norm = np.linalg.norm(t)
            if t_norm < self.tol:
                break
            t = t / t_norm
            c = (Ya.T @ t).reshape(q, 1)
            Ya = Ya - t @ c.T
            Pa = np.eye(n) - t@np.linalg.inv(t.T@t) @ t.T
            Ka = Pa @ Ka @ Pa
            T_list.append(t.ravel())
            C_list.append(c.ravel())
            self.A_ += 1

        if len(T_list) == 0:
            raise RuntimeError("No components extracted. Try lowering n_components or check data/kernel.")

        T = np.column_stack(T_list)  # n x A
        C = np.column_stack(C_list)  # q x A

        TK = T.T @ self.K_ @ T
        TK_reg = TK # + self.regularization * np.eye(TK.shape[0])
        inv_TKT = inv(TK_reg)
        B = T @ inv_TKT @ (T.T @ Yc)

        self.T_ = T
        self.C_ = C
        self.B_ = B  # n x q
        return self

    def predict(self, X_new):
        X_new = np.asarray(X_new)
        K_new = self._compute_kernel(X_new, self.X_train_)  # m x n
        row_mean_new = K_new.mean(axis=1, keepdims=True)  # m x 1
        K_new_c = K_new - self.col_mean_[None, :] - row_mean_new + self.K_mean_all_
        Y_pred = K_new_c @ self.B_
        print(self.y_mean_)
        return Y_pred+self.y_mean_[0]*np.ones((Y_pred.shape))

    def transform(self):
        return self.T_.copy()

    def score(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)
        return r2_score(Y_test, Y_pred)
