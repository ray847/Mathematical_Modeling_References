from math import exp

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

class GreyModel_1_1:
    """Implementation of the `GM(1, n)` prediction model.

    Usage:
        fit(x, t): Fit the model.
        predict(t): Predict x.
        predict_integral(t): Predict the value of the integral of x.
    Properties:
        a, b, C (numpy.ndarray): Coefficients/Constants of the ODE.
    """
    def __init__(self):
        self._t_offset: float = 0.0
        self._t_scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
        self._x_scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
        self._a: np.ndarray = None
        self._b: np.ndarray = None
        self._C: np.ndarray = None

    def fit(self, x: np.ndarray, t: np.ndarray = None) -> None:
        """Fit the grep model with the input values.

        Args:
            x (numpy.ndarray): The values at different time stamps. Works for
            both 1-dimensional and multi-dimensional data. But it does not
            implement GM(1, N) for multi-dimensional data. Instead the fitting
            is done to different dimensions individually.
            t (numpy.ndarray, optional): The times stamps corresponding to x.
            The length of this argument should be the same as the first
            dimension of `x`. This argument should also be 1-dimensional.
            Assumes t starts at 1 and increments by 1 if this argument is None.
            Defaults to None.
        """
        if t is not None:
            assert x.shape[0] == t.size
        else:
            t = np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=int)
        # Data Preprocessing.
        x = self._preprocess_x(x)
        t = self._preprocess_t(t)
        X = self._calculate_X(x, t)
        # Solve for a & b in: x + aX = b <=> x = a(-X) + b
        a = np.zeros((x.shape[1]))
        b = np.zeros((x.shape[1]))
        linear_regression = LinearRegression()
        for dim in range(x.shape[1]):
            linear_regression.fit(-X[:, dim].reshape(-1, 1),
                                  x[:, dim].reshape(-1, 1))
            a[dim] = linear_regression.coef_[0][0]
            b[dim] = linear_regression.intercept_[0]
        self._a: np.ndarray = a
        self._b: np.ndarray = b
        # Calculate C in the ODE solution:
        # X = b/a + C*e^(-a*t) <=> C = (X - b/a) * e^(a*t)
        C = np.zeros((a.size))
        for dim in range(a.size):
            C[dim] = np.dot(
                np.exp(-a[dim] * t),
                (X[:, dim] - b[dim] / a[dim]).flatten()
            ) / np.linalg.norm(np.exp(-a[dim] * t))**2
        self._C: np.ndarray = C

    def predict(self, t: np.ndarray) -> np.ndarray:
        """Predict the possible values at `ts`.

        According to the ODE:
        $$\mathbf{x} + a\mathbf{X} = b$$

        $$\Rightarrow \mathbf{X} = \frac{b}{a} + C e^{-at}$$

        $$\mathbf{x} = \frac{d\mathbf{X}}{dt} = -aCe^{-at}$$
        Args:
            t (numpy.ndarray): The time stampes at whihc to predict.

        Returns:
            numpy.ndarray: The prediction results.
        """
        t = self._scale_t(t)
        a_times_t = (t.reshape(-1, 1) @ self._a.reshape(1, -1))
        return self._inverse_scale_x(-self._a * self._C * np.exp(-a_times_t))
    
    def predict_integral(self, t: np.ndarray) -> np.ndarray:
        """Predict the possible integral values at `ts`.

        According to the ODE;
        $$\mathbf{x} + a\mathbf{X} = b$$

        $$\Rightarrow \mathbf{X} = \frac{b}{a} + C e^{-at}$$
        Args:
            t (numpy.ndarray): The time stampes at whihc to predict.

        Returns:
            numpy.ndarray: The prediction results.
        """
        t = self._scale_t(t)
        a_times_t = (t.reshape(-1, 1) @ self._a.reshape(1, -1))
        return self._inverse_scale_x(
            self._b/self._a + self._C * np.exp(-a_times_t)
        )

    def _preprocess_t(self, t: np.ndarray) -> np.ndarray:
        return self._t_scaler.fit_transform(t.reshape(-1, 1)).flatten()
        self._t_offset = -t.min()
        return t + self._t_offset
    
    def _scale_t(self, t: np.ndarray) -> np.ndarray:
        return self._t_scaler.transform(t.reshape(-1, 1)).flatten()
        return t + self._t_offset
    
    def _preprocess_x(self, x: np.ndarray) -> np.ndarray:
        """Scale the x into the range of [0, 1].

        Notes: This method transforms the scaler according to the input data.
        For scaling without transforming, use `_scale()`.

        Args:
            x (numpy.ndarray): The data points.

        Returns:
            numpy.ndarray: The scaled data points. Shape is always
            (n data points) x (n dimensions)
        """
        assert len(x.shape) == 1 or 2
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        return self._x_scaler.fit_transform(x)
    
    def _scale_x(self, x: np.ndarray) -> np.ndarray:
        """Scale the data points according to the scaling during preprocessing.
        
        Notes: This method does not transform the scaler according to the input
        data. For transforming and scaling, use `_preprocess()`.

        Args:
            x (numpy.ndarray): The data points before scaling.

        Returns:
            numpy.ndarray: The scaled data points. Shape is always
            (n data points) x (n dimensions)
        """
        return self._x_scaler.transform(x)

    def _inverse_scale_x(self, x: np.ndarray) -> np.ndarray:
        """Reverse the scaling done to data points.
        
        Args:
            x (numpy.ndarray): The data points that has been scaled scaling.

        Returns:
            numpy.ndarray: The unscaled data points. Shape is always
            (n data points) x (n dimensions)
        """
        return self._x_scaler.inverse_transform(x)

    def _calculate_X(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Calculate X from x.

        X(t) is supposed to be the integral of x(t):

        $$\mathbf{X}(t) = \int_{0}^{t}\mathbf{x}(t) dt$$

        For approximation, x is linearly interpolated to calculate X:

        $$\mathbf{X}(t_n) = \sum_{k = 0}^{n - 1}\int_{t_k}^{t_{k + 1}}\mathbf{x}(t) dt$$

        $$\mathbf{X}(t_n) \approx \sum_{k = 0}^{n - 1}
        \frac{1}{2}(\mathbf{x}_{t_k} + \mathbf{x}_{t_{k + 1}})\times(t_{k + 1} - t_k) $$

        Args:
            x (numpy.ndarray): The x values.
            t (numpy.ndarray): The times stamps for the x values.
        Returns:
            numpy.ndarray: The X array.
        """
        n = t.size
        X_segments = []
        for k in range(n - 1):
            X_segments.append(0.5 * (x[k] + x[k + 1]) * (t[k + 1] - t[k]))
        X = np.cumsum(np.array(X_segments), axis=0)
        return np.concatenate([
            np.zeros((1, x.shape[1])),
            X
        ], axis=0)

    @property
    def a(self) -> np.ndarray:
        """Get the coefficient a in the ODE.
        """
        return self._a

    @property
    def b(self) -> np.ndarray:
        """Get the coefficient b in the ODE.
        """
        return self._b

    @property
    def C(self) -> np.ndarray:
        """Get the constant C in the ODE.
        """
        return self._C
    
    @property
    def coeffs(self) -> tuple[np.ndarray]:
        """Get all the coeffs of the model: (a, b, C)

        Returns:
            tuple[numpy.ndarray]: The coefficients in the order of a, b, C.
        """
        return self.a, self.b, self.C

"""
Testing
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    model = GreyModel_1_1()
    x = np.array([
        [0.5, 0],
        [0, 0.5],
        [0.5, 0],
        [1, 0.5],
        [0.5, 1],
        [1, 0.5],
    ])
    #x = np.array([
    #    [0.5],
    #    [0],
    #    [0.5],
    #    [1],
    #    [0.5],
    #    [1],
    #])
    t = np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=int)
    model.fit(x, t)
    print("right:", model.coeffs)
    X = model._inverse_scale_x(
        model._calculate_X(model._scale_x(x), model._scale_t(t))
    )
    t_pred = np.linspace(0, x.shape[0] + 1, 100)
    x_pred = model.predict(t_pred)

    # Plotting
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(t, x, ".", label="data", color="black", linestyle="dotted")
    ax.plot(t, X, ".", label="X", color="green")
    ax.plot(t_pred, model.predict_integral(t_pred), label="X_pred", color="green")
    ax.plot(t_pred, x_pred, label="pred", color="blue")

    ax.legend()
    plt.show()
