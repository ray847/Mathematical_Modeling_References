"""
Imports
"""
from typing import Any

import numpy as np  # np.ndarray
import sympy  # sympy.Matrix

class MarkovModel:
    """the Markov model for single dimension data
    
    This model uses the markove probability chain to predict future values based
    on the most recent state.

    Usage:
        fit(state_sequence: numpy.ndarray): To fit the model.
        predict(state_sequence: numpy.ndarray): To predict with the input state.
        predict_proba(state_sequence: numpy.ndarray): The probability version of
        `precdict()`
        trend(): The stable state in the infinite future.
        trend_proba(): The probability version of `trend()`.
    Properties:
        probability_matrix
        frequency_matrix
        unique_states
        unique_indicies
        n_unique_values
        state_to_index_map
        index_to_state_map
    """
    def __init__(self):
        self._prob_mat: sympy.Matrix = sympy.Matrix()
        self._freq_mat: sympy.Matrix = sympy.Matrix()
        self._states: np.ndarray = np.array([])
        self._state_to_index: dict[Any, int] = {}
        self._index2state: dict[int, Any] = {}

    def fit(self, state_sequence: np.ndarray) -> None:
        """Fit the model with the input state sequence.

        This method extracts the state to index mapping(`state_to_index_map` &
        `index_to_state_map`) then calculates the transition frequency &
        probability matrices.

        Args:
            state_sequence (np.ndarray): A `numpy.ndarray` object containing the
            sequence of states from oldest to newest.
        """
        data_transformed: np.ndarray = self._embed_transform(state_sequence)
        self._calculate_prob_mat(data_transformed)
    
    def predict(self,
                state_sequence: np.ndarray,
                length: int = 1) -> np.ndarray:
        """Predict the future values based on the most recent input state.

        Note: This method does not re-fit the model with tht input data.
        Args:
            state_sequence (numpy.ndarray): A `numpy.ndarray` object containing
            the sequence of states from oldest to newest.
            length (int, default=1): The number of future states to predict.
        Returns:
            numpy.ndarray: A `numpy.ndarray` containing the predicted future
            states.
        """
        proba: np.ndarray = self.predict_proba(state_sequence, length)
        result = []
        for i in range(length):
            max_proba_index: int = list(proba[i]).index(proba[i].max())
            result.append(self._index2state[max_proba_index])
        return np.array(result)

    def predict_proba(self,
                      state_sequence: np.ndarray,
                      length: int = 1) -> np.ndarray:
        """Predict the probabilities of all the future values.

        Args:
            state_sequence (numpy.ndarray): The data to base on for prediction.
            length (int, default=1): The number of future values to predict.
        Returns:
            numpy.ndarray: The probabilities in the form of
            np.ndarray[sympy.matrix]. The index of the probabilities should
            match `self.index_to_state_map`.
        """
        assert length > 0
        # Search for the input element among the possible states.
        old_vec = sympy.zeros(self.n_unique_states, 1)
        old_index: int = self._embed(state_sequence)[0]
        old_vec[old_index] = 1.0
        # Calculate the results based on the current probability matrix.
        results = np.array([sympy.zeros(self.n_unique_states, 1)] * length)
        results[0] = self._prob_mat @ old_vec
        for i in range(1, length):
            results[i] = self._prob_mat @ results[i - 1]
        return results
    
    def trend(self) -> list[Any]:
        """The state at which the model can stably maintain(in the infinite
        future).

        Returns:
            list[Any]: The predicted stable state(s).
        """
        solves = self.trend_proba()
        res = []
        for p_trend in solves:
            max_index = list(p_trend).index(max(p_trend))
            res.append(self._index2state[max_index])
        return res

    def trend_proba(self) -> sympy.Matrix:
        """The state at which the model can stably maintain(in the infinity
        future).

        i.e. $$\mathbf{P} \times \mathbf{p}_{trend} = \mathbf{p}_{trend}$$

        To avoid inacuraccies when dealing with floating point calculations,
        the frequency matrix is used:
        $$\mathbf{P} \times \mathbf{p}_{trend} = \mathbf{p}_{trend}$$
        $$\Rightarrow (\mathbf{P - E}) \times \mathbf{p}_{trend} = 0$$
        $$\Rightarrow(\mathbf{F}\begin{bmatrix}
            (\sum col_1) & 0 &  \ldots & 0\\
            0 & (\sum col_2) &  \ldots & 0\\
            \vdots & \vdots & \ddots & \vdots \\
            0 & 0 &  \ldots & (\sum col_n)\\
        \end{bmatrix}^{-1} - E) \times \mathbf{p}_{trend} = 0$$
        $$\Rightarrow(\mathbf{F} - \begin{bmatrix}
            (\sum col_1) & 0 &  \ldots & 0\\
            0 & (\sum col_2) &  \ldots & 0\\
            \vdots & \vdots & \ddots & \vdots \\
            0 & 0 &  \ldots & (\sum col_n)\\
        \end{bmatrix})\begin{bmatrix}
            (\sum col_1) & 0 &  \ldots & 0\\
            0 & (\sum col_2) &  \ldots & 0\\
            \vdots & \vdots & \ddots & \vdots \\
            0 & 0 &  \ldots & (\sum col_n)\\
        \end{bmatrix}^{-1} \times \mathbf{p}_{trend} = 0$$
        $$\Rightarrow \mathbf{p}_{trend} = \mathbf{S}\times N(\mathbf{F - S})$$

        Returns:
            list[sympy.Matrix]: The possible $$\mathbf{p}_{trend}$$ probability vectors.
        """
        S = sympy.zeros(self.n_unique_states)
        for col in range(self.n_unique_states):
            col_sum = 0
            for row in range(self.n_unique_states):
                col_sum += self._freq_mat[row, col]
            S[col, col] = col_sum
        F_minus_S: sympy.Matrix = self._freq_mat - S
        nullspace = F_minus_S.nullspace()
        if len(nullspace) == 0:
            return None
        res = []
        for solve in nullspace:
            p_trend: sympy.Matrix = S @ solve
            res.append(p_trend / sum(p_trend))
        return res

    def _embed(self, state_sequence: np.ndarray) -> np.ndarray:
        """Transform the input array of states to an array of indicies based on
        the `self._state_to_index` dict.

        Args:
            state_sequence (numpy.ndarray): The state sequence to embed.
        Returns:
            numpy.ndarray: The index array.
        """
        res: np.ndarray = np.zeros((len(state_sequence)), dtype = int)
        for i in range(len(state_sequence)):
            if not state_sequence[i] in self._state_to_index.keys():
                raise RuntimeError(f"{state_sequence[i]} not found in possible states: {self._state_to_index.keys()}.")
            res[i] = self._state_to_index[state_sequence[i]]
        return res

    def _unembed(self, embeded_data: np.ndarray) -> np.ndarray:
        """Transform the input array of indicies to an array of states based on
        the `self._index2state` dict.

        Returns:
            numpy.ndarray: The state array.
        """
        res = []
        for i in range(len(embeded_data)):
            if not data[i] in self._index2state.keys():
                raise RuntimeError(f"{data[i]} not found in possible indicies: {self._index2state.keys()}.")
            res[i] = self._index2state[embeded_data[i]]
        return res

    def _embed_transform(self, state_sequence: np.ndarray) -> np.ndarray:
        """Extract all the possible states from the data.

        The extracted array is stored in `self._state2index` &
        `self._index2state`.

        Args:
            data (numpy.ndarray): A state sequence.
        Returns:
            numpy.ndarray: The index sequence.
        """
        # Transform.
        for state in np.unique(state_sequence):
            self._state_to_index[state] = len(self._state_to_index)
            self._index2state[len(self._index2state)] = state
        return self._embed(state_sequence)

    def _calculate_prob_mat(self, index_sequence: np.ndarray) -> None:
        """Calculate the probabiliy matrix of the markov model.

        Element $$p_{i, j}$$ should store the possibility of $$A_i$$ occuring after $$A_j$$.
        This method also automatically resizes the probability matrix according
        to the length of `self.n_unique_states`.

        Note: This method calls `_cal_freq_mat` to get the frequencies first.
        Args:
            index_sequence (numpy.ndarray): The index sequence from oldest to
            newest.
        """
        # Resize the probability matrix to avoid out of bounds accesses.
        self._prob_mat: sympy.Matrix = sympy.zeros(self.n_unique_states)
        # Calculate the elements from the frequency matrix.
        self._calculate_freq_mat(index_sequence)
        for col in range(self.n_unique_states):
            col_sum = 0
            for row in range(self.n_unique_states):
                col_sum += self._freq_mat[row, col]
            if col_sum == 0:
                for row in range(self.n_unique_states):
                    self._prob_mat[row, col] = 1 / self.n_unique_states
            else:
                for row in range(self.n_unique_states):
                    self._prob_mat[row, col] = self._freq_mat[row, col]/col_sum
    def _calculate_freq_mat(self, index_sequence: np.ndarray) -> None:
        """Calculate the frequency matrix of the markov model.

        Element $$f_{i, j}$$ should store the number of times where $$A_i$$ occured after $$A_j$$.
        This method also automatically resizes the frequency matrix according
        to the length of `self.n_unique_states`.

        Args:
            index_sequence (numpy.ndarray): The index sequence from oldest to
            newest.
        """
        # Resize the frequency matrix to avoid out of bounds accesses.
        self._freq_mat: sympy.Matrix = sympy.zeros(self.n_unique_states)
        # Iterate over the index data points to construct the frequency matrix.
        for i in range(len(index_sequence) - 1):
            old_index = index_sequence[i]
            new_index = index_sequence[i + 1]
            self._freq_mat[new_index, old_index] += 1

    @property
    def probability_matrix(self) -> sympy.Matrix:
        """Probability Matrix.

        The dimension should be the same as the number of unique values in the
        training data sequence.
        """
        return self._prob_mat

    @property
    def frequency_matrix(self) -> sympy.Matrix:
        """Frequency Matrix.

        The dimension should be the same as the number of unique values in the
        training data sequence.
        """
        return self._freq_mat

    @property
    def unique_states(self) -> np.ndarray:
        """Every Possible State.

        This property is inferred from the training data sequence.
        """
        return np.array(list(self._state_to_index.keys()))
    
    @property
    def unique_indicies(self) -> np.ndarray:
        """The indicies for probability encoding.
        """
        return np.array(list(self._index2state.keys()))

    @property
    def n_unique_states(self) -> int:
        """Number of unique values.
        """
        return len(self._state_to_index)

    @property
    def state_to_index_map(self) -> dict[Any, int]:
        """The mapping of the actual states to the probability encoding
        indicies.
        """
        return self._index2state

    @property
    def index_to_state_map(self) -> dict[int, Any]:
        """The mapping of probability encoding indicies to the actual states.
        """
        return self._index2state

"""
This section is used for testing.
"""
if __name__ == "__main__":
    data = np.array([1,4,2,1,4,2,4,4,1,3,3,1,3,3,4,4,4,2,4,4])
    model = MarkovModel()
    model.fit(data)
    print(model.n_unique_states)
    print(model.unique_indicies)
    print(model.unique_states)
    print(model.frequency_matrix)
    print(model.probability_matrix)
    print(model.predict_proba(data, 3))
    print(model.predict(data, 3))
    print(model.probability_matrix)
    print(model.trend_proba())
    print(model.trend())