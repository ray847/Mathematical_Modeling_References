# Imports
import numpy as np
import pandas as pd

def positively_correlated(data: pd.DataFrame,
                         targets: list = []) -> pd.DataFrame:
    """Make the input data positively correleted.

    Args:
        data (pandas.DataFrame): The data to transform. Every column should be
        in the same catagory. Different samples should accommodate different
        rows.
        targets (list, optional): The targets for all catagories of data. Every
        targt must be either one of the following: "min", "max", or (ideal_min,
        idal_max). Defaults to empty list (all "max").

    Raises:
        ValueError: Raised when the targets parameter doesn't fulfill the
        requirements.

    Returns:
        pd.DataFrame: Positively correlated data.
    """
    # Fill the empty targets with "max".
    targets = targets + ["max" for i in range(data.shape[1] - len(targets))]
    # Iterate over every column.
    for col in range(data.shape[1]):
        target = targets[col]
        if target == "min":
            data.iloc[:, col] = data.iloc[:, col].max() - data.iloc[:, col]
        elif target == "max":
            pass     # already positively correlated
        elif type(target) == float or type(target) == int: # Arithmetic targets
            M = (data.iloc[:, col] - target).abs().max()  # Max distance
            data.iloc[:, col] = 1 - (data.iloc[:, col] - target) / M
        elif type(target) == tuple and len(target) == 2:
            a = target[0]
            b = target[1]
            M = max(a - data.iloc[:, col].min(), data.iloc[:, col].max() - b)
            for i in range(data.shape[0]):
                if data.iloc[i, col] < a:
                    data.iloc[i, col] = 1 - (a - data.iloc[i, col]) / M
                elif data.iloc[i, col] > b:
                    data.iloc[i, col] = 1 - (data.iloc[i, col] - b) / M
                else:
                    data.iloc[i, col] = 1
        else:
            raise ValueError("Invalid target. Please choose from'min','max', float, int, or tuple.")
    return data

def normalized(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize every column of the input dataframe.

    Args:
        data (pandas.DataFrame): The dataframe to normalize.

    Returns:
        pandas.DataFrame: Normalized data.
    """
    for col in range(data.shape[1]):  # iterate over each column
        D = (data.iloc[:, col].pow(2).sum())**0.5
        data = data.astype(float)
        data.iloc[:, col] /= D
    return data

def evaluate(data: pd.DataFrame,
           targets = [],
           weights = []) -> np.ndarray:
    """Calculates the scores for every alternative in the input dataframe.

    Args:
        data (pandas.DataFrame): The dataframe containing the data for every
         alternative. Every column should indicate different catagories. And
         every alternative should accommodate a single row.
        targets (list, optional): The targets for every catagory of the data.
         Every target should be either one of the following: "max", "min", or
         (ideal_min, ideal_max). Empty values defaults to "max". Defaults to [].
        weights (list, optional): Weights for every catagory of data. Empty
         values defaults to 1. Defaults to [].

    Returns:
        numpy.ndarray: An array containing the total scores for every
        alternative.
    """
    # Make the data matrix positively correlated.
    data = positively_correlated(data, targets)
    # Normalize the data matrix so that the norm of every column is 1.
    data = normalized(data)
    # Calculate the ideal best and worst values.
    data_best = []
    data_worst = []
    for col in range(data.shape[1]):  # iterate over each column
        data_best.append(data.iloc[:, col].max())
        data_worst.append(data.iloc[:, col].min())
    # Calculate the distance between each alternative and the ideal&worst values
    data_pos = []
    data_neg = []
    for i in range(data.shape[0]):  # iterate over each alternative
        # apply weights if provided
        weight = 1
        if i < len(weights) - 1:
            weight = weights[i] if weights[i] != 0 else 1e-6
        data_pos.append(
            (weight * (data.iloc[i, :] - data_best).pow(2)).sum() ** 0.5
        )
        data_neg.append(
            (weight * (data.iloc[i, :] - data_worst).pow(2)).sum() ** 0.5
        )
    # Calculate the final scores.
    scores = []
    for i in range(data.shape[0]):  # iterate over each alternative
        score = data_neg[i] / (data_neg[i] + data_pos[i])
        scores.append(score)
    # Standardize the scores.
    scores = scores / sum(scores)
    return np.array(scores)

"""
Testing
"""
if __name__ == "__main__":
    data = pd.DataFrame([
        [89, 2],
        [60, 0],
        [74, 1],
        [99, 3]
    ])

    print(evaluate(data))