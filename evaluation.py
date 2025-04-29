import pandas as pd
import numpy as np
import sklearn.metrics


def score(solution: pd.DataFrame, wildnet_predictions: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Computes macro-averaged ROC-AUC, ignoring classes with no positive labels.
    """
    solution = solution.drop(columns=[row_id_column_name])
    wildnet_predictions = wildnet_predictions.drop(columns=[row_id_column_name])

    # Find columns with at least one positive label
    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)

    if not scored_columns:
        raise ValueError("No valid classes with positive labels.")

    return sklearn.metrics.roc_auc_score(solution[scored_columns], wildnet_predictions[scored_columns], average='macro')


def evaluate_model(model, val_dataset, row_id_column_name):
    y_true_list = []
    y_pred_list = []
    row_ids = []

    for batch in val_dataset:
        x_val, y_val = batch  # Unpack features & labels
        y_pred = model.predict(x_val)  # Get model predictions

        y_true_list.append(y_val.numpy())  # Convert Tensor to NumPy array
        y_pred_list.append(y_pred)

    # Convert lists to arrays
    y_true = np.vstack(y_true_list)  # Shape: (num_samples, num_classes)
    y_pred = np.vstack(y_pred_list)  # Shape: (num_samples, num_classes)

    # Create Pandas DataFrames
    solution_df = pd.DataFrame(y_true, columns=[f"class_{i}" for i in range(y_true.shape[1])])
    wildnet_predictions_df = pd.DataFrame(y_pred, columns=[f"class_{i}" for i in range(y_pred.shape[1])])

    # Add row IDs (if required)
    solution_df[row_id_column_name] = range(len(solution_df))
    wildnet_predictions_df[row_id_column_name] = range(len(wildnet_predictions_df))

    # Compute metric
    auc_score = score(solution_df, wildnet_predictions_df, row_id_column_name)
    return auc_score


# Example usage:
auc = evaluate_model(model, val_dataset, "row_id")
print("Validation Macro-Averaged ROC-AUC Score:", auc)
