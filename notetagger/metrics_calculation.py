from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


def calculate_performance_metrics(y_true, y_pred_prob):
    """
    Takes a true label and the predicted output by a model and returns a json with metrics calculated
    at various thresholds

    Arguments:
        y_true (array): true outcome labels
        y_pred_prob (array): predicted probability outcome labels (should be floats)

    Returns:
        performance_metrics (json): json with scores at various thresholds
    """

    # initalize dictionary and create auc
    performance_metrics = {
        "auc": '{:.4f}'.format(roc_auc_score(y_true=y_true, y_score=y_pred_prob)),
        "metrics_by_threshold": []
    }

    # loop through probability thresholds of .3 - .9 incremented by .1
    # and calculate accuracy, precision, and recall for each threshold
    for threshold in range(3, 10):
        threshold /= 10
        y_pred = y_pred_prob > threshold
        performance_metrics["metrics_by_threshold"].append(
            {"threshold": '{:.1f}'.format(threshold),
             "accuracy": '{:.4f}'.format(accuracy_score(y_true=y_true, y_pred=y_pred)),
             "precision": '{:.4f}'.format(precision_score(y_true=y_true, y_pred=y_pred)),
             "recall": '{:.4f}'.format(recall_score(y_true=y_true, y_pred=y_pred))}
        )

    return performance_metrics
