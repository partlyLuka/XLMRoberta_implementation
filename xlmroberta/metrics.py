from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def get_metrics(actual, predicted):
    metrics = {'accuracy': accuracy_score(actual, predicted),
               'recall': recall_score(actual, predicted, average="macro"),
               'precision': precision_score(actual, predicted, average="macro"),
               'f1': f1_score(actual, predicted, average="macro")}

    return metrics
