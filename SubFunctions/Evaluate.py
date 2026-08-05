from mealpy.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef





def Evaluation_Metrics(y, y_pred):
    """
    :param y_pred: Model predicted labels
    :param y: Original labels
    :return: return values is the evaluation metrics

    we get confusion matrix from the predictions
    Then implement the formula for getting the evaluation metrics from the confusion matrix
    """

    cnf_matrix, y_true, y_pred_ = confusion_matrix(y.astype(int), y_pred.astype(int))
    # cnf_matrix = np.sum(cnf_matrix, axis=0)
    TP = cnf_matrix[0, 0]
    FN = cnf_matrix[0, 1]
    FP = cnf_matrix[1, 0]
    TN = cnf_matrix[1, 1]
    # cnf_matrix = np.sum(cnf_matrix, axis=0)
    SEN = TP / (TP + FN)  # Sensitivity
    SPE = TN / (TN + FP)  # Specificity
    ACC = (TP + TN) / (TP + TN + FP + FN)
    FMS = (2 * TP) / (2 * TP + FP + FN)
    PRE = TP / (TP + FP)
    REC = SEN
    TS = TP / (TP + FP + FN)  # Threat score
    NPV = TN / (TN + FN)  # negative predictive value
    PPV = TP / (TP + FP)
    FOR = FN / (FN + TN)  # false omission rate
    F1 = 2 * TP / ((2 * TP) + FP + FN)
    MCC = matthews_corrcoef(y_true, y_pred_)
    # print([[TP, TN], [FP, FN]])     #
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [ACC, SEN, SPE, PRE, F1]


def Evaluation_Metrics1(y, y_pred):
    """
    :param y_pred: Model predicted labels
    :param y: Original labels
    :return: return values is the evaluation metrics

    we get confusion matrix from the predictions
    Then implement the formula for getting the evaluation metrics from the confusion matrix
    """

    cnf_matrix, y_true, y_pred_ = confusion_matrix(y.astype(int), y_pred.astype(int))
    # cnf_matrix = np.sum(cnf_matrix, axis=0)
    TP = cnf_matrix[0, 0]
    FN = cnf_matrix[0, 1]
    FP = cnf_matrix[1, 0]
    TN = cnf_matrix[1, 1]
    # cnf_matrix = np.sum(cnf_matrix, axis=0)
    SEN = TP / (TP + FN)  # Sensitivity
    SPE = TN / (TN + FP)  # Specificity
    ACC = (TP + TN) / (TP + TN + FP + FN)
    FMS = (2 * TP) / (2 * TP + FP + FN)
    PRE = TP / (TP + FP)
    REC = SEN
    TS = TP / (TP + FP + FN)  # Threat score
    NPV = TN / (TN + FN)  # negative predictive value
    PPV = TP / (TP + FP)
    FOR = FN / (FN + TN)  # false omission rate
    F1 = 2 * TP / ((2 * TP) + FP + FN)
    MCC = matthews_corrcoef(y_true, y_pred_)
    # print([[TP, TN], [FP, FN]])     #
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [TPR, FPR]


