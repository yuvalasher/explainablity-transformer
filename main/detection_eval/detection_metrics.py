import pandas as pd
import numpy as np
from typing import List

import sklearn.metrics


# def precision_recall_curve(y_true, pred_scores, thresholds):
#     precisions = []
#     recalls = []
#
#     for threshold in thresholds:
#         y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]
#
#         precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
#         recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
#
#         precisions.append(precision)
#         recalls.append(recall)
#
#     return precisions, recalls

# def calculate_AP_by_rectangles(IOUs: List[float]):
#     y_true = ["positive", "negative", "positive", "negative", "positive",
#               "positive"]  # , "positive", "negative", "positive", "negative"]
#
#     # pred_scores = [0.7, 0.3, 0.5, 0.6, 0.55, 0.9]  # , 0.75, 0.2, 0.8, 0.3]
#     pred_scores = IOUs
#     thresholds = np.arange(start=0.2, stop=0.9, step=0.05)
#
#     precisions, recalls = precision_recall_curve(y_true=y_true,
#                                                  pred_scores=pred_scores,
#                                                  thresholds=thresholds)

    # plt.plot(recalls, precisions, linewidth=4, color="red", zorder=0)

    # plt.xlabel("Recall", fontsize=12, fontweight='bold')
    # plt.ylabel("Precision", fontsize=12, fontweight='bold')
    # plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    # plt.show()
    # precisions.append(1)
    # recalls.append(0)
    #
    # precisions = np.array(precisions)
    # recalls = np.array(recalls)
    # print(precisions)
    # print(recalls)
    # AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    # print(AP)


def calculate_AP(IOUs: List[float], iou_threshold: float) -> float:
    eval_table = pd.DataFrame({'idx': range(len(IOUs)), 'IOU': IOUs})
    eval_table['TP/FP'] = eval_table['IOU'].apply(lambda x: 'TP' if x >= iou_threshold else 'FP')

    Precision = []
    Recall = []
    TP = FP = 0
    # assuming that we have a very bad model which misclassified all the objects. so all true positive becomes false negative
    FN = len(eval_table['TP/FP'] == 'TP')
    for index, row in eval_table.iterrows():

        if row.IOU >= iou_threshold:
            TP = TP + 1
        else:
            FP = FP + 1
        try:
            AP = TP / (TP + FP)
            Rec = TP / (TP + FN)

        except ZeroDivisionError:
            AP = Recall = 0.0

        Precision.append(AP)
        Recall.append(Rec)

    eval_table['Precision'] = Precision
    eval_table['Recall'] = Recall

    eval_table['IP'] = eval_table.groupby('Recall')['Precision'].transform('max')

    prec_at_rec = []

    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            x = eval_table[eval_table['Recall'] >= recall_level]['Precision'] # TODO - is the has blog typo from from precision
            prec = max(x)
        except:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
    # print('11 point precision is ', prec_at_rec)
    # print('mAP is ', avg_prec)
    return avg_prec
