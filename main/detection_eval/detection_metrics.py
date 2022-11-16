import pandas as pd
import numpy as np
from typing import List


def calculate_AP(IOUs: List[float], iou_threshold: float) -> float:
    eval_table = pd.DataFrame({'idx': range(len(IOUs)), 'IOU': IOUs})
    eval_table['TP/FP'] = eval_table['IOU'].apply(lambda x: 'TP' if x >= iou_threshold else 'FP')

    Precision = []
    Recall = []
    TP = FP = 0
    # assuming that we have a very bad model which misclassified all the objects. so all true positive becomes false negative
    FN = len(eval_table['TP/FP'] == 'TP')
    for index, row in eval_table.iterrows():

        if row.IOU > iou_threshold:
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
            x = eval_table[eval_table['Recall'] >= recall_level]['Precision']
            prec = max(x)
        except:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
    # print('11 point precision is ', prec_at_rec)
    # print('mAP is ', avg_prec)
    return avg_prec
