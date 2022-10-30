def print_segmentation_results(pixAcc: float, mAp: float, mIoU: float, mF1: float) -> None:
    print("mIoU over %d classes: %.4f" % (2, mIoU))
    print("Pixel-Accuracy: %2.2f%%" % (pixAcc * 100))
    print("AP over %d classes: %.4f" % (2, mAp))
    print("F1 over %d classes: %.4f" % (2, mF1))
