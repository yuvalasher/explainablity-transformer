def print_segmentation_results(pixAcc: float, mAp: float, mIoU: float, mF1: float) -> None:
    print(f"Pixel-wise Accuracy: {round(pixAcc * 100, 4)}")
    print(f"Mean AP over {2} classes: {round(mAp * 100, 4)}")
    print(f"Mean IoU over {2} classes: {round(mIoU * 100, 4)}")
    print(f"Mean F1 over {2} classes:{round(mF1 * 100, 4)}")