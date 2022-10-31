def print_segmentation_results(pixAcc: float, mAp: float, mIoU: float, mF1: float) -> None:
    print(f"\nPixel-wise-Accuracy {round(pixAcc * 100, 4)}%")
    print(f"mAP {round(mAp * 100, 4)}%")
    print(f"mIoU {round(mIoU * 100, 4)}%")
    print(f"F1 {round(mF1 * 100, 4)}%")
