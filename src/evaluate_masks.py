import numpy as np

def calculate_metrics(prediction, reference):
    """
    prediction and reference must be binary masks of the same shape.
    Nonzero pixels are treated as foreground.
    """
    pred = np.asarray(prediction) > 0
    ref = np.asarray(reference) > 0

    if pred.shape != ref.shape:
        raise ValueError(
            f"Mask shapes differ: {pred.shape} vs {ref.shape}"
        )

    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    pred_area = pred.sum()
    ref_area = ref.sum()

    iou = intersection / union if union else 1.0
    dice_denominator = pred_area + ref_area
    dice = (
        2 * intersection / dice_denominator
        if dice_denominator else 1.0
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "prediction_area": int(pred_area),
        "reference_area": int(ref_area),
        "intersection": int(intersection),
    }

if __name__ == "__main__":
    # Sanity check using small artificial masks.
    reference = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)

    result = calculate_metrics(prediction, reference)
    print("Sanity-check metrics:", result)
    print("Evaluation function is ready.")
