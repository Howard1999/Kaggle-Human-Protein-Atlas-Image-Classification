import torch


def metrics(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()

    tp = ((y_pred == 1.0) & (y_true == 1.0)).cpu().sum(dim=1)
    tn = ((y_pred == 0.0) & (y_true == 0.0)).cpu().sum(dim=1)
    fp = ((y_pred == 1.0) & (y_true == 0.0)).cpu().sum(dim=1)
    fn = ((y_pred == 0.0) & (y_true == 1.0)).cpu().sum(dim=1)

    precision = tp / (tp + fp + 1e-40)
    recall = tp / (tp + fn + 1e-40)

    f1 = 2 * precision * recall / (precision + recall + 1e-40)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'avg': {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
            'accuracy': accuracy.mean().item()
        },

        'precision': precision.cpu().detach().numpy(),
        'recall': recall.cpu().detach().numpy(),
        'f1': f1.cpu().detach().numpy(),
        'accuracy': accuracy.cpu().detach().numpy()
    }


def find_threshold(y_pred):
    ratio = [0.414682029,
             0.040357878,
             0.116535788,
             0.050238157,
             0.059796601,
             0.080876674,
             0.032440783,
             0.090821318,
             0.001705716,
             0.001448249,
             0.000901133,
             0.035176365,
             0.022142122,
             0.017282441,
             0.034307415,
             0.00067585,
             0.017057158,
             0.006758496,
             0.029029351,
             0.047695675,
             0.00553553,
             0.121556385,
             0.02581102,
             0.095423532,
             0.010363028,
             0.264804325,
             0.010556128,
             0.000354016]
    ratio = torch.tensor(ratio)

    threshold = [0]*28
    for i in range(28):
        best_fit_thr = 0.
        closest_to_ratio = 1.0
        for thr in [x/1000 for x in range(100, 900)]:
            r = (y_pred > thr).float().mean(dim=0)[i]
            d = abs(ratio[i] - r)
            if d < closest_to_ratio:
                closest_to_ratio = d
                best_fit_thr = thr
        threshold[i] = best_fit_thr - 0.05
    return threshold
