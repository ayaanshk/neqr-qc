import numpy as np


def prf(pred, ref):
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    tp, fp, fn = (pred & ref).sum(), (pred & ~ref).sum(), (~pred & ref).sum()
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=float(p), recall=float(r), f1=float(f), exact=bool((pred == ref).all()))
