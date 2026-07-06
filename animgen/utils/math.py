def range_norm(t, lb=None, ub=None, offset=None, eps=1e-8):
    """
    Given tensor of continuous values, return corresponding range normalized values.
    """
    if lb is None:
        lb = t.min() - offset if offset else t.min()
    if ub is None:
        ub = t.max()
    return (t - lb) / (ub - lb + eps)
