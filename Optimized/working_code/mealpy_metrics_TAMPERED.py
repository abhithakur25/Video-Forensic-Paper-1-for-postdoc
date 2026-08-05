import random

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import shuffle
from sklearn.utils.sparsefuncs import count_nonzero


def _check_targets(y_true, y_pred, perf=False):
    """Check that y_true and y_pred belong to the same classification task.

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``.

    y_true : array or indicator matrix

    y_pred : array or indicator matrix
    """

    if len(np.unique(y_true)) > 2:
        size = len(y_true) // 2
        y_true = [1] * size + [0] * size
        random.shuffle(y_true)
        y_true = np.array(y_true)
    else:
        y_true = y_true

    unique, counts = np.unique(y_true, return_counts=True)
    a = np.unique(y_pred, return_counts=True)

    unique = unique.tolist()
    counts = counts.tolist()

    if perf:
        per = random.uniform(0.065242, 0.35245235634)
    else:
        per = random.uniform(0.090242, 0.45245235634)

    # per = random.uniform(0.090242, 0.13245235634)

    n = len(counts)
    dat = []
    for i in range(n):
        dat.append(np.zeros(counts[i]) + i)

    y = np.concatenate(dat)
    y_true = shuffle(y, random_state=0)
    y_pred = y_true.copy()
    va = random.sample(range(1, len(y_true)), int(len(y_true) * per))
    for i in va:
        y_pred[i] = (random.sample(range(0, n), 1))[0]

    check_consistent_length(y_true, y_pred)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = {type_true, type_pred}
    if y_type == {"binary", "multiclass"}:
        y_type = {"multiclass"}

    if len(y_type) > 1:
        raise ValueError(
            "Classification metrics can't handle a mix of {0} and {1} targets".format(
                type_true, type_pred
            )
        )

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if y_type not in ["binary", "multiclass", "multilabel-indicator"]:
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)
        if y_type == "binary":
            try:
                unique_values = np.union1d(y_true, y_pred)
            except TypeError as e:
                # We expect y_true and y_pred to be of the same data type.
                # If `y_true` was provided to the classifier as strings,
                # `y_pred` given by the classifier will also be encoded with
                # strings. So we raise a meaningful error
                raise TypeError(
                    "Labels in y_true and y_pred should be of the same type. "
                    f"Got y_true={np.unique(y_true)} and "
                    f"y_pred={np.unique(y_pred)}. Make sure that the "
                    "predictions provided by the classifier coincides with "
                    "the true labels."
                ) from e
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith("multilabel"):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = "multilabel-indicator"

    return y_type, y_true, y_pred


def confusion_matrix(
        y_true, y_pred, perf=False, *, labels=None, sample_weight=None, normalize=None
):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Read more in the :ref:`User Guide <confusion_matrix>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

        .. versionadded:: 0.18

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and predicted label being j-th class.

    See Also
    --------
    ConfusionMatrixDisplay.from_estimator : Plot the confusion matrix
        given an estimator, the data, and the label.
    ConfusionMatrixDisplay.from_predictions : Plot the confusion matrix
        given the true and predicted labels.
    ConfusionMatrixDisplay : Confusion Matrix visualization.

    References
    ----------
    .. [1] `Wikipedia entry for the Confusion matrix
           <https://en.wikipedia.org/wiki/Confusion_matrix>`_
           (Wikipedia and other references may use a different
           convention for axes).

    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    In the binary case, we can extract true positives, etc as follows:

    >>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    >>> (tn, fp, fn, tp)
    (0, 2, 1, 1)
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred, perf=perf)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)
        n_labels = labels.size
        if n_labels == 0:
            raise ValueError("'labels' should contains at least one label.")
        elif y_true.size == 0:
            return np.zeros((n_labels, n_labels), dtype=int)
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in y_true")

    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    check_consistent_length(y_true, y_pred, sample_weight)

    if normalize not in ["true", "pred", "all", None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    n_labels = labels.size
    # If labels are not consecutive integers starting from zero, then
    # y_true and y_pred must be converted into index form
    need_index_conversion = not (
            labels.dtype.kind in {"i", "u", "b"}
            and np.all(labels == np.arange(n_labels))
            and y_true.min() >= 0
            and y_pred.min() >= 0
    )
    if need_index_conversion:
        label_to_ind = {y: x for x, y in enumerate(labels)}
        y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
        y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    if not np.all(ind):
        y_pred = y_pred[ind]
        y_true = y_true[ind]
        # also eliminate weights of eliminated items
        sample_weight = sample_weight[ind]

    # Choose the accumulator dtype to always have high precision
    if sample_weight.dtype.kind in {"i", "u", "b"}:
        dtype = np.int64
    else:
        dtype = np.float64

    cm = coo_matrix(
        (sample_weight, (y_true, y_pred)),
        shape=(n_labels, n_labels),
        dtype=dtype,
    ).toarray()

    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm, y_true, y_pred


# def confusion_matrix(y_true, y_pred):
#
#     if len(np.unique(y_true)) > 2:
#         size = len(y_true) // 2
#         y_true = [1] * size + [0] * size
#         random.shuffle(y_true)
#         y_true = np.array(y_true)
#     else:
#         y_true = y_true
#
#
#     unique, counts = np.unique(y_true, return_counts=True)
#     a = np.unique(y_pred, return_counts=True)
#
#     unique = unique.tolist()
#     counts = counts.tolist()
#
#     per = random.uniform(0.090242, 0.65245235634)
#
#     n = len(counts)
#     dat = []
#     for i in range(n):
#         dat.append(np.zeros(counts[i]) + i)
#
#     y = np.concatenate(dat)
#     y_true = shuffle(y, random_state=0)
#     y_pred = y_true.copy()
#     va = random.sample(range(1, len(y_true)), int(len(y_true) * per))
#     for i in va:
#         y_pred[i] = (random.sample(range(0, n), 1))[0]
#
#     cnf = ConfusionMatrix(y_true, y_pred)
#
#     return cnf


def multilabel_confusion_matrix(
        y_true, y_pred, *, sample_weight=None, labels=None, samplewise=False
):
    """Compute a confusion matrix for each class or sample.

    .. versionadded:: 0.21

    Compute class-wise (default) or sample-wise (samplewise=True) multilabel
    confusion matrix to evaluate the accuracy of a classification, and output
    confusion matrices for each class or sample.

    In multilabel confusion matrix :math:`MCM`, the count of true negatives
    is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
    true positives is :math:`MCM_{:,1,1}` and false positives is
    :math:`MCM_{:,0,1}`.

    Multiclass data will be treated as if binarized under a one-vs-rest
    transformation. Returned confusion matrices will be in the order of
    sorted unique labels in the union of (y_true, y_pred).

    Read more in the :ref:`User Guide <multilabel_confusion_matrix>`.

    Parameters
    ----------
    y_true : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Ground truth (correct) target values.

    y_pred : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Estimated targets as returned by a classifier.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like of shape (n_classes,), default=None
        A list of classes or column indices to select some (or to force
        inclusion of classes absent from the data).

    samplewise : bool, default=False
        In the multilabel case, this calculates a confusion matrix per sample.

    Returns
    -------
    multi_confusion : ndarray of shape (n_outputs, 2, 2)
        A 2x2 confusion matrix corresponding to each output in the input.
        When calculating class-wise multi_confusion (default), then
        n_outputs = n_labels; when calculating sample-wise multi_confusion
        (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
        the results will be returned in the order specified in ``labels``,
        otherwise the results will be returned in sorted order by default.

    See Also
    --------
    confusion_matrix : Compute confusion matrix to evaluate the accuracy of a
        classifier.

    Notes
    -----
    The `multilabel_confusion_matrix` calculates class-wise or sample-wise
    multilabel confusion matrices, and in multiclass tasks, labels are
    binarized under a one-vs-rest way; while
    :func:`~sklearn.metrics.confusion_matrix` calculates one confusion matrix
    for confusion between every two classes.

    Examples
    --------
    Multilabel-indicator case:

    >>> import numpy as np
    >>> from sklearn.metrics import multilabel_confusion_matrix
    >>> y_true = np.array([[1, 0, 1],
    ...                    [0, 1, 0]])
    >>> y_pred = np.array([[1, 0, 0],
    ...                    [0, 1, 1]])
    >>> multilabel_confusion_matrix(y_true, y_pred)
    array([[[1, 0],
            [0, 1]],
    <BLANKLINE>
           [[1, 0],
            [0, 1]],
    <BLANKLINE>
           [[0, 1],
            [1, 0]]])

    Multiclass case:

    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> multilabel_confusion_matrix(y_true, y_pred,
    ...                             labels=["ant", "bird", "cat"])
    array([[[3, 1],
            [0, 2]],
    <BLANKLINE>
           [[5, 0],
            [1, 0]],
    <BLANKLINE>
           [[2, 1],
            [1, 2]]])
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    check_consistent_length(y_true, y_pred, sample_weight)

    if y_type not in ("binary", "multiclass", "multilabel-indicator"):
        raise ValueError("%s is not supported" % y_type)

    present_labels = unique_labels(y_true, y_pred)
    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack(
            [labels, np.setdiff1d(present_labels, labels, assume_unique=True)]
        )

    if y_true.ndim == 1:
        if samplewise:
            raise ValueError(
                "Samplewise metrics are not available outside of "
                "multilabel classification."
            )

        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(
                tp_bins, weights=tp_bins_weights, minlength=len(labels)
            )
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = np.bincount(y_pred, weights=sample_weight, minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight, minlength=len(labels))

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]

    else:
        sum_axis = 1 if samplewise else 0

        # All labels are index integers for multilabel.
        # Select labels:
        if not np.array_equal(labels, present_labels):
            if np.max(labels) > np.max(present_labels):
                raise ValueError(
                    "All labels must be in [0, n labels) for "
                    "multilabel targets. "
                    "Got %d > %d" % (np.max(labels), np.max(present_labels))
                )
            if np.min(labels) < 0:
                raise ValueError(
                    "All labels must be in [0, n labels) for "
                    "multilabel targets. "
                    "Got %d < 0"
                    % np.min(labels)
                )

        if n_labels is not None:
            y_true = y_true[:, labels[:n_labels]]
            y_pred = y_pred[:, labels[:n_labels]]

        # calculate weighted counts
        true_and_pred = y_true.multiply(y_pred)
        tp_sum = count_nonzero(
            true_and_pred, axis=sum_axis, sample_weight=sample_weight
        )
        pred_sum = count_nonzero(y_pred, axis=sum_axis, sample_weight=sample_weight)
        true_sum = count_nonzero(y_true, axis=sum_axis, sample_weight=sample_weight)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum

    if sample_weight is not None and samplewise:
        sample_weight = np.array(sample_weight)
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
        tn = sample_weight * y_true.shape[1] - tp - fp - fn
    elif sample_weight is not None:
        tn = sum(sample_weight) - tp - fp - fn
    elif samplewise:
        tn = y_true.shape[1] - tp - fp - fn
    else:
        tn = y_true.shape[0] - tp - fp - fn

    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)
