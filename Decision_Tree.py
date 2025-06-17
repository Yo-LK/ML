# 엔트로피 계산
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))


def information_gain(X_column, y, threshold):
    left_idx = X_column <= threshold
    right_idx = X_column > threshold

    if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
        return 0

    y_left, y_right = y[left_idx], y[right_idx]
    H_before = entropy(y)
    H_after = (len(y_left) / len(y)) * entropy(y_left) + (len(y_right) / len(y)) * entropy(y_right)

    return H_before - H_after


# 최적 분할 찾기
def best_split(X, y):
    best_gain = -1
    best_feature = None
    best_threshold = None

    n_samples, n_features = X.shape

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            gain = information_gain(X[:, feature], y, t)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold


class Node:
    def __init__(self, feature=None,
                       threshold=None,
                       left=None,
                       right=None,
                       value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def build_tree(X, y, depth=0):
    # 해당 tree 구역 내의 class의 갯수가 1개의 경우 : 리프노트
    if len(np.unique(y)) == 1:
        return Node(value=y[0])

    # 최적의 특성과 threshold 탐색
    feature, threshold = best_split(X, y)


    # 예외처리: 어떻게 나눠도 entropy가 줄어들지 않는 경우 --> 현재 상태 그대로 return
    if feature is None:
        most_common = np.bincount(y).argmax()
        return Node(value=most_common)

    # 탐색한 threshold를 기준으로 데이터 분류
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold

    x_left = X[left_idx]
    y_left = y[left_idx]

    x_right = X[right_idx]
    y_right = y[right_idx]


    # 각각의 그룹에 대해 Decision tree 적용
    left = build_tree(x_left, y_left, depth + 1)
    right = build_tree(x_right, y_right, depth + 1)

    return Node(feature=feature, threshold=threshold, left=left, right=right)


