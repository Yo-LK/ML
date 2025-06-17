from math import dist
def L2_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
    # initialization
        self.k = k

    def fit(self, X, y):
        # Storage training datas
        self.train_x = X
        self.train_y = y

    def predict(self, X):
        # Prediction
        # 새로운 데이터 X와 train_x간 거리 측정
        # X = [num_data, x1, x2]
        y_pred = []

        for input in X:
            distances = []

            for train_x in self.train_x:
                distances.append(L2_distance(train_x, input))

            # 거리를 기준으로 sorting
            index = np.argsort(distances)

            # sorting 된 data 중 상위 k개의 label 갯수 확인
            index = index[:self.k]

            label = []

            for i in index:
                label.append(self.train_y[i])

            # 최대 갯수 label 출력
            prediction = max(label, key=label.count)
            y_pred.append(prediction)

        return y_pred


