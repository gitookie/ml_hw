import numpy as np
import math

class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.threshold = None
        self.alpha = None
        self.feature_index = None
    
    def fit(self):
        pass
    
    def predict(self):
        pass

class Adaboost():
    """相当于一个强分类器"""

    def __init__(self):
        self.num_classifier = 5
        self.classifiers = []
        self.w = None

    def fit(self, features, y, appended = 0):
        """appended用来指示是不是在已有的adaboost分类器的基础上继续训练, 如果是则沿用之前的w, 如果是从头训, 则w重新初始化一下"""

        num_samples = features.shape[0]
        num_features = features.shape[1]
        predictions = np.ones(y.shape)

        if appended == 0:    
            self.w = np.ones(num_samples) / num_samples
        
        for i in range(num_features):
            if i % 300 == 0:
                print(f'current feature:{i}/{num_features}')
            
            for j in range(self.num_classifier):    # 针对每一个特征，训练num_classifier个弱分类器
                # if j % 300 == 0:
                #     print(f'the {j} th weak classifier for the feature {i}/{self.num_classifier}')
                stump = DecisionStump()
                stump.feature_index = i
                cur_features = features[:, i]
                unique = np.unique(cur_features)
                min_error = math.inf
                # k = 0
                for threshold in unique:    # 下面找出最好的阈值（因为目前是用决策树桩作弱分类器）
                    # if k % 300 == 0:
                    #     print(f'the {k} th candidate threshold among {len(unique)} threshold')
                    # k += 1
                    predictions = np.ones(y.shape)
                    predictions[cur_features < threshold] = -1
                    # predictions[~(cur_features < threshold)] = -1
                    error = np.sum(self.w[predictions != y])
                    if error > 0.5:     # 如果当前的方向误差超过0.5，其实就意味着
                        # 可以反向预测一下，即此时变成大于阈值则判定为-1（原来是
                        # 小于阈值判定为-1），算是一个小技巧
                        stump.polarity = -1
                        error = 1 - error
                    if error < min_error:
                        stump.threshold = threshold
                        min_error = error
                stump.alpha = 0.5 * math.log((1 - min_error) / (min_error + 1e-10))
                # print(f'the {j} th weak classifier accuracy:{1 - min_error}')
                # print(f'the {j} the weak classifier weight:{stump.alpha}')
                predictions = np.ones(y.shape)  # 找到了最好的阈值并决定用它了，再重新算一下当前弱分类器的预测结果
                predictions[stump.polarity * cur_features < stump.polarity * stump.threshold] = -1     # 这里的polarity很关键
                
                self.w = self.w * np.exp(-stump.alpha * y * predictions)
                # 这里用y * predictions的结果直接表示是否预测正确，比较巧妙，刚好是二分类问题
                # 且是用1和-1表示正负类
                
                self.w /= np.sum(self.w)    # 别忘了归一化权重
                self.classifiers.append(stump)    

    def append_features_fit(self, appended_features, y):
        """在已有的adaboost分类器的基础上增加新的特征, 然后继续进行训练"""

        num_samples = appended_features.shape[0]
        num_features = appended_features.shape[1]
        predictions = np.ones(y.shape)
        for i in range(num_features):
            if i % 300 == 0:
                print(f'current feature:{i}/{num_features}')
            self.w = np.ones(num_samples) / num_samples
            for j in range(self.num_classifier):    # 针对每一个特征，训练num_classifier个弱分类器
                # if j % 300 == 0:
                #     print(f'the {j} th weak classifier for the feature {i}/{self.num_classifier}')
                stump = DecisionStump()
                stump.feature_index = i
                cur_features = features[:, i]
                unique = np.unique(cur_features)
                min_error = math.inf
                # k = 0
                for threshold in unique:    # 下面找出最好的阈值（因为目前是用决策树桩作弱分类器）
                    # if k % 300 == 0:
                    #     print(f'the {k} th candidate threshold among {len(unique)} threshold')
                    # k += 1
                    predictions = np.ones(y.shape)
                    predictions[cur_features < threshold] = -1
                    # predictions[~(cur_features < threshold)] = -1
                    error = np.sum(self.w[predictions != y])
                    if error > 0.5:     # 如果当前的方向误差超过0.5，其实就意味着
                        # 可以反向预测一下，即此时变成大于阈值则判定为-1（原来是
                        # 小于阈值判定为-1），算是一个小技巧
                        stump.polarity = -1
                        error = 1 - error
                    if error < min_error:
                        stump.threshold = threshold
                        min_error = error
                stump.alpha = 0.5 * math.log((1 - min_error) / (min_error + 1e-10))
                print(f'the {j} th weak classifier accuracy:{1 - min_error}')
                print(f'the {j} the weak classifier weight:{stump.alpha}')
                predictions = np.ones(y.shape)  # 找到了最好的阈值并决定用它了，再重新算一下当前弱分类器的预测结果
                predictions[stump.polarity * cur_features < stump.polarity * stump.threshold] = -1     # 这里的polarity很关键
                
                self.w = self.w * np.exp(-stump.alpha * y * predictions)
                # 这里用y * predictions的结果直接表示是否预测正确，比较巧妙，刚好是二分类问题
                # 且是用1和-1表示正负类
                
                self.w /= np.sum(self.w)    # 别忘了归一化权重
                self.classifiers.append(stump)    

    def predict(self, features):
        """输入图像特征, 判定是否为人脸"""

        predictions = np.zeros(features.shape[0])
        for i in range(len(self.classifiers)):
            # if i % 300 == 0:
            #     print(f'the {i} th weak classifier among {len(self.classifiers)} classifiers')
            cur_classifier = self.classifiers[i]
            cur_features = features[:, cur_classifier.feature_index]
            threshold = cur_classifier.threshold
            cur_predictions = np.ones(features.shape[0])
            cur_predictions[cur_classifier.polarity * cur_features < cur_classifier.polarity * threshold] = -1
            predictions += cur_classifier.alpha * cur_predictions
        predictions = np.sign(predictions)
        return predictions
    
"""predictions = np.array([1, 1, 1, 1, 1])
y = np.array([1, -1, -1, -1, 1])
features = np.array([1, 1, -1, 1, -1])
threshold = 0.5
# predictions[features < threshold] = -1
# predictions[~(features < threshold)] = -1
# print(predictions)
w = np.ones(5) / 5
error = np.sum(w[predictions != y])     # 这里不进行算数, 而是进行索引操作, 所以这里的predictions != y的结果应该还是布尔变量
# print(error)
error2 = np.sum(w * (predictions != y))"""  # 这里要进行乘法，则predictions != y的结果应该会被改成整数


"""print(predictions != y)
print(error2)
print(np.exp(-0.2 * (predictions == y)))
print(np.exp(-0.2 * y * predictions))"""
"""neg_idx = (-1 * features < -1 * threshold)
neg_idx2 = (-1 * (features < threshold))
print(neg_idx2)
print(neg_idx)
# print(predictions)
predictions[neg_idx] = -1
# print(predictions)"""

"""adaboost = Adaboost()
features = np.array([[1, 2, 3]])
y = np.array([1])
adaboost.fit(features, y)"""