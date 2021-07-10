import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# 线性回归模型
class linear_regression():
    def fit(self, train_X_in, train_Y, learning_rate=0.03, lamda=0.03, regularization="l2"):
        # 样本个数、样本的属性个数
        case_cnt, feature_cnt = train_X_in.shape
        # X矩阵X0向量
        train_X = np.c_[train_X_in, np.ones(case_cnt, )]
        # 初始化待调参数theta
        self.theta = np.zeros([feature_cnt + 1, 1])

        max_iter_num = sys.maxsize  # 最多迭代次数
        step = 0  # 当前已经迭代的次数
        pre_step = 0  # 上一次得到较好学习误差的迭代学习次数

        last_error_J = sys.maxsize  # 上一次得到较好学习误差的误差函数值
        threshold_value = 1e-6  # 定义在得到较好学习误差之后截止学习的阈值
        stay_threshold_times = 10  # 定义在得到较好学习误差之后截止学习之前的学习次数

        for step in range(0, max_iter_num):
            # 预测值
            pred = train_X.dot(self.theta)
            # 损失函数
            J_theta = sum((pred - train_Y) ** 2) / (2 * case_cnt)
            # 更新参数theta
            self.theta -= learning_rate * (lamda * self.theta + (train_X.T.dot(pred - train_Y)) / case_cnt)

            # 检测损失函数的变化值，提前结束迭代
            if J_theta < last_error_J - threshold_value:
                last_error_J = J_theta
                pre_step = step
            elif step - pre_step > stay_threshold_times:
                break

            # 定期打印，方便用户观察变化
            if step % 50 == 0:
                print("step %s: %.6f" % (step, J_theta))

    def predict(self, X_in):
        case_cnt = X_in.shape[0]
        X = np.c_[X_in, np.ones(case_cnt, )]
        pred = X.dot(self.theta)
        return pred


# 主函数
def main():
    # # 读取训练集和测试集文件
    # train_data = pd.read_csv("boston_house.train", header=None)
    # test_data = pd.read_csv("boston_house.test", header=None)
    #
    # # 对训练集和测试集进行X，Y分离
    # train_X, train_Y = feature_label_split(train_data)
    # test_X, test_Y = feature_label_split(test_data)

    # # 对X（包括train_X, test_X）进行归一化处理，方便后续操作
    # unif_trainX, X_max, X_min = uniform_norm(train_X)
    # unif_testX = (test_X - X_min) / (X_max - X_min)

    # 数据加载
    boston = load_boston()
    x = boston['data']
    y = boston['target']
    print(x.shape)
    print(y.shape)

    # 将y转换形状
    y = y.reshape(-1, 1)

    from sklearn.preprocessing import MinMaxScaler

    # 数据规范化
    ss_input = MinMaxScaler()
    x = ss_input.fit_transform(x)
    # Splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=5)

    # 模型训练
    model = linear_regression()
    model.fit(x_train, y_train, learning_rate=0.3, lamda=0.03)
    test_pred = model.predict(x_test)
    test_pred_error = sum((test_pred - y_test) ** 2) / (x_test.shape[0])
    print("Test error is %d" % (test_pred_error))

    plt.scatter(y_test, test_pred)
    plt.xlabel("prices")
    plt.ylabel("predicted prices")
    plt.show()




if __name__ == "__main__":
    main()