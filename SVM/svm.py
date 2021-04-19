import time
import numpy as np
import math
import random


def loadData(fileName):
    '''
    加载文件
    :param fileName: 要加载的文件路径
    :return: 数据集和标签集
    '''
    # 存放数据及标记
    dataArr = []
    labelArr = []
    # 读取文件
    fr = open(fileName)
    # 遍历文件中的每一行
    for line in fr.readlines():
        # 获取当前行，并按“,”切割成字段放入列表中
        # strip: 去掉每行字符串首尾指定的字符
        # split: 按照指定的字符将字符串切割成每个字段，返回列表形式
        curLine = line.strip().split(',')
        # 将每行中除标记外的数据放入数据集中(curLine[0]为标记信息)
        # 在放入的同时将原先字符串形式的数据转换为0-1的浮点型
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        # 将标记信息放入标记集中
        # 放入的同时将标记转换为整型
        # 数字0标记为1 其余标记为-1
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    # 返回数据集和标记
    return dataArr, labelArr


class SVM:
    '''
    SVM
    '''

    def __init__(self, trainDataList, trainLabelList, sigma=10, C=200, toler=0.001):
        '''
        SVM相关参数初始化
        :param trainDataList: 训练数据集
        :param trainLabelList: 训练标签集
        :param sigma: 高斯核中分母的σ
        :param C: 软间隔中的惩罚参数
        :param toler: 松弛变量
        '''
        self.trainDataMat = np.mat(trainDataList)  # 训练数据集
        self.trainLabelMat = np.mat(trainLabelList)  # 训练标签集

        self.m, self.n = np.shape(self.trainDataMat)  # m: 训练集数量   n: 样本特征数目
        self.sigma = sigma  # 高斯核分母中的σ
        self.C = C  # 惩罚参数
        self.toler = toler  # 松弛变量

        self.k = self.calcKernel()  # 核函数(初始化时提前计算)
        self.b = 0  # svm中的偏置b
        self.alpha = [0] * self.trainDataMat.shape[0]  # α长度为训练集数目
        self.E = [0 * self.trainLabelMat[i, 0] for i in range(self.trainLabelMat.shape[0])]  # SMO运算过程中的Ei
        self.supportVecIndex = []

    def calcKernel(self):
        '''
        计算核函数
        :return:高斯核矩阵
        '''
        # 初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m
        # k[i][j] = Xi * Xj
        k = [[0 for i in range(self.m)] for j in range(self.m)]

        # 大循环遍历Xi,Xi为式7.9中的x
        for i in range(self.m):
            # 每100个循环打印一次
            # 不能每次都打印，会极大拖慢程序运行速度
            # 因为print是比较慢的
            if i % 100 == 0:
                print('construct the kernek:', i, self.m)
            # 得到式7.90中的X
            X = self.trainDataMat[i, :]
            # 小循环遍历Xj，Xj为式7.90中的Z
            # 由于 Xi * Xj 等于 Xj * Xi，一次计算得到的结果可以
            # 同时放在k[i][j]和k[j][i]中，这样一个矩阵只需要计算一半即可
            # 所以小循环直接从i开始
            for j in range(i, self.m):
                # 获得Z
                Z = self.trainDataMat[j, :]
                # 先计算||X-Z||^2
                result = (X - Z) * (X - Z).T
                # 分子除以分母后去指数，得到的几位高斯核结果
                result = np.exp(-1 * result / (2 * self.sigma ** 2))
                # 将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k[i][j] = result
                k[j][i] = result
        # 返回高斯核矩阵
        return k

    def isSatisfyKKT(self, i):
        '''
        查看第i个α是否满足KKT条件
        :param i: α的下标
        :return:
            True: 满足
            False: 不满足
        '''
        gxi = self.calc_gxi(i)
        yi = self.trainDataMat[i]
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) and (
                math.fabs(yi * gxi - 1) < self.toler):
            return True
        return False

    def calc_gxi(self, i):
        '''
        计算g(xi)
        :param i: x的下标
        :return: g(xi)的值
        '''
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # 遍历每一个非零α，i为非零α的下标
        for j in index:
            gxi += self.alpha[j] * self.trainLabelMat[j] * self.k[j][i]

        gxi += self.b
        # 返回
        return gxi

    def calcEi(self, i):
        '''
        计算Ei
        :param i: E的下标
        :return:
        '''
        gxi = self.calc_gxi(i)
        return gxi - self.trainLabelMat[i]

    def getAlphaJ(self, E1, i):
        '''
        SMO中选择第二个变量
        :param E1: 第一个变量的E1
        :param i: 第一个变量α的下标
        :return: E2，α2的下标
        '''
        # 初始化E2
        E2 = 0
        # 初始化|E1-E2|为-1
        maxE1_E2 = -1
        # 初始化第二个变量的下标
        maxIndex = -1
        # 获得Ei非零的对应索引组成的列表，列表内容为非0Ei的下标i
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        # 对每个非零Ei的下标i进行遍历
        for j in nozeroE:
            # 计算E2
            E2_tmp = self.calcEi(j)
            # 如果|E1-E2|大于目前最大值
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                # 更新最大值
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                # 更新最大值E2的索引j
                E2 = E2_tmp
                # 更新最大值E2的索引j
                maxIndex = j
        # 如果列表中没有非零元素了
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                # 获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = int(random.uniform(0, self.m))
            # 获得E2
            E2 = self.calcEi(maxIndex)
        # 返回第二个变量的E2值以及其索引
        return E2, maxIndex

    def train(self, iter=100):
        # iterStep: 迭代次数，超过设置次数还未收敛则强制停止
        # parameterChanged: 单次迭代中有参数改变则增加1
        iterStep = 0
        parameterChanged = 1

        # 如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
        # parameterChanged==0时代表上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明
        # 达到了收敛状态，可以停止了
        while (iterStep < iter) and (parameterChanged > 0):
            # 打印当前迭代论数
            print('iter:%d:%d' % (iterStep, iter))
            # 迭代步数加1
            iterStep += 1
            # 新的一轮将参数改变标志位新置0
            parameterChanged = 0

            # 大循环遍历所有样本，用于找SMO中第一个变量
            for i in range(self.m):
                # 查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量
                if self.isSatisfyKKT(i) == False:
                    # 如果下标i的α不满足KKT条件，则进行优化
                    # 第一个变量α的下标i已经确定
                    # 选择变量2.由于变量2的选择中涉及到|E1 - E2|,因此先计算E1
                    E1 = self.calcEi(i)

                    # 选择第2个变量
                    E2, j = self.getAlphaJ(E1, i)

                    # 获得两个变量的标签
                    y1 = self.trainLabelMat[i]
                    y2 = self.trainLabelMat[j]
                    # 复制α值作为old值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    # 依据标签是否一致来生成不同的L和H
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    # 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H: continue

                    # 计算α的新值
                    # 更新α2值
                    # 先获得几个k值，用来计算分母η
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]

                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
                    # 剪切α2
                    if alphaNew_2 < L:
                        alphaNew_2 = L
                    elif alphaNew_2 > H:
                        alphaNew_2 = H
                    # 更新α1
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) - y2 * k21 * (
                                alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) - y2 * k22 * (
                                alphaNew_2 - alphaOld_2) + self.b

                    # 依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    # 将更新后的各类值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)

                    # 如果α的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    # 反之则自增1
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1
                # 打印迭代轮数，i值，该迭代轮数修改α数目
                print("iter: %d i: %d, pairs changed %d" % (iterStep, i, parameterChanged))
        # 全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(self.m):
            # 如果α>0, 说明是支持向量
            if self.alpha[i] > 0:
                # 将支持向量的索引保存起来
                self.supportVecIndex.append(i)

    def calcSinglKernel(self, x1, x2):
        '''
        单独计算核函数
        :param x1: 向量1
        :param x2: 向量2
        :return: 核函数结果
        '''
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        # 返回结果
        return np.exp(result)

    def predict(self, x):
        '''
        对样本的标签进行预测
        :param x: 要预测的样本x
        :return: 预测结果
        '''
        result = 0
        for i in self.supportVecIndex:
            # 遍历所有支持向量，计算求和式
            # 如果是非支持向量，求和子式必为0， 没有必须进行计算
            # 这也是为什么在SVM最后只有支持向量起作用
            # ----------------------
            # 先单独将核函数计算出来
            tmp = self.calcSinglKernel(self.trainDataMat[i, :], np.mat(x))
            # 对每一项子式进行求和，最终计算得到求和项的值
            result += self.alpha[i] * self.trainLabelMat[i] * tmp
        # 求和项计算结束后加上偏执b
        result += self.b
        # 使用sign函数返回预测结果
        return np.sign(result)

    def test(self, testDataList, testLabelList):
        '''
        测试
        :param testDataList: 测试数据集
        :param testLabelList: 测试标签集
        :return: 正确率
        '''
        # 错误计数值
        errorCnt = 0
        # 遍历测试集所有样本
        for i in range(len(testDataList)):
            # 打印目前进度
            print('test:%d:%d' % (i, len(testDataList)))
            # 获取预测结果
            result = self.predict(testDataList[i])
            # 如果预测与标签不一致，错误计数值加一
            if result != testLabelList[i]:
                errorCnt += 1
        # 返回正确率
        return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    start = time.time()
    # 获取训练集及标签
    print('start read transSet')
    trainDataList, trainLabelList = loadData('data/mnist_train.csv')

    # 获取测试集及标签
    print('start read testSet')
    testDataList, testLabelList = loadData('data/mnist_test.csv')

    # 初始化SVM
    print('Start init SVM')
    svm = SVM(trainDataList[:1000], trainLabelList[:1000], 10, 200, 0.001)

    # 开始训练
    print('Start to train')
    svm.train()

    # 开始测试
    print('Start to test')
    accuracy = svm.test(testDataList[:100], testLabelList[:100])
    print('the accuracy is:%d' % (accuracy * 100), '%')

    # 打印时间
    print('time span:', time.time() - start)

dataSets, labels = loadData("data/mnist_train.csv")
SVM(dataSets, labels)
