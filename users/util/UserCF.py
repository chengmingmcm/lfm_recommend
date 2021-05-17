#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/3/30 9:00
# @Author : MCM
# @Site :
# @File : UserBasedCF.py
# @Software: PyCharm

from operator import itemgetter
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt

# # data_file = 'ratings.csv'
# data_file = 'text3.csv'
# # file_path = 'ratings.csv'
data_file = r'D:\Demos\python\Recommend\django_auth_example\users\static\users_resulttable.csv'
# model_path = 'lfm.model'
model_path = r'D:\Demos\python\Recommend\django_auth_example\users\util\lfm_train.model'


class UserCF:
    def __init__(self):
        self.frame = pd.read_csv(data_file, usecols=range(3))
        self.frame.columns = ['user', 'item', 'rating']
        self.data = {}
        self.train = {}
        self.test = {}
        self.similarity = {}

    @staticmethod
    def _process_data(input_data):
        """
        自定义数据处理函数
        :param input_data: DataFrame
        :return: dict{user_id: {item_id: rating}}
        """
        output_data = {}
        for _, items in input_data.iterrows():
            user = int(items['user'])
            item = int(items['item'])
            rating = float(items['rating'])
            if user in output_data.keys():
                currentRatings = output_data[user]
            else:
                currentRatings = {}
            currentRatings[item] = rating
            output_data[user] = currentRatings
        return output_data

    def load_data(self, train_size, normalize=True):
        """
        划分训练集、测试集，并定义数据结构为：dict{userid: {movieId: rating}}
        :param train_size:
        :param normalize:
        :return:
        """
        print('loading data...')
        if normalize is True:  # 利用pandas对整列进行归一化，评分在(0,1)之间
            rating = self.frame['rating']
            self.frame['rating'] = (rating - rating.min()) / (rating.max() - rating.min())

        train_data = self.frame.sample(frac=train_size, random_state=10, axis=0)
        test_data = self.frame[~self.frame.index.isin(train_data.index)]

        self.data = self._process_data(self.frame)
        self.train = self._process_data(train_data)
        self.test = self._process_data(test_data)

        print('loaded data finish...')

    def users_similarity(self, normal=False):
        """
        计算用户和用户之间的矩阵相似度
        :return:
        """
        users_matrix = {}
        for user, items in self.train.items():
            for item in items.keys():
                users_matrix.setdefault(item, set())
                users_matrix[item].add(user)
        user_item_count = {}
        user_count = {}
        for item, users in users_matrix.items():
            for i in users:
                user_item_count.setdefault(i, 0)
                user_item_count[i] += 1
                for j in users:
                    if i == j:
                        continue
                    user_count.setdefault(i, {})
                    user_count[i].setdefault(j, 0)
                    # user_count[i][j] += 1
                    user_count[i][j] += 1 / np.math.log(1 + len(users) * 1.0)

        if not normal:
            for i, related_users in user_count.items():
                self.similarity.setdefault(i, {})
                for j, rating in related_users.items():
                    # print(related_users)
                    self.similarity[i][j] = rating / np.sqrt(
                        user_item_count[i] * user_item_count[j] * 1.0)  # 余弦相似度
        else:
            self.similarity_max = {}
            for i, related_users in user_count.items():
                self.similarity.setdefault(i, {})
                for j, rating in related_users.items():
                    self.similarity_max.setdefault(j, 0)
                    self.similarity[i][j] = rating / np.sqrt(
                        user_item_count[i] * user_item_count[j] * 1.0)
                    if self.similarity[i][j] > self.similarity_max[j]:
                        self.similarity_max[j] = self.similarity[i][j]
            for i, related_users in self.similarity.items():
                for j, rating in related_users.items():
                    self.similarity[i][j] = self.similarity[i][j] / self.similarity_max[j]

    def predict(self, user, K, N=None):
        """
        对输入的一个user进行推荐
        :param K:
        :param N:
        :param user:
        :return:
        """
        recommendations = {}

        if user not in list(self.train.keys()) or user not in list(self.similarity.keys()):  # 训练集中不存在用户返回空结果
            return list()

        interacted_items = self.train[user]
        for sim_user, similarity_factor1 in sorted(self.similarity[user].items(), key=itemgetter(1), reverse=True)[:K]:
            for related_item, similarity_factor2 in self.train[sim_user].items():
                if related_item in interacted_items:
                    continue
                recommendations.setdefault(related_item, 0)
                recommendations[related_item] += similarity_factor1 * similarity_factor2

        result_list = [(item, rating) for item, rating in recommendations.items()]
        result_list.sort(key=lambda x: x[1], reverse=True)
        if N:
            N = int(N)
            return result_list[:N]
        else:
            return result_list

    def validate(self, K=20):
        """
        计算MAE、RMSE评估指标
        :return:
        """
        print('calculating MAE and RMSE...')
        error_sum = 0.0
        sqrError_sum = 0.0
        setSum = 0
        # i = 0
        for user in self.test:
            # if i % 50 == 0:
            #     print('calculating %d users' % i)
            recommendation = self.predict(user, K)  # ->list
            userRatings = self.test[user]  # ->dict
            for each in recommendation:
                item = each[0]
                rating = each[1]
                if item in userRatings:
                    error_sum += abs(userRatings[item] - rating)
                    sqrError_sum += (userRatings[item] - rating) ** 2
                    setSum += 1

        mae = error_sum / setSum
        rmse = np.sqrt(sqrError_sum / setSum)
        return mae, rmse

    def evaluate(self, K=20, N=10):
        """
        推荐topN结果评估，计算precision和recall
        :return:
        """
        print('calculating precision and recall...')
        hit = 0
        recall_sum = 0
        precision_sum = 0
        # i = 0
        for user in self.test.keys():
            # if i % 50 == 0:
            #     print('calculating %d users' % i)
            real_items = self.test.get(user)
            recommendation = self.predict(user, K, N)
            rec_result = [(item, rating) for item, rating in recommendation]
            pred_items = [p[0] for p in rec_result]
            hit += len([h for h in pred_items if h in real_items])
            recall_sum += len(real_items)
            # precision_sum += len(pred_items)
            precision_sum += N

        # print(precision_sum, recall_sum)
        precision = hit / (precision_sum * 1.0)
        recall = hit / (recall_sum * 1.0)
        return precision, recall


if __name__ == '__main__':
    usercf = UserCF()
    # train_range = [0.9, 0.8, 0.7, 0.6, 0.5]
    mae_rmse = []
    pre_rec = []
    for i in list(range(10, 31, 1)):
        usercf.load_data(train_size=0.8, normalize=True)
        usercf.users_similarity(normal=False)

        print('K：', i)

        mae, rmse = usercf.validate(K=i)
        print(mae, rmse)
        mae_rmse.append((mae, rmse))

        pre, rec = usercf.evaluate(K=i)
        print(pre, rec)
        pre_rec.append((pre, rec))

        # res = usercf.predict(user=1, top_n=10)
        # print(res)

    # plt.plot(train_range, [i[0] for i in mae_rmse])
    # plt.xlabel('train data')
    # plt.ylabel('MAE')
    # plt.show()
    #
    # plt.plot(train_range, [i[1] for i in mae_rmse])
    # plt.xlabel('train data')
    # plt.ylabel('RMSE')
    # plt.show()
    #
    # plt.plot(train_range, [i[0] for i in pre_rec])
    # plt.xlabel('train data')
    # plt.ylabel('precision')
    # plt.show()
    #
    # plt.plot(train_range, [i[1] for i in pre_rec])
    # plt.xlabel('train data')
    # plt.ylabel('recall')
    # plt.show()
