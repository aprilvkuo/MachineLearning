#!/usr/bin/env python
# encoding: utf-8

"""
三个问题：
1. 概率计算问题
2. 学习问题
3. 预测问题
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: hmm.py
@time: 2020/5/2 21:34
"""


class HMM(object):
    def __init__(self, A: list, B:list, pi:list)->None:
        """
        :param A: 状态转移概率矩阵, m * m
        :param B: 发射概率矩阵, m * n
        :param pi: 初始状态向量, m
        """
        self._A = A
        self._B = B
        self._pi = pi
        self._state_size = len(B)
        self._vocab_size = len(B[0])

    def __update_state_prob(self, final_states: list, state_index: int)->float:
        """
        状态转移更新
        :param final_states: 
        :param state_index: 
        :return: 
        """
        new_state_prob = 0
        for i in range(len(final_states)):
            new_state_prob += final_states[i] * self._A[i][state_index]
        return new_state_prob

    def prob_caculate(self, output: list, method="forward")->str:
        """
        概率计算问题
        :param output: 观测序列, vocabulary_index
        :param method:    类型， forward/ backward
        :return: 观测概率出现的概率
        """
        assert method in ["forward", "backward"]
        if method == "forward":
            final_prob = [1] * self._state_size
            for index in range(len(output)):
                output_index = output[index]
                # 状态转移
                if index == 0:
                    final_prob = self._pi
                else:
                    new_state_prob = [0] * self._state_size
                    for s_i in range(self._state_size):
                        new_state_prob[s_i] = self.__update_state_prob(final_prob, s_i)
                    final_prob = new_state_prob
                # 发射概率计算
                final_prob = [item * self._B[state_i][output_index]
                              for state_i, item in enumerate(final_prob)]
                # print(final_prob)
        elif method == "backward":
            final_prob = [1] * self._state_size
            # 状态转移
            for index in range(len(output)-2, -1, -1):

                new_prob = []
                for i in range(self._state_size):
                    probs = [self._A[i][state_i] * self._B[state_i][output[index+1]]* item
                            for state_i, item in enumerate(final_prob)]
                    new_prob.append(sum(probs))
                final_prob = new_prob
                # print(final_prob)
            final_prob = [item * self._pi[state_i] * self._B[state_i][output[0]]
                          for state_i, item in enumerate(final_prob)]
        return sum(final_prob)

    def __get_next_state(self, final_prob: list)->list:
        prob_list = []
        state_list = []
        for state_i in range(self._state_size):
            next_state_probs = [item * self._A[pre_state][state_i]
                                  for pre_state, item in enumerate(final_prob)]
            prob_list.append(max(next_state_probs))
            state_list.append(next_state_probs.index(prob_list[-1]))
        state_dict = dict(zip(range(self._state_size), state_list))
        return prob_list, state_dict

    def predite(self, output: list)->list:
        """
        维特比算法， 对hidden state进行预测
        :param output: 
        :return: 
        """
        final_prob = self._pi
        optimal_state_dict_list = []
        for index in range(len(output)):
            # 状态转移
            next_output = output[index]
            if index != 0:
                final_prob, state_dict = self.__get_next_state(final_prob)
                optimal_state_dict_list.append(state_dict)
            # 观测序列生成
            final_prob = [item * self._B[state_i][next_output]
                                  for state_i, item in enumerate(final_prob)]
        # 通过保存的信息进行回溯
        max_prob = max(final_prob)
        state_list = []
        last_state = final_prob.index(max_prob)
        for i in range(len(optimal_state_dict_list)-1, -1, -1):
            state = optimal_state_dict_list[i][last_state]
            state_list.append(last_state)
            last_state = state
        state_list.append(last_state)
        state_list = state_list[::-1]
        print("predite hidden state list is %s, prob is %f" % (str(state_list), max_prob))
        return state_list, max_prob

def test():
    a = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    b = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    pi = [0.2, 0.4, 0.4]
    model = HMM(a, b, pi)
    output_list = [0, 1, 0]
    print(model.prob_caculate(output_list))
    print(model.prob_caculate(output_list, method="backward"))
    print(model.predite(output_list))

# def test_back_ward_prb_caculate():
#     a = [[0.5, 0.2, 0.3],
#          [0.3, 0.5, 0.2],
#          [0.2, 0.3, 0.5]]
#     b = [[0.5, 0.5],
#          [0.4, 0.6],
#          [0.7, 0.3]]
#     pi = [0.2, 0.4, 0.4]
#     model = HMM(a, b, pi)
#     output_list = [0, 1, 0]
#     print(model.prob_caculate(output_list))

if __name__ == "__main__":
    test()
