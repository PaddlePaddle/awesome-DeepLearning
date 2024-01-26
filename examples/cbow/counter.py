# -*- coding: utf-8 -*-

from collections import Counter
import jieba
from operator import itemgetter as _itemgetter


class WordCounter:
    def __init__(self, word_list):
        self.word_list = word_list
        self.stop_word = {}
        self.count_res = None
        self.count_word(self.word_list)

    def count_word(self, word_list, cut_all=False):
        filtered_word_list = []
        index = 0
        for line in word_list:
            res = list(jieba.cut(line, cut_all=cut_all))
            word_list[index] = res
            index += 1
            filtered_word_list += res

        self.count_res = MulCounter(filtered_word_list)
        for word in self.count_res:
            if word in self.stop_word:
                self.count_res.pop(word)


class MulCounter(Counter):
    # extends from collections.Counter
    # add some methods, larger_than and less_than
    def __init__(self, element_list):
        super(MulCounter, self).__init__(element_list)

    def larger_than(self, minvalue, ret='list'):
        temp = sorted(self.items(), key=_itemgetter(1), reverse=True)
        low = 0
        high = temp.__len__()
        while high - low > 1:
            mid = (low + high) >> 1
            if temp[mid][1] >= minvalue:
                low = mid
            else:
                high = mid
        if temp[low][1] < minvalue:
            if ret == 'dict':
                return {}
            else:
                return []
        if ret == 'dict':
            ret_data = {}
            for ele, count in temp[:high]:
                ret_data[ele] = count
            return ret_data
        else:
            return temp[:high]

    def less_than(self, maxvalue, ret='list'):
        temp = sorted(self.items(), key=_itemgetter(1))
        low = 0
        high = len(temp)
        while high - low > 1:
            mid = (low + high) >> 1
            if temp[mid][1] <= maxvalue:
                low = mid
            else:
                high = mid
        if temp[low][1] > maxvalue:
            if ret == 'dict':
                return {}
            else:
                return []
        if ret == 'dict':
            ret_data = {}
            for ele, count in temp[:high]:
                ret_data[ele] = count
            return ret_data
        else:
            return temp[:high]
