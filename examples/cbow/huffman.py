# -*- coding: utf-8 -*-
import numpy as np


class HuffmanTreeNode:
    def __init__(self, value, possibility):
        self.possibility = possibility
        self.left = None
        self.right = None
        self.value = value  # the value of word
        self.code = '' # huffman code


class HuffmanTree:
    def __init__(self, word_dict, vec_len=15000):
        self.vec_len = vec_len  # the length of word vector
        self.root = None
        #
        word_dict_list = list(word_dict.values())
        node_list = [HuffmanTreeNode(item['word'], item['possibility']) for item in word_dict_list]
        self.build(node_list)
        self.generate_huffman_code(self.root, word_dict)

    def build(self, node_list):
        while len(node_list) > 1:
            i1 = 0
            i2 = 1
            if node_list[i2].possibility < node_list[i1].possibility:
                [i1, i2] = [i2, i1]
            for i in range(2, len(node_list)):
                if node_list[i].possibility < node_list[i2].possibility:
                    i2 = i
                    if node_list[i2].possibility < node_list[i1].possibility:
                        [i1, i2] = [i2, i1]
            top_node = self.merge(node_list[i1], node_list[i2])
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0, top_node)
        self.root = node_list[0]

    def generate_huffman_code(self, node, word_dict):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            # go along left tree
            while node.left or node.right:
                code = node.code
                node.left.code = code + "1"
                node.right.code = code + "0"
                stack.append(node.right)
                node = node.left
            word = node.value
            code = node.code
            word_dict[word]['code'] = code

    def merge(self, node1, node2):
        top_pos = node1.possibility + node2.possibility
        top_node = HuffmanTreeNode(np.zeros([1, self.vec_len]), top_pos)
        if node1.possibility >= node2.possibility:
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node