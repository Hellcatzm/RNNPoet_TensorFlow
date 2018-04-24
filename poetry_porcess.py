# Author : hellcat
# Time   : 18-3-12

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
np.set_printoptions(threshold=np.inf)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""
import numpy as np
from collections import Counter


def poetry_process(batch_size=64,
                   poetry_file='./data/poems.txt'):
    poetrys = []
    with open(poetry_file,'r',encoding='utf-8') as f:
        for line in f:
            try:
                title,content = line.strip().split(':')
                content = content.replace(' ','')  # 去空格，实际上没用到
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = 'B' + content + 'E'
                poetrys.append(content)
            except Exception as e:
                pass

    # 依照每首诗的长度排序
    # poetrys = sorted(poetrys, key=lambda poetry: len(poetry))
    print('唐诗数量：',len(poetrys))

    # 统计字出现次数
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = Counter(all_words)
    # print(counter.items())
    # item会把字典中的每一项变成一个2元素元组，字典变成大list
    count_pairs = sorted(counter.items(),key=lambda x: -x[1])
    # 利用zip提取，因为是原生数据结构，在切片上远不如numpy的结构灵活
    words,_ = zip(*count_pairs)
    # print(words)

    words = words[:len(words)] + (' ',)  # 后面要用' '来补齐诗句长度
    # print(words)
    # 字典:word->int
    word_num_map = dict(zip(words,range(len(words))))
    # 把诗词转换为向量
    to_num = lambda word: word_num_map.get(word,len(words))
    poetry_vector = [list(map(to_num,poetry)) for poetry in poetrys]

    n_chunk = len(poetry_vector) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batches = poetry_vector[start_index:end_index]
        length = max(map(len,batches))  # 记录下最长的诗句的长度
        xdata = np.full((batch_size,length),word_num_map[' '],np.int32)
        for row in range(batch_size):
            xdata[row,:len(batches[row])] = batches[row]
        # print(len(xdata[0])) 每个batch中数据长度不相等
        ydata = np.copy(xdata)
        ydata[:,:-1] = xdata[:,1:]
        """
            xdata             ydata
            [6,2,4,6,9]       [2,4,6,9,9]
            [1,4,2,8,5]       [4,2,8,5,5]
            """
        x_batches.append(xdata)  # (n_chunk, batch, length)
        y_batches.append(ydata)
    return words,poetry_vector,to_num,x_batches,y_batches