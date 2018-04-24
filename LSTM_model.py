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
import tensorflow as tf

def rnn_model(num_of_word,input_data,output_data=None,rnn_size=128,num_layers=2,batch_size=128):
    end_points = {}
    """

    :param num_of_word: 词的个数
    :param input_data: 输入向量
    :param output_data: 标签
    :param rnn_size: 隐藏层的向量尺寸
    :param num_layers: 隐藏层的层数
    :param batch_size: 
    :return: 
    """
    
    '''构建RNN核心'''
    # cell_fun = tf.contrib.rnn.BasicRNNCell
    # cell_fun = tf.contrib.rnn.GRUCell
    cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size,state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)

    # 如果有标签(output_data)则初始化一个batch的cell状态，否则初始化一个
    if output_data is not None:
        initial_state = cell.zero_state(batch_size,tf.float32)
    else:
        initial_state = cell.zero_state(1,tf.float32)

    # 词向量嵌入
    embedding = tf.get_variable('embedding',initializer=tf.random_uniform(
        [num_of_word + 1,rnn_size],-1.0,1.0))
    inputs = tf.nn.embedding_lookup(embedding,input_data)

    outputs,last_state = tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state)
    output = tf.reshape(outputs,[-1,rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size,num_of_word + 1]))
    bias = tf.Variable(tf.zeros(shape=[num_of_word + 1]))
    logits = tf.nn.bias_add(tf.matmul(output,weights),bias=bias)

    if output_data is not None:
        labels = tf.one_hot(tf.reshape(output_data,[-1]),depth=num_of_word + 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(0.01).minimize(total_loss)
        tf.summary.scalar('loss',total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction
    return end_points
