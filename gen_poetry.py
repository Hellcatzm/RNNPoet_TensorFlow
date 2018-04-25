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
import tensorflow as tf
 
from LSTM_model import rnn_model
 
 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
 
 
def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]  # [np.argmax(predict)]
 
 
def gen_poetry(words, to_num):
    batch_size = 1
    print('模型保存目录为： {}'.format('./model'))
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    end_points = rnn_model(len(words), input_data=input_data, batch_size=batch_size)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=config) as sess:
        sess.run(init_op)
 
        checkpoint = tf.train.latest_checkpoint('./model')
        saver.restore(sess, checkpoint)
 
        x = np.array(to_num('B')).reshape(1, 1)
 
        _, last_state = sess.run([end_points['prediction'], end_points['last_state']], feed_dict={input_data: x})
 
        word = input('请输入起始字符:')
        poem_ = ''
        while word != 'E':
            poem_ += word
            x = np.array(to_num(word)).reshape(1, 1)
            predict, last_state = sess.run([end_points['prediction'], end_points['last_state']],
                                           feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, words)
        print(poem_)
        return poem_
 
 
def generate(words, to_num, style_words="狂沙将军战燕然，大漠孤烟黄河骑。"):
 
    batch_size = 1
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    end_points = rnn_model(len(words), input_data=input_data, batch_size=batch_size)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=config) as sess:
        sess.run(init_op)
 
        checkpoint = tf.train.latest_checkpoint('./model')
        saver.restore(sess, checkpoint)
 
        x = np.array(to_num('B')).reshape(1, 1)
        _, last_state = sess.run([end_points['prediction'], end_points['last_state']], feed_dict={input_data: x})
 
        if style_words:
            for word in style_words:
                x = np.array(to_num(word)).reshape(1, 1)
                last_state = sess.run(end_points['last_state'],
                                      feed_dict={input_data: x, end_points['initial_state']: last_state})
 
        # start_words = list("少小离家老大回")
        start_words = list(input("请输入起始语句："))
        start_word_len = len(start_words)
 
        result = start_words.copy()
        max_len = 200
        for i in range(max_len):
 
            if i < start_word_len:
                w = start_words[i]
                x = np.array(to_num(w)).reshape(1, 1)
                predict, last_state = sess.run([end_points['prediction'], end_points['last_state']],
                                               feed_dict={input_data: x, end_points['initial_state']: last_state})
            else:
                predict, last_state = sess.run([end_points['prediction'], end_points['last_state']],
                                               feed_dict={input_data: x, end_points['initial_state']: last_state})
                w = to_word(predict, words)
                # w = words[np.argmax(predict)]
                x = np.array(to_num(w)).reshape(1, 1)
                if w == 'E':
                    break
                result.append(w)
 
        print(''.join(result))
