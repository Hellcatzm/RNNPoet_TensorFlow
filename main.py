# Author : hellcat
# Time   : 18-3-11

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
np.set_printoptions(threshold=np.inf)
"""

import os
import numpy as np
import tensorflow as tf

import poetry_porcess
from gen_poetry import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

batch_size = 64
poetry_file = './data/poems.txt'


def train(words,poetry_vector,x_batches,y_batches):
    input_data = tf.placeholder(tf.int32,[batch_size,None])
    output_targets = tf.placeholder(tf.int32,[batch_size,None])
    end_points = rnn_model(len(words),input_data=input_data,output_data=output_targets,batch_size=batch_size)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    merge = tf.summary.merge_all()
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('./logs',sess.graph)
        sess.run(init_op)

        start_epoch = 0
        model_dir = "./model"
        epochs = 50
        checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint:
            saver.restore(sess,checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
            print('## start training...')
        try:
            for epoch in range(start_epoch,epochs):
                n_chunk = len(poetry_vector) // batch_size
                for n in range(n_chunk):
                    loss,_,_ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op'],
                    ],feed_dict={input_data: x_batches[n],output_targets: y_batches[n]})
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch,n,loss))
                    if epoch % 5 == 0:
                        saver.save(sess,os.path.join(model_dir,"poetry"),global_step=epoch)
                        result = sess.run(merge,feed_dict={input_data: x_batches[n],output_targets: y_batches[n]})
                        writer.add_summary(result,epoch * n_chunk + n)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess,os.path.join(model_dir,"poetry"),global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))


if __name__ == "__main__":
    words,poetry_vector,to_num,x_batches,y_batches = poetry_porcess.poetry_process()
    # train(words, poetry_vector, x_batches, y_batches)
    # gen_poetry(words, to_num)
    generate(words_, to_num_, style_words="狂沙将军战燕然，大漠孤烟黄河骑。")
