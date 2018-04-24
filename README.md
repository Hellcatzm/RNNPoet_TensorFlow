RNNPoet项目
===========

# 文件简介
`LSTM_model.py`：LSTM网络模型，提供了end_points接口，被其他部分调用<br>
`poetry_porcess.py`：数据读取、预处理部分，会返回打包好的batch，被main调用<br>
`gen_poetry.py`：古诗生成程序，拥有可选的风格参数，被main调用<br>
`main.py`：主函数，既可以调用前两个程序获取预处理数据并使用LSTM网络进行训练，也可以调用gen_poetry.py生成古诗<br>
