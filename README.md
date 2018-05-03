RNNPoet项目
===========
## 相关文章
###### 项目介绍
[『TensotFlow』RNN/LSTM古诗生成](http://www.cnblogs.com/hellcat/p/7710034.html)
###### 文字预处理脚本介绍
[『TensotFlow』RNN中文文本_上](http://www.cnblogs.com/hellcat/p/7410027.html)
###### 梯度处理函数介绍
[『TensorFlow』梯度优化相关](http://www.cnblogs.com/hellcat/p/7435977.html)

## 1、文件简介
`LSTM_model.py`：LSTM网络模型，提供了end_points接口，被其他部分调用<br>
`poetry_porcess.py`：数据读取、预处理部分，会返回打包好的batch，被main调用<br>
`gen_poetry.py`：古诗生成程序，拥有可选的风格参数，被main调用<br>
`main.py`：主函数，既可以调用前两个程序获取预处理数据并使用LSTM网络进行训练，也可以调用gen_poetry.py生成古诗<br>

## 2、调用指令
使用编辑工具进入`main.py`，可以看到程序末尾有如下指令，
```Python
if __name__ == "__main__":
    words,poetry_vector,to_num,x_batches,y_batches = poetry_porcess.poetry_process()
    # train(words, poetry_vector, x_batches, y_batches)
    # gen_poetry(words, to_num)
    generate(words_, to_num_, style_words="狂沙将军战燕然，大漠孤烟黄河骑。")
```
此时实际上处于生成模式，对于最后的三行，
train：表示训练
gen_poetry：表示根据首字符生成
generate：表示根据首句和风格句生成古诗

###### 训练模型
注释掉后两行，保留train行，即修改如下,
```Python
if __name__ == "__main__":
    words,poetry_vector,to_num,x_batches,y_batches = poetry_porcess.poetry_process()
    train(words, poetry_vector, x_batches, y_batches)
    # gen_poetry(words, to_num)
    # generate(words_, to_num_, style_words="狂沙将军战燕然，大漠孤烟黄河骑。")
```
然后运行脚本，
```Shell
python main.py
```
即开始训练。

###### 生成古诗
使用最上面的原版就可以，即如下所示
```Python
if __name__ == "__main__":
    words,poetry_vector,to_num,x_batches,y_batches = poetry_porcess.poetry_process()
    # train(words, poetry_vector, x_batches, y_batches)
    # gen_poetry(words, to_num)
    generate(words_, to_num_, style_words="狂沙将军战燕然，大漠孤烟黄河骑。")
```
运行脚本，
```Shell
python main.py
```
即可显示结果。

如果希望更换风格，同样在这几行代码中（就是最后一行），
```Python
generate(words_, to_num_, style_words="狂沙将军战燕然，大漠孤烟黄河骑。")
```
可以替换style_word为任何你想要的风格句，注意最好使用7言或者5言，因为这句会大概率影响到你生成的古诗的句子长度(不绝对)，这只是风格提取，你可以输入任意长度；在运行了脚本后，屏幕会提示输入起始句，输入的句子一般5或者7个字，这个由于会拿来直接做首句(由结果示范可以看到)，输入长度不宜过长。

## 3、结果示范
```Shell
head:床前明月光 + style:黄沙百战金甲：

床前明月光辉，魏武征夫血絮红。
数步崩云复遗主，缟衣东，帝京举，玉轮还满出书初。
秋秋惨惨垂杨柳，梦断黄莺欲断肠。
花凋柳映阮家几，屋前病，歇马空留门。
当年皆月林，独往深山有素。
 
 
head:少小离家老大回 + style:山雨欲来风满楼：

少小离家老大回，四壁百月弄鸦飞。
扫香花间春风地，隔天倾似烂桃香。
近来谁伴清明日，两株愁味在罗帏。
仍通西疾空何处，轧轧凉吹日方明。
 
 
head:少小离家老大回 + style:铁马冰河入梦来：

少小离家老大回，化空千里便成丝。
官抛十里同牛颔，莫碍风光雪片云。
饮水远涛飞汉地，云连城户翠微低。
一树铁门万象耸，白云三尺各关高。
同言东甸西游子，谁道承阳要旧忧。
 
少小离家老大回，含颦玉烛拂楼台。
初齐去府芙蓉死，细缓行云向国天。
```
