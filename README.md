# hand-writting num recognizer

/*Test programme*/: This programme is my first deepLearning programme.

This projekt is designed to recognize hand-writting number image.

# 深度学习：构建神经网络并识别手写数字

~~正确率：0.63764（好像有点低~~

正确率：0.91175（发现有个cell有点错误，修正后达到91%了，好耶！

## 疑难杂症

### tenserflow把numpy卸载了…

没关系，会安装对应版本的numpy的

### Python关于%matplotlib inline报错

这个是 jupyter notebook 或者 jupyter qtconsole 的专属代码，使用```%matplotlib inline```这段代码可以输出图像。但是vscode编辑器并没有这个功能。（但这并不妨碍vscode是宇宙第一编辑器的事实）

### FileNotFoundError

```shell
PS F:\Programme_works\Python> python -u "f:\Programme_works\Python\DeepLearning\numRecog\neuralNetwork.py"
Traceback (most recent call last):
  File "f:\Programme_works\Python\DeepLearning\numRecog\neuralNetwork.py", line 106, in \<module\>
    with open('train.csv', 'r') as data_file:
FileNotFoundError: [Errno 2] No such file or directory: 'train.csv'
```

未找到文件，检查一下文件是否在命令执行时的目录（一般都是vscode打开的文件夹，或者jupyter notebook运行的文件所在的文件夹）

### 非jupyter编辑器输出csv图像

将

```python
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
```

改为

```python
matplotlib.pyplot.imsave('temp.png', image_array, cmap='Greys')
```

便会保存在目录下
