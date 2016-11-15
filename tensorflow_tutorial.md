## Tensorflow c++ 使用指南
### 目录
- [背景介绍](#背景介绍)
- [创建模型](#创建模型)
- [生成动态链接库](#生成动态链接库)
- [使用动态链接库](#使用动态链接库)
- [注意事项](#注意事项)
- [总结](#总结)
- [参考](#参考)

### 背景介绍
[Tensorflow](https://www.tensorflow.org)是谷歌开源的一款深度学习框架，支持分布式训练和多GPU训练。虽然其开源时间比Theano和Torch等深度学习框架更晚，但是因为有Google的强大支持和其在开源社区强大的影响力，Tensorflow迅速流行，目前已有大量使用Tensorflow实现的论文代码和经典模型。从个人的使用经验来说，使用Torch进行快速原型实验以及调试相对来说方法一些，因此相对来说更适合做纯研究的人使用。而Tensorflow在研究和应用两个方面相对来说平衡一些，不仅因为Tensorflow支持分布式的训练和多GPU，而且还提供了生产级别的高性能服务框架[Tensorflow Serving](https://tensorflow.github.io/serving/)，方便模型快速部署到生产环境。

如果你还不熟悉Tensorflow的基本使用，可以先学习下Tensorflow官网的[教程](https://www.tensorflow.org/versions/r0.11/tutorials/index.html)或者参考Github上的一些[教程](https://github.com/AlbertXiebnu/TF-Tutorials)。这篇文章主要是想介绍如何
把训练好的模型应用在生产环境中，我将会以一个实际的例子来说明整个过程以及中间的一些坑，希望可以给大家提供一个简单的参考。

> 为了方便大家学习和参考，文章中所有的代码都放在我的github项目中 [tf_han](https://github.com/AlbertXiebnu/tf_han)

在实际的使用过程中，最常见的一个生产流程是用Tensorflow的python脚本来进行算法实验和模型的训练，当模型的效果调整到最优后，我们希望可以把调优后的模型部署到实际的生产环境中。完成这个过程，一般来说有两种方式，第一种可以采用上文提到了Tensorflow Serving的方式，把模型的预测以服务的方式提供，客户端提供rpc调用的方式来获取预测结果。另一种方式是以动态链接库的方式提供，把模型的预测封装成.so文件，客户端可以直接把预测进行代码层面的集成。其中第一种方式提供了对模型生命周期的管理，模型切换以及机群和GPU的支持等，可以做到方便地从模型生成（训练）到模型部署（提供服务）这一流程。我觉得这种部署方式效率非常高，可以优先考虑。不过，因为在工作中，需要和现有的代码做集成，我主要尝试了第二种方法，下面将着重介绍。


### 创建模型
事实上，Tensorflow提供了两种打包动态链接库的方式。一种方式是只打包Tensorflow核心代码，生成tensorflow.so文件，然后客户端代码通过调用Tensorflow提供的[c api](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h)来使用。另一种方式就是直接调用Tensorflow的c++代码，然后和模型预测代码一起封装并打包成动态链接库，提供给外部接口来调用。第一种方法计划在下一篇文章中解释，这里主要讲下第二种方法。大体分为三个步骤：
1. 创建模型。用Tensorflow Python接口创建Graph，并训练和保存Graph和模型参数。
2. 生成动态链接库。利用Tensorflow c++ api来完成模型的加载，并对外提供预测接口。
3. 使用动态链接库。直接引用接口文件，在链接时链接第2步生成的动态链接库。

首先，第一步我们先定义模型。假设我们的应用场景是做文本分类，首先我们先定义一个层次LSTM的文本分类模型。模型的输入是文章的词序列，输出是文章的类别，即label。我们先看下模型的输入是如何定义的。
```python
x = tf.placeholder("int32",[None,FLAGS.max_sen,FLAGS.max_len],name="input")
y = tf.placeholder("float32",[None,n_classes],name="target")
keep_prob = tf.placeholder(tf.float32,name="keep_prob")
```
上面的代码中 *x* 表示模型的输入，为文章的词序列，每个词用词典的下标表示。*y* 表示文章的label，取值为0或1。*keep_prob* 为dropout层的参数。注意到我们为模型的每个输入结点进行了命名，这个命名在用 *c++ api* 加载模型是将会用到。模型的具体定义可以参考 [*model.py*](https://github.com/AlbertXiebnu/tf_han/blob/master/model.py)，太长了就不贴出来了。注意到，和模型输入一样，模型最后的输出结点同样进行了命名，命名为*output*。
```python
W = tf.get_variable("fullconnect_weights",shape=[2 * self.docsize,n_classes])
B = tf.get_variable("fullconnect_bias",shape=[n_classes])
output = tf.add(tf.matmul(att,W),B,name="output")
return output
```
因为这里举的例子是我实际项目代码，所以模型可能会比较复杂，如果不好理解可以简单地把模型想象成一个线性模型，输入结点是*x*和*y*，输出是*output*。接下来我们需要用训练数据对这个模型进行训练，同时把模型的定义和模型的参数保存下来。

```python
with tf.Session(config=config) as sess:
        sess.run(init)
        for epoch in range(FLAGS.max_epochs):
            step = 1
            training_iters = data.trainset.num_samples
            print "Epoch %d: " %(epoch+1)
            while step * batch_size < training_iters:
                batch_x , batch_y = data.trainset.next_batch(batch_size)
                sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:FLAGS.keep_prob})
                if step % display_step == 0:
                    loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,\
                                            y:batch_y,keep_prob:FLAGS.keep_prob})
                    print "Iter " + str(step * batch_size) + ",Minibatch Loss = " + \
                            "{:.6f}".format(loss) + ", Training Accuracy = " + \
                            "{:.5f}".format(acc)
                step += 1
            # valid for 1000 examples
            acc,loss = sess.run([accuracy,cost],feed_dict={x:data.validset.data[:1000],\
                                    y:data.validset.label[:1000],keep_prob:1.0})
            print "Top 1000 validset, loss ="+ "{:.6f}".format(loss) + \
                                ", accuracy = " + "{:.5f}".format(acc)

            file_name = os.path.join(FLAGS.save,"model_"+str(epoch)+".ckpt")
            save_path = saver.save(sess,file_name)
            tf.train.write_graph(sess.graph.as_graph_def(),FLAGS.save,"graph.pb")
            print "Model saved in file: %s" %(save_path)

            if best_valid_acc < acc:
                best_valid_loss = loss
                best_valid_acc = acc
                stop_count = 0
                print "Testing Accuracy: ", sess.run(accuracy,\
                feed_dict={x:data.testset.data,y:data.testset.label,keep_prob:1.0})
            else:
                stop_count += 1

            if stop_count >= FLAGS.early_stop_count:
                print "early stop training at epoch %d" %epoch
                break
```
需要注意的是，Tensorflow在保存时，模型的定义和模型参数是分开保存的。其中模型的定义，也就是Tensorflow里的计算图(Computation Graph)通过函数*tf.train.write_graph*来保存。模型的具体参数，也就是定义为*tf.Variable*的结点来保存参数，可以使用*tf.train.Saver*来保存。在上面的例子中，假设我们把模型的定义和参数分别保存成*graph.pb*和*model.ckpt*两个文件。

### 生成动态链接库
第一步生成的*model.ckpt*和*graph.pb*并不是我们最终要load的模型，需要先把这两个文件合并成一个文件，这个过程称为*Freezing*。那么什么是*Freezing*呢？官网教程对这个[Freezing](https://www.tensorflow.org/versions/r0.10/how_tos/tool_developers/index.html#freezing)操作解释如下：
> What this does is load the GraphDef, pull in the values for all the variables from the latest checkpoint file, and then replace each Variable op with a Const that has the numerical data for the weights stored in its attributes It then strips away all the extraneous nodes that aren't used for forward inference, and saves out the resulting GraphDef into an output file

翻译一下，就是把*tf.Variable*全部用相同参数的*tf.Constant*来替换，并去除*forward*过程中没有用到的Operation。最后把模型的参数和定义合并为一个文件。具体操作如下：
1. git clone https://github.com/tensorflow/tensorflow.git
2. cd tensorflow. 切换到tensorflow目录。
3. 执行如下命令：

```
bazel build tensorflow/python/tools:freeze_graph && \
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=graph.pb \
--input_checkpoint=model.ckpt \
--output_graph=/tmp/frozen_graph.pb --output_node_names=output
```
通过上面的步骤，我们最终得到模型文件*froen_graph.pb*。接下来，我们编写两个文件[*lstm.h*](https://github.com/AlbertXiebnu/tf_han/blob/master/loadgraph/lstm.h)和[*lstm.cc*](https://github.com/AlbertXiebnu/tf_han/blob/master/loadgraph/lstm.cc)两个文件来实现对模型的加载和预测。我们主要来看下*lstm.cc*这个文件。

首先看下*lstm.cc*中*Init*函数的内容:
```cpp
int32_t LSTM::Init()
{
    int32_t res = LoadVocab();
    if(res) return res;
    SessionOptions sess_options = SessionOptions();
    Status status = NewSession(sess_options,&m_sess);
    if(!status.ok()){
        cout << status.ToString() << endl;
        return 1;
    }
    status = m_sess->Create(m_graph_def);
    if(!status.ok()){
        cout << status.ToString() << endl;
        return 1;
    }
    //省略
}
```

上面主要有三个步骤，首先是创建*Session*，然后是读取*graph*，也就是上面通过*model.ckpt*和*graph.pb*合并得到的*frozen_graph.pb*。最后一步是在当前的*Session*下创建*Graph*。接下来，我们看下*lstm.cc*中的预测函数*Predict*内容：

```cpp
int32_t LSTM::Predict(const vector<string>& content,int& label,float& prob)
{
    int32_t fill_len = FillFeatureVec(content);
    auto dst = x.flat<int>().data();
    if(fill_len >= max_length){
    	copy_n(feature_vec,max_length,dst);
    }else{
    	copy_n(feature_vec,fill_len,dst);
    	dst += fill_len;
    	copy_n(padding_vec,max_length - fill_len,dst);
    }

    Tensor keep_prob(DT_FLOAT,TensorShape());
    keep_prob.scalar<float>()()=1.0;
    vector<pair<string,Tensor> > inputs = {{"input",x},{"keep_prob",keep_prob}};
    vector<Tensor> outputs;
    Status status = m_sess->Run(inputs,{"output"},{},&outputs);
    if(!status.ok()){
        cout << status.ToString() << endl;
        return 1;
    }

    // check output and get predict label and probability
    auto item = outputs[0].shaped<float,2>({batch_size,num_classes});
    float normal_weight = item(0,0);
    float ad_weight = item(0,1);
    label = normal_weight < ad_weight? 1:0;
    prob = exp(ad_weight)/(exp(ad_weight)+exp(normal_weight));
    return 0;
}
```
*Predict* 函数主要做了三件事情。第一，把输出的文本通过查字典的方式转成整数向量，并用 *Tensor* 来封装，作为模型的输入。第二，执行*Session*的*run*方法，传入输入*Tensor*，并用*output*来接收模型的输出。第三，根据模型的输出，来判断预测的类别，并使用*softmax*计算对应类别的概率。

因为*lstm.h*直接依赖了Tensorflow的头文件，如何直接把*lstm.h*作为外部引用的头文件则会间接依赖Tensorflow的头文件，破坏了库的封装性。因此我们在此基础上再封装一层，去除对Tensorflow头文件的依赖。新建文件[*lstm_api.h*](https://github.com/AlbertXiebnu/tf_han/blob/master/loadgraph/lstm_api.h)和[*lstm_api.cc*](https://github.com/AlbertXiebnu/tf_han/blob/master/loadgraph/lstm_api.cc)。

```cpp
#include <string>
#include <vector>
extern C {
int32_t InitLSTM(const std::string vocab_file,const std::string graph_file);
int32_t UnitLSTM();
int32_t LSTMPredict(const std::vector<std::string>& content,int& label,float& prob);
}
```
*lstm_api.cc*

```cpp
#include "lstm_api.h"
#include "lstm.h"
static LSTM* glstm;

extern C int32_t InitLSTM(const std::string vocab_file,const std::string graph_file)
{
    glstm = new LSTM(vocab_file,graph_file);
    if(glstm == NULL){
         printf("new LSTM failed\n");
	 return 1;
    }
    return glstm->Init();
}

extern C int32_t UnitLSTM()
{
    if(glstm==NULL){
	printf("lstm instance is NULL\n");
	return 1;
    }
    glstm->UnInit();
    delete glstm;
    glstm = NULL;
    return 0;
}

extern C int32_t LSTMPredict(const std::vector<std::string>& content,int& label,float& prob)
{
    return glstm->Predict(content,label,prob);
}
```
最后，我们新建一个项目*loadgraph*来编译和链接所有的文件，并生成动态链接库。步骤如下：
1. 新建目录 __*tensorflow/tensorflow/loadgraph*__ ，把四个文件*lstm.h*、*lstm.cc*、*lstm_api.h*、*lstm_api.cc* 拷贝到该目录下。
2. 新建*BUILD*文件，添加编译规则。
3. 执行```bazel build -c opt --copt=mavx :lstm.so```

在*BUILD* 内添加如下编译规则：
```build
cc_binary(
    name = "lstm.so",
    srcs = ["lstm_api.cc","lstm_api.h","lstm.cc","lstm.h"],
    linkshared = 1,
    deps = [
        "//tensorflow/core:tensorflow",
    ],
)
```
编译完成后，我们将在目录 __*bazel-bin/tensorflow/loadgraph/*__ 下得到编译生成的*lstm.so*文件。

### 使用动态链接库
经过打包生成的*lstm.so*将不依赖于Tensorflow的运行环境。任何代码如果想要使用我们训练的模型，只需要在源码中引用头文件*lstm_api.h*，并在编译时链接*lstm.so*即可。通过这种方式生成的可执行文件，可以在任何机器上运行，而无需事先按照Tensorflow，非常方便和灵活。不过因为Tensorflow底层是用c++实现，而且采用了c++ 11的语法，因此要求gcc的版本必须支持c++ 11。为了保险起见，gcc的版本最好是4.8以后的版本。

下面用一个非常简单的例子来说明如何来使用。以公司的blade编译工具为例，首先创建项目根目录 __*lstm_test*__ ，并添加BLADE_ROOT文件，表示当前目录为Blade项目根目录。在根目录下，新建 __*lstm_test/lib64_release*__ ，把*lstm.so*重命名为*liblstm.so*，并放在 __*lib64_release*__ 目录下。

![](https://cl.ly/020E3o0f101J/pic1.png)

新建*lstm_test.cc*测试文件，添加如下内容：
```cpp
#include <iostream>
#include <string>
#include <vector>
#include "lstm_api.h"
using namespace std;
int main(){
    string vocab_file = "./data/vocab.txt";
    string graph_file = "./data/frozen_graph.pb";
    InitLSTM(vocab_file,graph_file);
    // ...
    LSTMPredict(content,label,prob);
    // ...
    UnitLSTM();
    return 0;
}
```
最后，为项目添加*BUILD*文件
```
cc_library(
    name = "lstm",
    prebuilt = True,
    link_all_symbols = 1,
)
cc_binary(
    name = "lstm_test",
    srcs = "lstm_test.cc",
    deps = [
        ":lstm",
    ],
)
```

### 注意事项
在实际的预测时，你可能会发现模型预测的非常慢，甚至比你用python的api预测还要慢。原因可能有很多方面，一方面Tensorflow本身在快速的迭代过程中，因此，不同版本的性能差异可能会比较大，另一方面也可能是由于编译选项没有设置对，甚至是你自己代码部分的效率问题。这些都是我在实际使用过程中踩过的一些坑，希望可以给大家一些借鉴和参考。

首先查看*Eigen*矩阵运算库的版本。Tensorflow的矩阵运算都是采用开源矩阵运算库*Eigen*来实现的。在编译的过程中，尽可能的采用新版的*Eigen*。新版本的*Eigen*相比老旧版本的*Eigen*，效率可能会有近10倍的差距。我们可以通过修改 __*tensorflow/workspace.bzel*__ 文件来切换*Eigen*的版本。
```
eigen_version = "97c1ebe6ccc2"
eigen_sha256 = "58ab9fa44391c850d783fe0867f42a00b5300293b7d73bbbbc8756c2e649fea2"

native.new_http_archive(
  name = "eigen_archive",
  url = "http://bitbucket.org/eigen/eigen/get/" + eigen_version + ".tar.gz",
  sha256 = eigen_sha256,
  strip_prefix = "eigen-eigen-" + eigen_version,
  build_file = str(Label("//:eigen.BUILD")),
)
```
其次，尽可能采用最新版本Tensorflow来编译。Tensorflow是一个活跃的开源项目，性能优化和改进在不断的进行中，新版本的Tensorflow往往性能上比老版的要高一些。比如较新的版本对[ThreadPool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/core/threadpool.cc)进行了重新，实现了无锁的线程池。相对之前的有锁的版本，执行效率得到了很大的提升。

最后，在编译时，一定要加上指令优化的选项，可以大大提升编译出来的代码效率。
```
bazel build -c opt --copt=mavx <target>
```
其中```--copt=mavx```表示编译时采用avx指令。

上述方法编译出来的模型只能在CPU上运行。如果你的机器支持CUDA，你可以在编译时加上```--config=cuda```，这样编译的代码在执行时将会在GPU上执行，性能比CPU会有非常大的提升。
```
bazel build -c opt --copt=mavx --config=cuda <target>
```
bazel编译生成的文件默认会在放系统的 __*/tmp*__ 目录下，并在项目根目录下用软链的方式链接。如果你想自定义输出文件的目录，可以指定```--output_base```选项。
```
bazel --output_base=/home/someone build -c opt --copt=mavx --config=cuda <target>
```
### 总结
本文主要讲解了Tensorflow c++ api的主要用法，包括训练代码，预测代码，如何编译和使用等一整套流程。通过这种方式，我们可以把模型打包成动态链接库，在使用时非常的方便。不过这种方法有个缺点，就是生成动态链接库时需要在Tensorflow的源码路径下进行编译，且需要用Bazel来编译。文章的开始也介绍了Tensorflow还提供了[c api](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h)的接口，客户端代码可以直接引用这个头文件，用c api的方式来进行操作。用这种方式，我们既可以实现和本文类似的功能，也可以通过其他语言提供的[ffi](https://en.wikipedia.org/wiki/Foreign_function_interface)来调用tensorflow对外暴露的c api，从而实现用其他语言来调用Tensorflow。

### 参考
* [Loading a tensorflow graph with the C++ API by using Mnist](http://jackytung8085.blogspot.hk/2016/06/loading-tensorflow-graph-with-c-api-by.html)
* [Learning Note: Try to downsize graph.pb model](http://jackytung8085.blogspot.hk/2016/08/tensorflow-learning-note-try-to.html)
* [Loading a TensorFlow graph with the C++ API](https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.chz3r27xt)
* [Loading TensorFlow graphs from Node.js](https://medium.com/jim-fleming/loading-tensorflow-graphs-via-host-languages-be10fd81876f#.23m71j1sd)
* [TensorFlow C++ Session API reference documentation](https://www.tensorflow.org/versions/r0.11/api_docs/cc/index.html)
* [TensorFlow in other languages](https://www.tensorflow.org/versions/r0.11/how_tos/language_bindings/index.html)
