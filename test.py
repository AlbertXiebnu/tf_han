import tensorflow as tf
import dataloader as dl
import model
import pprint
import os
import time

flags = tf.app.flags
flags.DEFINE_string("trainset","./data/trainset.txt","train set file path")
flags.DEFINE_string("testset","./data/testset.txt","test set file path")
flags.DEFINE_string("word_vec","./data/word_embeddings","word embedding file path")
flags.DEFINE_string("save","./model","model save path")
flags.DEFINE_integer("batchsize",128,"batch size")
flags.DEFINE_integer("max_sen",50,"max sentence number in one doc")
flags.DEFINE_integer("max_len",20,"max words in one sentence")
flags.DEFINE_float("valid_ratio",0.2,"validset ratio in trainset")
flags.DEFINE_integer("rnnsize",128,"sentence embedding size")
flags.DEFINE_integer("docsize",64,"doc embedding size")
flags.DEFINE_string("algo","adam","optim algorithm. adam | rmsprop")
flags.DEFINE_float("learning_rate",0.001,"learning rate")
flags.DEFINE_float("momentum",0.9,"momentum")
flags.DEFINE_float("keep_prob",1.0,"keep probability for dropout layer")
flags.DEFINE_integer("batchnorm",0,"batch normalization. 0 | 1")
flags.DEFINE_integer("max_epochs",5,"maximum epoch in training")
flags.DEFINE_integer("early_stop_count",2,"early stop count")
FLAGS = flags.FLAGS


def batch_eval(session,metrics,feed_dict,dataset,save=None):
    acc,label = session.run(metrics,feed_dict=feed_dict)
    print "Accuracy: %f" %acc 
    if save != None:
        outf = open(save,"w")
        num_examples = label.shape[0]
        for i in range(0,num_examples):
            outf.write(str(label[i])+" "+str(dataset.label[i][1])+"\n")
        outf.close()
        print "save predict results in file %s" %save

def single_eval(session,accuracy,x,y,keep_prob,dataset,batch_size):
    step = 1
    iters = dataset.num_samples
    start = time.time()
    while step * batch_size < iters:
        batch_x,batch_y = dataset.next_batch(batch_size)
        acc = session.run(accuracy,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
        print "Accuracy in batch: %f" %acc
        step += 1
    end = time.time()
    print "Time used: "+"{:.2f}".format(end-start)+ "s, " + "{:.3f}".format((end-start)*1000.0/iters)+"ms per doc"


def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
    
    n_classes = 2
    # load dataset
    data = dl.Dataloader(
            n_classes = n_classes,
            test_path= FLAGS.testset,
            embedding_path = FLAGS.word_vec,
            split_ratio = FLAGS.valid_ratio,
            max_sen = FLAGS.max_sen,
            max_len = FLAGS.max_len)
    # build model
    x = tf.placeholder("int32",[None,FLAGS.max_sen,FLAGS.max_len],name="input")
    y = tf.placeholder("float32",[None,n_classes],name="target")
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")

    ham = model.HAM(
            vocabsize = data.vocab_size+1,
            hiddensize = data.hidden_dim,
            rnnsize = FLAGS.rnnsize,
            docsize = FLAGS.docsize,
            max_sen = FLAGS.max_sen,
            max_len = FLAGS.max_len)
    pred = ham.build(x,keep_prob,n_classes,data.embedding)
    pred_label = tf.argmax(pred,1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
    correct_pred = tf.equal(pred_label,tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    
    # load testset 
    data2_path = "./data/testset2.txt"
    data2 = dl.Dataloader(n_classes = n_classes,test_path = data2_path,max_sen = FLAGS.max_sen,max_len = FLAGS.max_len)
    # load short text testset
    data3_path = "./data/testset3.txt"
    data3 = dl.Dataloader(n_classes = n_classes,test_path = data3_path,max_sen = FLAGS.max_sen,max_len = FLAGS.max_len)

    metrics = [accuracy,pred_label]
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True,device_count={'GPU':0})
    #config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(init)
        model_path = os.path.join(FLAGS.save,"model.ckpt")
        saver.restore(sess,model_path)
        
        # test: data
        print "main testset results:"
        res1 = "data/testset.res"
        feed_dict = {x:data.testset.data,y:data.testset.label,keep_prob:1.0}
        batch_eval(sess,metrics,feed_dict,data.testset,save=res1)
        # test: dataset 2
        print "testset 2 results:"
        feed_dict = {x:data2.testset.data,y:data2.testset.label,keep_prob:1.0}
        batch_eval(sess,metrics,feed_dict,data2.testset)
        # test: dataset 3
        print "testset 3 results: (short text)"
        feed_dict = {x:data3.testset.data,y:data3.testset.label,keep_prob:1.0}
        batch_eval(sess,metrics,feed_dict,data3.testset)

        # test data, one doc per run
        batch_size = 1
        #single_eval(sess,accuracy,x,y,keep_prob,data.testset,batch_size)

if __name__ == "__main__":
    tf.app.run()
