import tensorflow as tf
import dataloader as dl
import model
import pprint
import os

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

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
    
    n_classes = 2
    # load dataset
    data = dl.Dataloader(
            n_classes = n_classes,
            train_path = FLAGS.trainset,
            test_path= FLAGS.testset,
            embedding_path = FLAGS.word_vec,
            split_ratio = FLAGS.valid_ratio,
            max_sen = FLAGS.max_sen,
            max_len = FLAGS.max_len)
    # build model
    x = tf.placeholder("int32",[None,FLAGS.max_sen,FLAGS.max_len],name="input")
    y = tf.placeholder("float32",[None,n_classes],name="target")
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")

    display_step = 50

    ham = model.HAM(
            vocabsize = data.vocab_size+1,
            hiddensize = data.hidden_dim,
            rnnsize = FLAGS.rnnsize,
            docsize = FLAGS.docsize,
            max_sen = FLAGS.max_sen,
            max_len = FLAGS.max_len)
    pred = ham.build(x,keep_prob,n_classes,data.embedding)
    # define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
    if FLAGS.algo == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
    elif FLAGS.alog == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate = FLAGS.learning_rate)
    # clip gradients
    tvars = tf.trainable_variables()
    grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),5)
    optimizer = optimizer.apply_gradients(zip(grads,tvars))
    
    #grad_vars = optimizer.compute_gradients(cost)
    #clipped_grad_vars = [(tf.clip_by_value(grad,-5,5),var) for grad,var in grad_vars]
    #optimizer.apply_gradients(clipped_grad_vars)

    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    init = tf.initialize_all_variables()

    # launch graph and run
    batch_size = FLAGS.batchsize
    saver = tf.train.Saver()
    best_valid_loss = 100000
    best_valid_acc = 0
    stop_count = 0

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
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
                    loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob:FLAGS.keep_prob})
                    print "Iter " + str(step * batch_size) + ",Minibatch Loss = " + \
                            "{:.6f}".format(loss) + ", Training Accuracy = " + \
                            "{:.5f}".format(acc)
                step += 1
            # valid for 1000 examples
            acc,loss = sess.run([accuracy,cost],feed_dict={x:data.validset.data[:1000],y:data.validset.label[:1000],keep_prob:1.0})
            print "Top 1000 validset, loss ="+ "{:.6f}".format(loss) + ", accuracy = " + "{:.5f}".format(acc)
           
            # save model
            file_name = os.path.join(FLAGS.save,"model_"+str(epoch)+".ckpt")
            save_path = saver.save(sess,file_name)
            tf.train.write_graph(sess.graph.as_graph_def(),FLAGS.save,"graph.pb")
            print "Model saved in file: %s" %(save_path)
            
            if best_valid_acc < acc:
                best_valid_loss = loss
                best_valid_acc = acc
                stop_count = 0
                print "Testing Accuracy: ", sess.run(accuracy,feed_dict={x:data.testset.data,y:data.testset.label,keep_prob:1.0})
            else:
                stop_count += 1
            
            if stop_count >= FLAGS.early_stop_count:
                print "early stop training at epoch %d" %epoch
                break

        print "Optimization finished!"
        print "Testing Accuracy: ", sess.run(accuracy,feed_dict={x:data.testset.data,y:data.testset.label,keep_prob:1.0})

if __name__ == "__main__":
    tf.app.run()
