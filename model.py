import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell

"""
Build Hieracical LSTM attention model for classification
input: x (model inputs)
input: opts (parameters)
output: model 
"""
class HAM():
    def __init__(self, vocabsize, hiddensize, rnnsize,
            docsize, max_sen, max_len):
        self.vocabsize = vocabsize
        self.hiddensize = hiddensize
        self.rnnsize = rnnsize
        self.docsize = docsize 
        self.max_sen = max_sen
        self.max_len = max_len

    def activation(self,x,w):
        out = tf.tanh(tf.add(tf.matmul(x,w['attention_w']),w['attention_b']))
        sim = tf.matmul(out,w['attention_c'])
        return sim
    
    """
    Attention layer
    inputs: usually LSTM outputs list. inputs[i] is cell output at time t
    input shape: step_size * [batch_size x hidden_dim ]
    output shape: batch_size x hidden_dim
    """
    def attention_layer(self,inputs,w):
        step_size = len(inputs)
        weights=[]
        outputs=[]
        for x in inputs:
            a = self.activation(x,w)
            weights.append(a)
        attentions = tf.concat(1,weights)
        attentions = tf.nn.softmax(attentions) # b x seqlen
        attentions = tf.split(1,step_size,attentions) # seqlen * [bx1]
        for i in range(step_size):
            a = tf.mul(inputs[i],attentions[i])
            outputs.append(a)
        outputs = tf.add_n(outputs)
        #outputs = tf.concat(0,outputs)
        return outputs

    """
    input shape: batch_size x seq_len
    output shape: batch_size x sent_embedding
    """
    def sentence_embedding(self, inputs, keep_prob, w):
        with tf.device('/cpu:0'):
            embedding_layer = tf.nn.embedding_lookup(w['word_embedding_w'],inputs)
        # batch_size x max_len x word_embedding
        cell_input = tf.transpose(embedding_layer,[1,0,2])
        cell_input = tf.reshape(cell_input,[-1,self.hiddensize])
        cell_input = tf.split(0,self.max_len,cell_input)
        with tf.variable_scope('forward'):
            lstm_fw_cell = rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(self.rnnsize,forget_bias=1.0,state_is_tuple=True),input_keep_prob=keep_prob,output_keep_prob=keep_prob)
        with tf.variable_scope('backward'):
            lstm_bw_cell = rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(self.rnnsize,forget_bias=1.0,state_is_tuple=True),input_keep_prob=keep_prob,output_keep_prob=keep_prob)
        outputs,_,_ = rnn.bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,cell_input,dtype=tf.float32)
        # outputs shape: seq_len x [batch_size x (fw_cell_size + bw_cell_size)]
        att = self.attention_layer(outputs,w)
        return att
        
    """
     inputs: shape -> batch_size x max_sen  x max_len
    """
    def build(self, inputs, keep_prob, n_classes, word_embedding):
        
        inputs = tf.transpose(inputs,[1,0,2])
        inputs = tf.reshape(inputs,[-1,self.max_len])
        inputs = tf.split(0, self.max_sen, inputs)

        variable_dict = {
            "word_embedding_w": tf.get_variable(name="word_embedding",shape=[self.vocabsize,self.hiddensize],initializer=tf.constant_initializer(word_embedding),trainable=True),
            "attention_w" : tf.get_variable(name="word_attention_weights",shape=[2*self.rnnsize,2*self.rnnsize]),
            "attention_b" : tf.get_variable(name="word_attention_bias",shape=[2*self.rnnsize]),
            "attention_c" : tf.get_variable(name="word_attention_context",shape=[2*self.rnnsize,1]),
        }
        
        sent_embeddings = []
        with tf.variable_scope("embedding_scope") as scope:
            for x in inputs:
                embedding = self.sentence_embedding(x,keep_prob,variable_dict)
                sent_embeddings.append(embedding)
                scope.reuse_variables()
        
        with tf.variable_scope('forward'):
            lstm_fw_cell = rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(self.docsize,forget_bias=1.0,state_is_tuple=True),input_keep_prob=keep_prob,output_keep_prob=keep_prob)
        with tf.variable_scope('backward'):
            lstm_bw_cell = rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(self.docsize,forget_bias=1.0,state_is_tuple=True),input_keep_prob=keep_prob,output_keep_prob=keep_prob)
        outputs, _ , _ = rnn.bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,sent_embeddings,dtype=tf.float32)
        
        atten_variable_dict = {
            "attention_w" : tf.get_variable(name="sent_attention_weights", shape=[2*self.docsize,2*self.docsize]),
            "attention_b" : tf.get_variable(name="sent_attention_bias", shape=[2*self.docsize]),
            "attention_c" : tf.get_variable(name="sent_attention_context", shape=[2*self.docsize,1]),
        }
        
        att = self.attention_layer(outputs,atten_variable_dict)
        # full connected layer
        W = tf.get_variable("fullconnect_weights",shape=[2 * self.docsize,n_classes])
        B = tf.get_variable("fullconnect_bias",shape=[n_classes])
        output = tf.add(tf.matmul(att,W),B,name="output")
        return output

