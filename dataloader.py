import sys
import numpy as np
import itertools

class Dataset:
    def __init__(self,data,label,num_samples):
        self.data = data
        self.label = label
        self.num_samples = num_samples
        self.randind = np.random.permutation(num_samples)
        self.batch_index = 0

    def next_batch(self,batch_size = 64):
        if self.batch_index >= self.num_samples:
            self.batch_index = 0
            self.randind = np.random.permutation(self.num_samples)
        start = self.batch_index
        end = min(start+batch_size,self.num_samples)
        self.batch_index += batch_size
        return self.data[self.randind[start:end]],self.label[self.randind[start:end]]   

class Dataloader:
    def __init__(self,
            n_classes = None,
            train_path=None,
            valid_path=None,
            test_path=None,
            embedding_path=None,
            split_ratio=0.2,
            max_sen=50,
            max_len=20
            ):
        self.n_classes = n_classes
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        self.embedding_path = embedding_path
        self.split_ratio = split_ratio
        self.max_sen = max_sen
        self.max_len = max_len
        self.build()

    def load(self,path):
        total_lines = sum(1 for l in open(path))
        max_elements = self.max_sen * self.max_len
        y = np.zeros((total_lines,self.n_classes),dtype='float32')
        x = np.zeros((total_lines,max_elements),dtype='int32')
        with open(path) as f:
            for index,line in enumerate(f):
                splits = line.strip('\n').split('@')
                y[index][int(splits[0])] = 1
                sents = splits[1].split('#')
                sents = [sent.split(' ') for sent in sents]
                sents = [filter(lambda x:x!='',sen) for sen in sents]
                cnt = 0
                for s in itertools.chain(*sents):
                    if cnt >= max_elements:
                        break
                    x[index][cnt] = int(s)
                    cnt += 1
        x = x.reshape((total_lines,self.max_sen,self.max_len))
        return x,y,total_lines
   
    def rand_uniform(self,size_r,size_c,a,b):
        return a+(b-a)*np.random.rand(size_r,size_c)

    def load_embeding(self,path):
        self.vocab = set()
        with open(path) as f:
            for index,line in enumerate(f):
                line = line.strip('\n').split(' ')
                if index == 0:
                    line,dim = int(line[0]),int(line[1])
                    self.vocab_size = line
                    self.hidden_dim = dim
                    self.embedding = self.rand_uniform(line+1,dim,-1,1)
                else:
                    vec = filter(lambda x:x!='',line[1:])
                    self.embedding[index] = [ float(w) for w in vec]
                    self.vocab.add(line[0])

    def build(self):
        if self.test_path!=None:
            test_x,test_y, num_test = self.load(self.test_path)
            self.testset = Dataset(test_x,test_y,num_test)
        if self.valid_path!=None:
            valid_x,valid_y, num_valid = self.load(self.valid_path)
            self.validset = Dataset(valid_x,valid_y,num_valid)
        if self.train_path!=None:
            train_x,train_y,num_train = self.load(self.train_path)
            if self.valid_path!=None:
                self.trainset = Dataset(train_x,train_y,num_train)
            else:
                num_valid = int(num_train * self.split_ratio)
                self.validset = Dataset(train_x[:num_valid,:],train_y[:num_valid],num_valid)
                self.trainset = Dataset(train_x[num_valid:,:],train_y[num_valid:],num_train-num_valid)
        if self.embedding_path!=None:
            self.load_embeding(self.embedding_path)
