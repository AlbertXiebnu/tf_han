import sys
import os
import argparse
import pprint
import numpy as np
from sklearn.metrics import classification_report,accuracy_score

pp = pprint.PrettyPrinter()

class DocItem:
    def __init__(self,docid,url,title,content):
        self.__docid = docid
        self.__url = url
        self.__title = title
        self.__content = content
    
    @property
    def docid(self):
        return self.__docid

    @property
    def url(self):
        return self.__url

    @property
    def title(self):
        return self.__title

    @property
    def content(self):
        return self.__content


def load(res_file,raw_file):
    pred = []
    truth = []
    with open(res_file) as f:
        for line in f:
            if line == "":
                continue
            line = line.strip('\n').split(' ')
            pred.append(float(line[0]))
            truth.append(float(line[1]))
    doclist = []
    with open(raw_file) as f:
        for line in f:
            if line == "":
                continue
            line = line.strip('\n').split('\t')
            doc = DocItem(line[1],line[3],line[4],line[5])
            doclist.append(doc)
    return np.array(pred),np.array(truth),doclist

def metrics_report(y_pred,y):
    #classification report
    print "Metrics reports:\n"
    target_names=['normal','ad']
    print(classification_report(y,y_pred,target_names=target_names))
    print("Accuracy: %s" %accuracy_score(y,y_pred)) 

def error_report(y_pred,y,doclist):
    index = (y_pred!=y)&(y==1)
    print "\nad doc predict normal:"
    cnt = 0
    for i in range(len(index)):
        if index[i] == True:
            cnt += 1
            print str(i+1)+" "+doclist[i].title+" "+doclist[i].url
    print "%d docs in total" %cnt

    index = (y_pred!=y)&(y==0)
    print "\nnormal doc predict ad:"
    cnt = 0
    for i in range(len(index)):
        if index[i] == True:
            cnt += 1
            print str(i+1)+" "+doclist[i].title+" "+doclist[i].url
    print "%d docs in total" %cnt

def main(args):
    y_pred,y,doclist = load(args.predict_file,args.raw_file)
    metrics_report(y_pred,y)
    error_report(y_pred,y,doclist)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="analysis prediction results")
    parser.add_argument("--predict_file",help="predict result file",type=str)
    parser.add_argument("--raw_file",help="raw data file",type=str)
    args = parser.parse_args()
    main(args) 
