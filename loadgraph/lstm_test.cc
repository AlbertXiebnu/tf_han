#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <sys/time.h>
#include "lstm.h"

using namespace std;

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));
 
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
}

void print_vec(const std::vector<std::string>& v){
    for(size_t i=0;i<v.size();i++){
	cout<<v[i]<<" ";	
    }
    cout<<endl;
}

int main(int argc,char* argv[]){
    string vocab_file = "./vocab.txt";
    string graph_file = "./lstm_frozen_graph.pb";
    if(argc!=2){
	printf("usage: %s <input_file>\n",argv[0]);
	return 1;
    }
    LSTM lstm(vocab_file,graph_file);
    lstm.Init();
    ifstream fin(argv[1]);
    string line;
    vector<string> splits;
    vector<string> subsplits;
    vector<string> doc;
    string title="";
    string content="";
    int label = 0;
    int predict_label=0;
    float prob = 0.0;
    int nhit = 0;
    int nCount = 0;
    struct timeval start;
    struct timeval end;
    unsigned long diff=0;
    while(getline(fin,line)){
	nCount++;
        splits.clear();
        SplitString(line,splits,"\t");
	if(splits.size()!=6) continue;
	label = atoi(splits[0].c_str());
	title = splits[4];
	content = splits[5];
	subsplits.clear();
	SplitString(title,subsplits," ");
	doc.clear();
	for(size_t i = 0;i<subsplits.size();i++)
	    doc.push_back(subsplits[i]);
	subsplits.clear();
	SplitString(content,subsplits," ");
	for(size_t i = 0;i<subsplits.size();i++)
	    doc.push_back(subsplits[i]);
        gettimeofday(&start,NULL);
	lstm.Predict(doc,predict_label,prob);
	gettimeofday(&end,NULL);
	diff+=1000000 *(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
	if(label == predict_label) nhit++;
        printf("doc: %d\n",nCount);
	printf("true label : %d, predict label: %d, prob: %f\n",label,predict_label,prob);
    }
    printf("Total time used: %.2f, %.2f ms per doc\n",diff*1.0/1000000,diff*1.0/1000/nCount);	
    printf("Accuracy: %.2f\n",(nhit*100.0/nCount));
    return 0;
}



