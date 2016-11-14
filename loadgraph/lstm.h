#pragma once
#ifndef _LSTM_H_
#define _LSTM_H_

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;
using namespace chrono;
using namespace tensorflow;

class LSTM{
public:
    LSTM(const string vocab_file,const string graph_file);
    int32_t Init();
    void UnInit();
    int32_t LoadVocab();
    int32_t FillFeatureVec(const vector<string>& content);
    int32_t Predict(const vector<string>& content,int& label,float& prob);
private:
    GraphDef m_graph_def;
    Session* m_sess;
    map<string,int32_t> m_vocab;
    Tensor x;
    int32_t m_vocab_size;
    string m_vocab_file;
    string m_graph_file;
    int32_t max_sen_num;
    int32_t max_seq_len;
    int32_t max_length;
    int32_t batch_size;
    int32_t num_classes;
    int32_t* feature_vec;
    int32_t* padding_vec;
};

#endif
