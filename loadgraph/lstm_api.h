#ifndef _LSTM_API_H_
#define _LSTM_API_H_

#include <string>
#include <vector>

extern "C" {
    int32_t InitLSTM(const std::string vocab_file,const std::string graph_file);
    int32_t UnitLSTM();
    int32_t LSTMPredict(const std::vector<std::string>& content,int& label,float& prob);
} 

#endif
