#include "lstm_api.h"
#include "lstm.h"

static LSTM* glstm;

extern "C" int32_t InitLSTM(const std::string vocab_file,const std::string graph_file){
    glstm = new LSTM(vocab_file,graph_file);
    if(glstm == NULL){
         printf("new LSTM failed\n");
	 return 1;
    }
    return glstm->Init(); 
}

extern "C" int32_t UnitLSTM(){
    if(glstm==NULL){
	printf("lstm instance is NULL\n");
	return 1;
    }
    glstm->UnInit();
    delete glstm;
    glstm = NULL;
    return 0;
}

extern "C" int32_t LSTMPredict(const std::vector<std::string>& content,int& label,float& prob){
    return glstm->Predict(content,label,prob);
}
