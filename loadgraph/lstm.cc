#include "lstm.h"

LSTM::LSTM(const string vocab_file,const string graph_file){
    m_vocab_file = vocab_file;
    m_graph_file = graph_file;
    m_sess = NULL;
    max_sen_num = 50;
    max_seq_len = 20;
    max_length = max_sen_num * max_seq_len;
    batch_size = 1;
    num_classes = 2;
}

int32_t LSTM::LoadVocab(){
    ifstream fin(m_vocab_file.c_str());
    if(!fin){
        printf("vocab file not exists.\n");
        return 1;
    }
    string line;
    int index=1;
    while(getline(fin,line)){
        if(line=="") continue;
        if(m_vocab.find(line)==m_vocab.end()){
            m_vocab[line] = index;
            index++;
        }
    }
    m_vocab_size = index - 1;
    printf("vocab size: %d\n",m_vocab_size);
    return 0;
}

int32_t LSTM::Init(){
    int32_t res = LoadVocab();
    if(res) return res;
	SessionOptions sess_options = SessionOptions();
	//sess_options.config.set_intra_op_parallelism_threads(64);
	//sess_options.config.set_inter_op_parallelism_threads(64);
    Status status = NewSession(sess_options,&m_sess);
    if(!status.ok()){
        cout<<status.ToString()<<endl;
        return 1;
    }

    status = ReadBinaryProto(Env::Default(),m_graph_file.c_str(),&m_graph_def);
    if(!status.ok()){
        cout<<status.ToString()<<endl;
        return 1;
    }

    status = m_sess->Create(m_graph_def);
    if(!status.ok()){
        cout<<status.ToString()<<endl;
        return 1;
    }
    
    Tensor t(DT_INT32,TensorShape({batch_size,max_sen_num,max_seq_len}));
    x = t;
    feature_vec = (int32_t*) malloc(max_length * sizeof(int32_t));
    padding_vec = (int32_t*) malloc(max_length * sizeof(int32_t));
    for(int i = 0; i < max_length; i++){
    	feature_vec[i] = 0;
        padding_vec[i] = 0;
    }
    return 0;
}

void LSTM::UnInit(){
    m_sess->Close();
    m_sess = NULL;
    free(feature_vec);
    free(padding_vec);
}

int32_t LSTM::FillFeatureVec(const vector<string>& content){
    int32_t index = 0;
    for(size_t i=0; i < content.size(); i++){
        if(m_vocab.find(content[i])!=m_vocab.end() && index < max_length){
            feature_vec[index++] = m_vocab[content[i]];
        }
    }
    return index;
}


int32_t LSTM::Predict(const vector<string>& content,int& label,float& prob){
    int32_t fill_len = FillFeatureVec(content);
    //for(size_t i = 0;i<vec.size();i++){
    //	cout<<vec[i]<<" ";
    //}
    //cout<<endl;
    //Tensor x(DT_INT32,TensorShape({batch_size,max_sen_num,max_seq_len}));
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
    vector<pair<string,Tensor>> inputs = {{"input",x},{"keep_prob",keep_prob}};
    vector<Tensor> outputs;
    Status status = m_sess->Run(inputs,{"output"},{},&outputs);
    if(!status.ok()){
        cout << status.ToString()<<endl;
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
