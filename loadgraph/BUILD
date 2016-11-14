cc_binary(
    name = "mnistpredict.so",
    srcs = ["mnist.cc","MNIST.h"],
    linkshared = 1,    
    deps = [
        "//tensorflow/core:tensorflow",
    ],
)

cc_binary(
    name = "lstm.so",
    srcs = ["lstm_api.cc","lstm_api.h","lstm.cc","lstm.h"],
    linkshared = 1,    
    deps = [
        "//tensorflow/core:tensorflow",
    ],
)

cc_binary(
    name = "lstm_test",
    srcs = ["lstm_test.cc","lstm.cc","lstm.h"],
    deps = [
    	"//tensorflow/core:tensorflow",
    ],
)
