# pytorch-caffe
train caffe models on pytorch


# Todo
- [x] parse prototxt
- [x] parse solver
- [x] create network
- [x] train mnist
- [x] load/save weights

# Usage
python train.py --solver=examples/mnist/lenet_solver.prototxt 

# Notes
- Produce caffe_pb2.py
  - protoc --python_out=./ caffe.proto
  - pip install protobuf
# Reference
1. [python读取caffemodel文件](http://www.cnblogs.com/zjutzz/p/6185452.html?from=singlemessage&isappinstalled=0)

