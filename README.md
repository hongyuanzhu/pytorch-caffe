# pytorch-caffe
train caffe models on pytorch


# Todo
- [x] parse prototxt
- [x] parse solver
- [x] create network
- [x] train mnist
- [x] load/save weights
- [ ] test time consuming

# Usage
- python main.py train --solver=examples/mnist/lenet_solver.prototxt 
- python main.py train --solver=examples/mnist/lenet_solver.prototxt --weights=000000.pth.tar
- python main.py time  --model=test.prototxt
