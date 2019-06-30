# PDE-DNN

This repository contains the numerical examples from the paper "Data driven approximation of parametrized PDEs by Reduced Basis and Neural Networks", by N. Dal Santo, S. Deparis and L. Pegolotti. If you use the code, please cite the following reference [arXiv:1904.01514](https://arxiv.org/abs/1904.01514).


In this work we propose a novel way to integrate data and PDE simulations by combining DNNs and RB solvers for the prediction of the solution of a parametrized PDE. The proposed architecture features a MLP followed by a RB solver, acting as nonlinear activation function. The output of the MLP is interpreted as a prediction of parameter dependent quantities: physical parameters, theta functions of the approximated affine decomposition and approximated RB solutions. Compared to standard DNNs, we obtain as byproduct the solution in the full physical space and, for affine dependencies, the value of the parameter. Compared to the RB method, we obtain accurate solutions with a smaller number of affine components by solving a linear problem instead of a nonlinear one.

# Running the test

To train and use the networks, you need to have a working installation of [TensorFlow](https://www.tensorflow.org/). The RB structures during the offline phase of the RB method have been generated with [PyORB](https://github.com/ndalsanto/pyorb) and you need to have a working installation in order to run the example. PyORB itself must rely on a finite element (FE) library, which can be connected through the [pyorb-matlab-api](https://github.com/ndalsanto/pyorb-matlab-api). In the example provided [feamat](https://github.com/lucapegolotti/feamat) has been used as FE backend.
