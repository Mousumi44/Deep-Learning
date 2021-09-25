# COMP 6650: Deep Learning
**Spring 2021 / Auburn University**

This repository is my implementation of assignments and projects of COMP 6650: Deep Learning. The assignments were from CS231n: Convolutional Neural Networks for Visual Recognition by Stanford University.

**Useful Link**
- Course Page: http://cs231n.stanford.edu/
- Github Page: https://cs231n.github.io/
- Lecture Videos: [YouTube](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) (2017)

## [Assignment 1](https://cs231n.github.io/assignments2019/assignment1/)
✅ Q1: k-Nearest Neighbor classifier

✅ Q2: Training a Support Vector Machine

✅ Q3: Implement a Softmax classifier

✅ Q4: Two-Layer Neural Network

✅ Q5: Higher Level Representations: Image Features

## [Assignment 2](https://cs231n.github.io/assignments2019/assignment2/)

✅ Q1: Fully-connected Neural Network

✅ Q2: Batch Normalization

✅ Q3: Dropout

✅ Q4: Convolutional Networks

✅ Q5: PyTorch on CIFAR-10

Version: PyTorch 1.4.0

## Projects
### [Image Classification](https://github.com/Mousumi44/Deep-Learning/tree/main/Projects/Image%20Classification)

In this project, I've implemented several DNN models ([Feedforward NN](https://github.com/Mousumi44/Deep-Learning/blob/main/Projects/Image%20Classification/NN%20Models/feedforward_classify.py), [CNN](https://github.com/Mousumi44/Deep-Learning/blob/main/Projects/Image%20Classification/NN%20Models/cnn_classify.py), [RNN](https://github.com/Mousumi44/Deep-Learning/blob/main/Projects/Image%20Classification/NN%20Models/rnn_classify.py), [GRU](https://github.com/Mousumi44/Deep-Learning/blob/main/Projects/Image%20Classification/NN%20Models/gru_classify.py), [LSTM](https://github.com/Mousumi44/Deep-Learning/blob/main/Projects/Image%20Classification/NN%20Models/lstm_classify.py) and [BiLSTM](https://github.com/Mousumi44/Deep-Learning/blob/main/Projects/Image%20Classification/NN%20Models/bidirectional_lstm_classify.py)) using **PyTorch** on [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html) and compared their performance. The checkpoints can be found [here](https://github.com/Mousumi44/Deep-Learning/tree/main/Projects/Image%20Classification/Checkpoints). The comparison is shown below:

|    Model    | Num Layers | Hidden Size | Num Epochs | Learning Rate | Batch Size | Accuracy (Train) | Accuracy (Test) |
|:-----------:|:----------:|:-----------:|:----------:|:-------------:|:----------:|:----------------:|:---------------:|
| Feedforward |      1     |      50     |     10     |     0.001     |     64     |      83.62 %     |     82.38 %     |
|     CNN     |      2     |     ...     |     10     |     0.001     |     64     |      88.53 %     |     87.70 %     |
|     RNN     |      2     |     256     |     10     |     0.001     |     64     |      88.06 %     |     86.37 %     |
|     GRU     |      2     |     256     |     10     |     0.001     |     64     |      89.95 %     |     87.80 %     |
|     LSTM    |      2     |     256     |     10     |     0.001     |     64     |      85.40 %     |     84.06 %     |
|    BiLSTM   |      2     |     256     |     10     |     0.001     |     64     |      84.47 %     |     83.47 %     |

