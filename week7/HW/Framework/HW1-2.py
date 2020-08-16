#!/usr/bin/env python
# coding: utf-8

# Deep Learning Models -- A collection of various deep learning architectures, models, and tips for TensorFlow and PyTorch in Jupyter Notebooks.
# - Author: Sebastian Raschka
# - GitHub Repository: https://github.com/rasbt/deeplearning-models

# - Runs on CPU or GPU (if available)

# # Model Zoo -- Multilayer Perceptron

# ## Imports

# In[6]:


import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


# ## Settings and Dataset

# In[7]:


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 원래 코드대로 실행하니 model = model.to(device) 부분에서 invalid device ordinal 오류가 떠서 "cuda:3" -> "cuda" 로 수정

# Hyperparameters
random_seed = 1 
learning_rate = 0.1 # 우리가 구해야 할 weight를 update할 때 얼마나 update를 할지에 대한 계수. 높을 경우 overshooting 문제가 발생할 수 있으며, 낮을 경우 너무 느리고 local minimum에서 멈춰버릴 수 있다.
num_epochs = 10 # epoch의 갯수, 1 epoch는 전체 데이터 셋에 대해 전파, 역전파를 한번 거친 것을 의미한다.
batch_size = 64 # 한번의 batch마다 주는 데이터 샘플의 size를 의미한다.

# Architecture
num_features = 784 # 데이터 feature의 갯수
num_hidden_1 = 128 # 히든 레이어1 의 class 수
num_hidden_2 = 256 # 히든 레이어2 의 class 수
num_classes = 10 # 목표 변수의 class 수


##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True) #train dateset 불러오기

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor()) #test dataset 불러오기


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True) #dataloader를 통해 train data 불러오기

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False) # #dataloader를 통해 test data 불러오기

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


# In[8]:


##########################
### MODEL
##########################

class MultilayerPerceptron(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(MultilayerPerceptron, self).__init__()
        
        ### 1st hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1) # 히든 레이어의 설정
        # 밑 두줄은 Pytorch는 default값이 Xavier initialization이기 때문에 필요하지 않다. 
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        # detach : Returns a new Tensor, detached from the current graph. The result will never require gradient. 
        # The detach() method constructs a new view on a tensor which is declared not to need gradients, i.e., 
        # it is to be excluded from further tracking of operations, and therefore the subgraph involving this view is not recorded. 
        # (출처 : pytorch document)

        ### 2nd hidden layer
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()
        ### Output layer
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()
        
    def forward(self, x):
        out = self.linear_1(x)
        out = F.relu(out)
        out = self.linear_2(out)
        out = F.relu(out)
        logits = self.linear_out(out) # Logits simply means that the function operates on the unscaled output of earlier layers and that the relative scale to understand the units is linear. 
                                      # 출처 : https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits
        probas = F.log_softmax(logits, dim=1) # 신경망이 내놓은 클래스 구분 결과를 확률처럼 해석하도록 도와준다. 
        return logits, probas
    # hidden1 -> relu -> hidden2 -> relu -> logit -> softmax

    
torch.manual_seed(random_seed)
model = MultilayerPerceptron(num_features=num_features,
                             num_classes=num_classes)

model = model.to(device) # GPU 

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # optimizer 알고리즘 SGD 사용


# In[9]:


def compute_accuracy(net, data_loader): # 정확도 계산 함수
    net.eval() # test 데이터
    correct_pred, num_examples = 0, 0
    with torch.no_grad(): # 기록을 추적, 메모리 사용을 방지하기 위해 선언 (test이기 때문에 역전파가 필요없음)
        for features, targets in data_loader: 
            features = features.view(-1, 28*28).to(device) # 텐서의 모양을 변경하고 싶을때 view를 사용한다. -1경우 다른 차원으로부터 해당 값을 유추하는 것을 의미한다.
            targets = targets.to(device)
            logits, probas = net(features) 
            _, predicted_labels = torch.max(probas, 1) # 텐서 배열의 최댓값이 들어있는 index를 리턴하는 함수
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum() # 예측결과와 타겟이 일치하는 결과를 correct_pred로 더해준 뒤
        return correct_pred.float()/num_examples * 100 # 전체 갯수로 나눠서 정확도를 구한다
    

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets) # Cross-entropy 손실 함수 이용
        optimizer.zero_grad() # 역전파 단계를 실행하기 전에 gradient를 0으로 만든다
        
        cost.backward() # 역전파로 보낸다.
                        # Calling .backward() mutiple times accumulates the gradient (by addition) for each parameter. 
                        # This is why you should call optimizer.zero_grad() after each .step() call. Note that following the first .backward call, 
                        # a second call is only possible after you have performed another forward pass.
                        # 출처 : https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step() # optimizer.step is performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule.
                         # 출처 : https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost)) # 진행상황을 출력해준다.

    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))


# In[10]:


print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))

