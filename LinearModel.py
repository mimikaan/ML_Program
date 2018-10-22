# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:43:23 2018

@author: MimiK
"""

#training data
x_data =[1.0,2.0,3.0]
y_data =[2.0,4.0,6.0]
w=1.0 # a random guess : random value

#our model for forwad pass
def forward(x):
    return x*w
#loss function
def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)
#compute gradient
def gradient(x,y):
    return 2*x*(x*w-y) #partial derivatives of loss function
#Before training
print("Predict (before training)", 4,forward(4))
#traing loop
for epoch in range(10):
    for x_val,y_val in zip(x_data,y_data):
        grad = gradient(x_val,y_val)
        w = w-0.01*grad
        print("\t grad: ",x_val,y_val, round(grad, 2))
        l= loss(x_val,y_val)
    print("progress: ",epoch, "w =", round(w,2), "loss =",round(l,2))
#After training
print("Predict(after training)","4 hours", forward(4))