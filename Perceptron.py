# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:15:16 2018

@author: Kau
"""


import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import random
import time
    
def sigmoid(x):
    return 1 / (float(1) + float(np.exp(-x)))

def sign(a):
    if a > 0:
        return 1
    else:
        return -1
    
def train_perceptron(input_x, output_y, w_init):
    w = w_init
    for i in range(len(input_x)): #for each row
        total = 0
        for j in range(len(input_x[i])): #for every element
            total += float(input_x[i][j]) * float(w_init[j])
        y = sigmoid(total)
        for j in range(len(w)):
            error = output_y[j] - y
            w[j] = w[j] + error * float(input_x[i][j])
    return w

def classify_perceptron(input_x, w):
    y_pred = []
    activation = 0
    for i in range(len(input_x)):
        for j in range(len(input_x[i])):
            activation += input_x[i][j] * float(w[j])
        y = sign(activation)
        y_pred.append(y)
    return y_pred

def main():
    BCan = np.genfromtxt('breast-cancer-wisconsin.data', usecols = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), missing_values = '?', filling_values = '0', delimiter = ',', dtype = 'int')
    
#        print(i)
    S_BCan = BCan
    for i in range(len(S_BCan[0:699, 9])):
        if S_BCan[i, 9] == 2:
            S_BCan[i, 9] = -1
        else:
            S_BCan[i, 9] = 1
#    print(S_BCan[0:699, 9])
#    np.random.shuffle(S_BCan)
    #p = x_test (partitioned data)
    p1 = S_BCan[0:70, 0:9]
    p2 = S_BCan[70:140, 0:9]
    p3 = S_BCan[140:210, 0:9]
    p4 = S_BCan[210:280, 0:9]
    p5 = S_BCan[280:350, 0:9]
    p6 = S_BCan[350:420, 0:9]
    p7 = S_BCan[420:490, 0:9]
    p8 = S_BCan[490:560, 0:9]
    p9 = S_BCan[560:630, 0:9]
    p10 = S_BCan[630:699, 0:9]
    #t = y_test (labels)
    t1 = S_BCan[0:70, 9] 
    t2 = S_BCan[70:140, 9]
    t3 = S_BCan[140:210, 9]
    t4 = S_BCan[210:280, 9]
    t5 = S_BCan[280:350, 9]
    t6 = S_BCan[350:420, 9]
    t7 = S_BCan[420:490, 9]
    t8 = S_BCan[490:560, 9]
    t9 = S_BCan[560:630, 9]
    t10 = S_BCan[630:699, 9]
    #full test cross-validation info
    Test1 = np.asarray(p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist()) 
    Test2 = np.asarray(p1.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test3 = np.asarray(p1.tolist() + p2.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test4 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test5 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test6 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test7 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test8 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p9.tolist() + p10.tolist())
    Test9 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p10.tolist())
    Test10 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist())
    # populate y_trains (labels for the training set)
    y1 = np.asarray(t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist()) 
    y2 = np.asarray(t1.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y3 = np.asarray(t1.tolist() + t2.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y4 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y5 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y6 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y7 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y8 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t9.tolist() + t10.tolist())
    y9 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t10.tolist())
    y10 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist())
    
    x_train_list = [Test1, Test2, Test3, Test4, Test5, Test6, Test6, Test7, Test8, Test9, Test10] #holds test data for training iteration.
    y_train_list = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10] #holds labels for training data
    y_test_list = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10] #holds y_test, the labels for test
    x_test_list = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10] #holds x_test, 
    weights1 = [float(0.0), 0, 0, 0, 0, 0, 0, 0, 0]
    weights2 = []
    for i in range(9):
        weights2.append(float(random.random()))
    #w = vector    
#    w = train_perceptron(input_x, input_y, w_init) CHECK LATER.    
    weights = [weights1, weights2] #initialize a list that holds both.
#    print(x_test_list[0], y_test_list[0], weights[0])
    final_weight = []
    y_pred_table = [] #holds y-predicts for each iteration
    for p in weights: #get either empty or randomly initialized weights.
        
        final_weight = p #iterating weight, starts at whatever p is, ends at 
        numtot = len(x_test_list)
        acc_vals = [] #holds accuracy values
        acc_mean = [] #hold means for each acc_vals
        acc_std = [] #hold standard deviations for each mean
        sensi_vals = [] #holds sensitivities for each 10-f validation, = TP / (TP + FN) 
        sensi_mean = [] #hold means for each fold of sensi_means
        sensi_std = [] # hold standard deviation for each mean
        speci_vals = [] #holds specificities for each 10-f validation, = TN / (TN + FP)
        speci_mean = [] #hold means for each fold of speci_vals.
        speci_std = [] #hold standard deviations for each mean 
        
        for i in range(10): #10 folds of validation.
            y_pred = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            final_weight = train_perceptron(x_train_list[i], y_train_list[i], final_weight)
            print(final_weight)
            y_pred = classify_perceptron(x_test_list[i], final_weight)
            print(y_pred)
            y_pred_table.append(y_pred)
            for j in range(len(y_pred_table)):
                if(y_pred_table[i][j] == y_test_list[i][j] and y_test_list[i][j] == 1):
                    True_Pos += 1
                if(y_pred_table[i][j] == y_test_list[i][j] and y_test_list[i][j] == -1):
                    True_Neg += 1
                if(y_pred_table[i][j] != y_test_list[i][j] and y_test_list[i][j] == 1):
                    False_Neg += 1
                if(y_pred_table[i][j] != y_test_list[i][j] and y_test_list[i][j] == -1):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / float(True_Pos + False_Neg))
            speci_vals.append((True_Neg) / float(True_Neg + False_Pos))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        #append at the end, mean of the cross-validation's accuracies, sensitivities, specificities.
        acc_mean.append(mean(acc_vals))
        sensi_mean.append(mean(sensi_vals))
        speci_mean.append(mean(speci_vals))
        #append all standard deviations at the end for the three types of data.
        acc_std.append(stat.stdev(acc_vals))
        sensi_std.append(stat.stdev(sensi_vals))
        speci_std.append(stat.stdev(speci_vals))    
    
     
    
    
    #change x/y axes, output 
    #get accuracy graph
    plt.title('Perceptron Accuracy, p=' + str(p))
    plt.xlabel('Neighbor Number')
    plt.errorbar(x, y = np.asarray(acc_mean), yerr = np.asarray(acc_std), fmt='.-', linestyle='None', marker='^')
    plt.show()
    plt.savefig('Perceptron Accuracy, p=' + str(p), bbox_inches = 'tight', dpi = 100)
    #get sensitivity graph.
    plt.title('Perceptron Sensitivity, p=' + str(p))
    plt.xlabel('Neighbor Number')
    plt.errorbar(x, y = np.asarray(sensi_mean), yerr = np.asarray(sensi_std), fmt='.-', linestyle='None', marker='^')
    plt.show()
    plt.savefig('Perceptron Sensitivity, p=' + str(p), bbox_inches = 'tight', dpi = 100)
    #get specificity graph.
    plt.title('Perceptron Specificity, p=' + str(p))
    plt.xlabel('Perceptron Neighbor Number')
    plt.errorbar(x, y = np.asarray(speci_mean), yerr = np.asarray(speci_std), fmt='.-', linestyle='None', marker='^')
    plt.show()
    plt.savefig('Perceptron Specificity, p=' + str(p), bbox_inches = 'tight', dpi = 100)
    
   
main()