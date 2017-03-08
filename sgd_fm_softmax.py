#############################################################################################################
# Created by qqgeogor
# https://www.kaggle.com/qqgeogor
#############################################################################################################

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt,pow
import itertools
import math
from random import random,shuffle,uniform,seed
import pickle
import sys

seed(1024)

def data_generator(path,no_norm=False,task='c',nb_classes=3):
    data = open(path,'r')
    for row in data:
        row = row.strip().split(" ")
        y = float(row[0])
        row = row[1:]
        x = []
        for feature in row:
            feature = feature.split(":")
            idx = int(feature[0])
            value = float(feature[1])
            x.append([idx,value])

        if not no_norm:
            r = 0.0
            for i in range(len(x)):
                r+=x[i][1]*x[i][1]
            for i in range(len(x)):
                x[i][1] /=r

        if task=='c':
            
            yy = [0.0 for _ in range(nb_classes)]
            yy[int(y)]+=1
            y = yy

        yield x,y


def dot(u,v):
    u_v = 0.
    len_u = len(u)
    for idx in range(len_u):
        uu = u[idx]
        vv = v[idx]
        u_v+=uu*vv
    return u_v

def mse_loss_function(y,p):
    return (y - p)**2

def mae_loss_function(y,p):
    y = exp(y)
    p = exp(p)
    return abs(y - p)

def log_loss_function(y,p):
    return -(y*log(p)+(1-y)*log(1-p))

def exponential_loss_function(y,p):
    return log(1+exp(-y*p))

def sigmoid(inX):
    return 1/(1+exp(-inX))

def bounded_sigmoid(inX):
    return 1. / (1. + exp(-max(min(inX, 35.), -35.)))


def categorical_crossentropy(y,p):
    mlogloss=0.0
    for yy,pp in zip(y,p):
        # for n in range(len(yy)):
        loss_n =-yy*log(pp)
        mlogloss+=loss_n
    return mlogloss

def softmax(inX):
    p = []

    dom = 0.0
    for x in inX:
        dom += exp(x)

    for x in inX:
        p.append(exp(x)/dom)
    return p


class SGD(object):
    def __init__(self,lr=0.001,momentum=0.9,nesterov=True,l2=0.0,l2_fm=0.0,ini_stdev= 0.01,dropout=0.5,task='c',n_components=4,nb_epoch=5,interaction=False,no_norm=False):
        self.W = []
        self.V = []
        self.n_components=n_components
        self.lr = lr
        self.l2 = l2
        self.l2_fm = l2_fm
        self.momentum = momentum
        self.nesterov = nesterov
        self.nb_epoch = nb_epoch
        self.ini_stdev = ini_stdev
        self.task = task
        self.interaction = interaction
        self.dropout = dropout
        self.no_norm = no_norm
        if self.task!='c':
            self.loss_function = mse_loss_function
            # self.loss_function = mae_loss_function
        else:
            self.loss_function = categorical_crossentropy
            # self.loss_function = log_loss_function

    def preload(self,train,test):
        train = data_generator(train,self.no_norm,task='r')
        dim = 0
        count = 0
        ys = []
        for x,y in train:
            for i in x:
                idx,value = i
                if idx >dim:
                    dim = idx
            ys.append(y)
            count+=1
        print('Training samples:',count)
        test = data_generator(test,self.no_norm,task='r')
        count=0
        for x,y in test:
            for i in x:
                idx,value = i
                if idx >dim:
                    dim = idx
            count+=1
        print('Testing samples:',count)
        
        dim = dim+1
        print("Number of features:",dim)

        self.nb_classes = len(set(ys))

        self.W = []
        self.Velocity_W = []
        self.V = []
        self.Velocity_V = []
        for _ in range(self.nb_classes):
            self.W.append([uniform(-self.ini_stdev, self.ini_stdev) for _ in range(dim)])
            self.Velocity_W.append([0.0 for _ in range(dim)])
            self.V.append([[uniform(-self.ini_stdev, self.ini_stdev) for _ in range(self.n_components)] for _ in range(dim)])
            self.Velocity_V.append([[0.0 for _ in range(self.n_components)] for _ in range(dim)]) 

        self.dim = dim



    def droupout_x(self,x):
        new_x = []
        for i, var in enumerate(x):
            if random() > self.dropout:
                del x[i]

    def _predict_fm(self,x):
        len_x = len(x)
        n_components = self.n_components
        
        self.sum_f_dict = {}
        self.sum_f_dict = [{} for c in range(self.nb_classes)]

        pred = []
        for c in range(self.nb_classes):
            p = 0.0
            for f in range(n_components):
                sum_f = 0.0
                sum_sqr_f = 0.0
                for i in range(len_x):
                    idx_i,value_i = x[i]
                    d = self.V[c][idx_i][f] * value_i
                    sum_f +=d
                    sum_sqr_f +=d*d
                p+= 0.5 * (sum_f*sum_f - sum_sqr_f);
                self.sum_f_dict[c][f] = sum_f
            pred.append(p)

        return pred

    def _predict_one(self,x):
        pred = [0.0 for _ in range(self.nb_classes)]

        for idx,value in x:
            for c in range(self.nb_classes):
                pred[c]+=self.W[c][idx]*value
        
        if self.interaction:
            p_fm = self._predict_fm(x)
            for c in range(self.nb_classes):
                pred[c]+=p_fm[c]

        if self.task=='c':
            pred = softmax(pred)
        return pred


    def _update_fm(self,lr,x,residual):
        len_x = len(x)
        sum_f_dict = self.sum_f_dict
        n_components = self.n_components

        for c in range(self.nb_classes):
            for f in range(n_components):
                for i in range(len_x):
                    idx_i,value_i = x[i]
                    
                    sum_f = sum_f_dict[c][f]
                    v = self.V[c][idx_i][f]
                    grad = (sum_f*value_i - v *value_i*value_i)*residual[c]
                    
                    self.Velocity_V[c][idx_i][f] = self.momentum * self.Velocity_V[c][idx_i][f] - lr * grad
                    if self.nesterov:
                        self.Velocity_V[c][idx_i][f] = self.momentum * self.Velocity_V[c][idx_i][f] - lr * grad
                    self.V[c][idx_i][f] = self.V[c][idx_i][f] + self.Velocity_V[c][idx_i][f] - self.l2_fm*self.V[c][idx_i][f]



    def update(self,lr,x,residual):

        if 0.<self.dropout<1.:
            self.droupout_x(x)

        for sample in x:
            for c in range(self.nb_classes):
                idx,value = sample
                grad = residual[c]*value

                self.Velocity_W[c][idx] =  self.momentum * self.Velocity_W[c][idx] - lr * grad
                if self.nesterov:
                     self.Velocity_W[c][idx] = self.momentum * self.Velocity_W[c][idx] - lr * grad
                self.W[c][idx] = self.W[c][idx] + self.Velocity_W[c][idx] - self.l2*self.W[c][idx]
        
        if self.interaction:
            self._update_fm(lr,x,residual)

    def predict(self,path,out):

        data = data_generator(path,self.no_norm,self.task,nb_classes=self.nb_classes)
        y_preds =[]
        with open(out, 'w') as outfile:
            ID = 0
            # outfile.write('%s,%s\n' % ('ID', 'target'))
            header = "ID,"+','.join(['target%s'%i for i in range(self.nb_classes)])+"\n"
            outfile.write(header)

            for d in data:
                x,y = d
                p = self._predict_one(x)
                # print p
                line = '%s,' % ID
                line +=','.join([str(i) for i in p])
                line +='\n'
                outfile.write(line)
                ID+=1



    def validate(self,path):
        data = data_generator(path,self.no_norm,self.task,nb_classes=self.nb_classes)
        loss = 0.0
        count = 0.0

        for d in data:
            x,y = d
            p = self._predict_one(x)
            loss+=self.loss_function(y,p)
            count+=1
        return loss/count

    def save_weights(self):
        weights = []
        weights.append(self.W)
        weights.append(self.V)
        weights.append(self.Velocity_W)
        weights.append(self.Velocity_V)
        weights.append(self.dim)
        pickle.dump(weights,open('sgd_fm.pkl','wb'))

    def load_weights(self):
        weights = pickle.load(open('sgd_fm.pkl','rb'))
        self.W = weights[0]
        self.V = weights[1]
        self.Velocity_W = weights[2]
        self.Velocity_V = weights[3]
        self.dim = weights[4]
        
        
    def train(self,path,valid_path = None,in_memory=False):

        start = datetime.now()
        lr = self.lr

        if in_memory:
            data = data_generator(path,self.no_norm,self.task,nb_classes=self.nb_classes)
            data = [d for d in data]
        best_loss = 999999
        best_epoch = 0
        for epoch in range(1,self.nb_epoch+1):
            if not in_memory:
                data = data_generator(path,self.no_norm,self.task,nb_classes=self.nb_classes)
            train_loss = 0.0
            train_count = 0
            for x,y in data:
                p = self._predict_one(x)
                if self.task!='c':                    
                    residual = -(y-p)
                else:
                    residual = []
                    for yi,pi in zip(y,p):
                        residual.append(-(yi-pi))

                    # residual = -(y-p)

                self.update(lr,x,residual)
                if train_count%50000==0:
                    if train_count ==0:
                        print '\ttrain_count: %s, current loss: %.6f'%(train_count,0.0)
                    else:
                        print '\ttrain_count: %s, current loss: %.6f'%(train_count,train_loss/train_count)

                train_loss += self.loss_function(y,p)
                train_count += 1

            epoch_end = datetime.now()
            duration = epoch_end-start
            
            if valid_path:
                valid_loss = self.validate(valid_path)
                print('Epoch: %s, train loss: %.6f, valid loss: %.6f, time: %s'%(epoch,train_loss/train_count,valid_loss,duration))
                if valid_loss<best_loss:
                    best_loss = valid_loss
                    self.save_weights()
                    print 'save_weights'
            else:
                print('Epoch: %s, train loss: %.6f, time: %s'%(epoch,train_loss/train_count,duration))


path = "F:\\TwoSigma\\"

'''
ploy3 best round 63 lr = 0.002,adam = True
'''

sgd = SGD(lr=0.02,momentum=0.9,nesterov=True,dropout=0.0,l2=0.0,l2_fm=0.0,task='c',n_components=4,nb_epoch=20,interaction=True,no_norm=False)
sgd.preload(path+'X.svm',path+'X_t.svm')
# sgd.load_weights()
sgd.train(path+'X_train.svm',path+'X_test.svm',in_memory=False)
sgd.predict(path+'X_test.svm',out='valid.csv')
sgd.predict(path+'X_t.svm',out='out.csv')
