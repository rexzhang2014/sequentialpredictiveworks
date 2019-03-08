#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import os

df = pd.read_excel("test3.xlsx",file_encoding='utf-8')
data_X = df[["x1","x2","x3","x4","x5","x6"]]
data_Y = df["label"]
plt.figure()
plt.plot(data_X)
plt.show()

normalize_data=(data_X-np.mean(data_X))/np.std(data_X)  



time_step=3      
rnn_unit=5       
batch_size=5     
input_size=6      
output_size=1     
net_type  = "LSTM"


maxit = 1001
check_path = "./checkpoint/sequential"

lr=0.0006 
#%%
def split_data(X_data, Y_data, time_step, split) :
    N = X_data.shape[0]-time_step-1
    
    X_train, Y_train=[],[]   
    
    for i in range(N-split):
        x=X_data.values[i:i+time_step,:]
        y=Y_data.values[i:i+time_step]
        X_train.append(x.tolist())
        Y_train.append(y.tolist()) 
    
    X_test, Y_test = [], []
    
    for i in range(N-split, N) :
        x=X_data.values[i:i+time_step,:]
        y=Y_data.values[i:i+time_step]
        X_test.append(x.tolist())
        Y_test.append(y.tolist()) 
    
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_data(normalize_data, data_Y, time_step, 20)
    

#%%
X=tf.placeholder(tf.float32, [None,time_step,input_size])    
Y=tf.placeholder(tf.float32, [None,time_step,output_size]) 


weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit]), name="W0"),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]), name="W1")
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,]), name="B0"),
        'out':tf.Variable(tf.constant(0.1,shape=[output_size,]), name="B1")
        }


#%%
def BuildNet(weights, bias, input_size, time_step, rnn_unit, batch_size,net_type):
    net_map = {"LSTM": ("basic_lstm_cell", tf.nn.rnn_cell.LSTMCell),
               "RNN" : ("basic_rnn_cell", tf.nn.rnn_cell.BasicRNNCell),
               }
    with tf.variable_scope("",reuse=tf.AUTO_REUSE):
        w_in=weights['in']
        b_in=biases['in']
        
        input=tf.reshape(X,[-1,input_size])  
        input_rnn=tf.matmul(input,w_in)+b_in
        input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  
    #    cell=tf.nn.rnn_cell.LSTMCell(rnn_unit, reuse=tf.AUTO_REUSE, name='basic_lstm_cell')
   
        net_name, net_class = net_map[net_type]
        cell=net_class(rnn_unit, name=net_name)
        init_state=cell.zero_state(batch_size,dtype=tf.float32)
        
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  
        
        output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    
        w_out=weights['out']
        b_out=biases['out']
        
        pred=tf.matmul(output,w_out)+b_out
    
    return pred,final_states

pred, final_states = BuildNet(weights,biases, input_size, time_step, rnn_unit, batch_size,net_type)
#%%

def train_net(pred, train_x, train_y, maxit, lr, check_path, verbose=False):

    with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE) :
    
        loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
#        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(pred,[-1]),labels=tf.reshape(Y,[-1])))
        train_op=tf.train.AdamOptimizer(lr).minimize(loss)
        saver=tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for i in range(maxit):
    
                for step in range(0, len(train_x)-batch_size, batch_size):
                    _,loss_=sess.run([train_op,loss],
                                     feed_dict={X:np.array(train_x[step:step+batch_size]).reshape(batch_size, time_step, input_size).astype(np.float),
                                                Y:np.array(train_y[step:step+batch_size]).reshape(batch_size, time_step, output_size).astype(np.float),
                                                })
    
                if verbose :
                    print(i,loss_)   
                if i % (maxit // 2) ==0:
                    
                    print("保存模型：",saver.save(sess,check_path+"-"+str(i)))
tic=datetime.now()
train_net(pred, X_train, Y_train, maxit, lr, check_path)
toc=datetime.now()

print("net type:{}, iteration:{}, elapsed:{}".format(net_type, maxit-1, str(toc-tic)))

#%%
def prediction(X_test, weights, bias, input_size, time_step, rnn_unit, batch_size,save_path,net_type):
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        pred,_=BuildNet(weights, bias, input_size, time_step, rnn_unit, batch_size,net_type)    
        saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, module_file) 
        
        predict=[]
        
        for i in range(0,len(X_test)-batch_size+1,batch_size):

            pp = sess.run(pred,feed_dict={X:X_test[i:i+batch_size]} )

            predict.append(np.array(pp).reshape(batch_size,time_step))
            
    return predict

pre_tst = prediction(X_test, weights, biases, input_size, time_step, rnn_unit,  batch_size, save_path=os.path.dirname(check_path),net_type=net_type)
pre_trn = prediction(X_train, weights, biases, input_size, time_step, rnn_unit,  batch_size, save_path=os.path.dirname(check_path),net_type=net_type)
#%%
# deal with output format
pre_tst_real = np.array([p[:,2] for p in pre_tst]).reshape(-1)
pre_trn_real = np.array([p[:,2] for p in pre_trn]).reshape(-1)



#y_real   = data_Y[2:67].values
y_tst_real = np.array([p[2] for p in Y_test]).reshape(-1)
y_trn_real = np.array([p[2] for p in Y_train]).reshape(-1)
#%%
logistic = lambda x, w, b : 1 / (1+np.exp(-w*x+b))        

plt.plot(logistic(pre_trn_real,100,0),color='y')
plt.plot(logistic(y_trn_real,100,0),color='b')
plt.show()
plt.plot(logistic(pre_tst_real,100,0),color='y')
plt.plot(logistic(y_tst_real,100,0),color='b')
plt.show()

