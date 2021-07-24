import tensorflow as tf
import  numpy as np
import pandas as pd
from tensorflow.python.framework import ops 
#ops.reset_default_graph()

from sklearn.preprocessing import  OneHotEncoder
import pandas as pd
#参数设定
#特征数量
n_features = 8
#label个数
n_class = 2

#定义训练轮数
#training_steps = 1000
#学习率
learning_rate=0.01


#定义训练轮数
training_steps = 1000
#学习率
#learning_rate=0.1
#隐层K
fv=20

dnn_layer=[64,32]
dnn_active_fuc=['relu','relu','relu']


train = pd.read_csv("C:/Users/ASUS/Desktop/12/data/diabetes_train.txt",header=None,index_col=False)
test = pd.read_csv("C:/Users/ASUS/Desktop/12/data/diabetes_test.txt",header=None,index_col=False)
#数据转换

label = train.loc[:,[8]].values.reshape(-1,1)
data = train.drop(columns=8).values.reshape(-1,n_features)

y_test =  test.loc[:,[8]].values.reshape(-1,1)
X_test =  test.drop(columns=8).values.reshape(-1,n_features)

#one-hot编码
enc = OneHotEncoder()
#训练
enc.fit(label)
enc.fit(y_test)
#转换成array
label=enc.transform(label).toarray() 
y_test =enc.transform(y_test).toarray() 


def udf_full_connect(Input,input_size,output_size,activation='relu'):
    #生成或获取weights和biases
    weights=tf.get_variable("weights",[input_size,output_size],initializer=tf.glorot_normal_initializer(),trainable=True)
    biases=tf.get_variable("biases",[output_size],initializer=tf.glorot_normal_initializer(),trainable=True)
    
    #全链接 
    layer=tf.matmul(Input,weights)+biases
    if activation=="relu":
        layer=tf.nn.relu(layer)
    elif activation=="tanh":
        layer=tf.nn.tanh(layer)
        
    return layer
    
    
ops.reset_default_graph()
with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_class])
    Input_x = tf.reshape(x, shape=[-1, n_features, 1]) # None * feature_size 
    print("Input_x",Input_x)
    

# 模型参数parameter
with tf.name_scope("Parameter"):
    W = tf.Variable(tf.zeros([n_features, n_class]),name="w")
    b = tf.Variable(tf.zeros([n_class]),name="b")
    v = tf.Variable(tf.zeros([n_features, fv]),name="V")
    embeddings = tf.multiply(v, Input_x) # None * V * X 



    # 定义模型，此处使用与线性回归一样的定义
    # 因为在后面定义损失的时候会加上映射
with tf.name_scope("Prediction"):
    
    Y_liner = tf.matmul(x, W) + b
    #0.5*((sum(v*x))^2 - sum((v*x)^2)) 
    summed_features_emb = tf.reduce_sum(embeddings, 1)  # sum(v*x)
    summed_features_emb_square = tf.square(summed_features_emb)  # (sum(v*x))^2

    # square_sum part
    squared_features_emb = tf.square(embeddings) # (v*x)^2
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)   # sum((v*x)^2)

    
    Y_pair = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # 0.5*((sum(v*x))^2 - sum((v*x)^2))
    
    
    pred= tf.concat([Y_liner, Y_pair], axis=1) 
    
""" 3 Deep层网络输出 """
print("3 Deep层网络输出" )
with tf.name_scope("Deep"):
    # 第一层计算
    print("lay%s, input_size: %s, output_size: %s, active_fuc: %s" % (1, n_features*fv, dnn_layer[0], dnn_active_fuc[0]))
    with tf.variable_scope("deep_layer1", reuse=tf.AUTO_REUSE):
        input_size = n_features*fv
        output_size = dnn_layer[0]
        deep_inputs = tf.reshape(embeddings, shape=[-1, input_size]) # None * (F*K)
        print("%s: %s" % ("lay1, deep_inputs", deep_inputs))
       
        # 全连接计算    
        deep_outputs = udf_full_connect(deep_inputs, input_size, output_size, dnn_active_fuc[0])
        print("%s: %s" % ("lay1, deep_outputs", deep_outputs))
        # batch_norm
        #if is_batch_norm:
        #    deep_outputs = tf.layers.batch_normalization(deep_outputs, axis=-1, training=is_train) 
        # 输出dropout
        #if is_train and is_dropout_dnn:
        #    deep_outputs = tf.nn.dropout(deep_outputs, dropout_dnn[1])
    # 中间层计算
    
    for i in range(len(dnn_layer) - 1):
        with tf.variable_scope("deep_layer%d"%(i+2), reuse=tf.AUTO_REUSE):
            print("lay%s, input_size: %s, output_size: %s, active_fuc: %s" % (i+2, dnn_layer[i], dnn_layer[i+1], dnn_active_fuc[i+1]))
            # 全连接计算
            deep_outputs = udf_full_connect(deep_outputs, dnn_layer[i], dnn_layer[i+1], dnn_active_fuc[i+1])
            print("lay%s, deep_outputs: %s" % (i+2, deep_outputs))
            # batch_norm
            #if is_batch_norm:
            #    deep_outputs = tf.layers.batch_normalization(deep_outputs, axis=-1, training=is_train)
            # 输出dropout  
           # if is_train and is_dropout_dnn:
             #   deep_outputs = tf.nn.dropout(deep_outputs, dropout_dnn[i+2])
             
    # 输出层计算
    print("lay_last, input_size: %s, output_size: %s, active_fuc: %s" % (dnn_layer[-1], 2, dnn_active_fuc[-1]))
    with tf.variable_scope("deep_layer%d"%(len(dnn_layer)+1), reuse=tf.AUTO_REUSE):
        deep_outputs = udf_full_connect(deep_outputs, dnn_layer[-1],2, dnn_active_fuc[-1])
        print("lay_last, deep_outputs: %s" % (deep_outputs))

    # 正则化，默认L2
    dnn_regularization = 0.0
    for j in range(len(dnn_layer)+1):        
        with tf.variable_scope("deep_layer%d"%(j+1), reuse=True):
            weights = tf.get_variable("weights")
            dnn_regularization = dnn_regularization + tf.nn.l2_loss(weights)

Y_deep=deep_outputs    
concat_input = tf.concat([Y_liner, Y_pair, Y_deep], axis=1)    
Y_sum = tf.reduce_sum(concat_input, 1)
print("Y_sum",Y_sum) 
score=tf.nn.sigmoid(Y_sum,name='score')
#score=tf.reshape(score, shape=[-1, 1])
 
    
    
    
    
# 定义损失函数
with tf.name_scope("losses"):
    with tf.name_scope("error_loss"):
        print("pred",tf.shape(Y_sum))
        print("y",tf.shape(y))
        error_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(Y_sum, [-1]), labels=tf.reshape(tf.cast(tf.argmax(y,axis=1),tf.float32), [-1]))) 

    tf.add_to_collection("losses", error_loss)      #加入集合的操作

    #在权重参数上实现L2正则化
    with tf.name_scope("regularization"):
        regularizer = tf.contrib.layers.l2_regularizer(0.01)
        regularization = regularizer(W)+regularizer(v)+dnn_regularization
    tf.add_to_collection("losses",regularization)     #加入集合的操作

    #get_collection()函数获取指定集合中的所有个体，这里是获取所有损失值
    #并在add_n()函数中进行加和运算
    loss = tf.add_n(tf.get_collection("losses"))

#定义一个优化器，学习率为固定为0.01，注意在实际应用中这个学习率数值应该大于0.01
with tf.name_scope("Train"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 准确率
with tf.name_scope("accuracy"):

    #correct_prediction = tf.equal(tf.argmax(score, axis=1), tf.argmax(y, axis=1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.metrics.auc(tf.argmax(y, axis=1), score)
    tf.summary.histogram("accuracy",accuracy)
    #tf.summary.scalar("accuracy",accuracy)

merged=tf.summary.merge_all()

with tf.Session() as sess:

    tf.global_variables_initializer().run()
    sess.run(tf.local_variables_initializer())
    writer = tf.summary.FileWriter("./log",sess.graph)
    #在for循环内进行30000训练
    for i in range(training_steps):

        sess.run(train_op, feed_dict={x: data, y: label})
        
        loss_value = sess.run(loss, feed_dict={x: data, y: label})
        summary,voliadata_accuracy=sess.run([merged,accuracy],feed_dict={x: data, y: label})
        writer.add_summary(summary,i)

        #训练30000轮，但每隔2000轮就输出一次loss的值
        if i % 100 == 0 or i <= 100:
            loss_value = sess.run(loss, feed_dict={x: data, y: label})
            
            print("After %d steps, loss_value is: %f" % (i,loss_value))
            print("After %d trainging steps ,validation accuarcy is %g%%"%(i,voliadata_accuracy[0]*100))
        #xs,ys =data.train.next_batch(200)
        #sess.run(train_op,feed_dict={x:xs,y:ys})
  
    print("Testing Accuracyis %g%%"%(accuracy[0].eval({x: X_test, y:y_test})*100))
writer.close()




