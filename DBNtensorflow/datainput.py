import tensorflow as tf
import numpy as np

# file = tf.placeholder(tf.string, shape=[None])
# filename=['data.txt']
#
# dataset=tf.data.Dataset.from_tensor_slices(filename)
# dataset = dataset.flat_map(
#     lambda filename: (
#         tf.data.TextLineDataset(filename)
#         .skip(1)
#         ))
#
# dataset = dataset.batch(20)
# iterator = dataset.make_initializable_iterator()
# next=iterator.get_next()
#
# sess=tf.Session()
# sess.run(iterator.initializer, feed_dict={file: filename})
# print(sess.run(next))

fr = open('data.txt', 'r')
content = fr.readlines()
datas = []
for x in content[1:]:
    k=content[1:].index(x)
    x = x.strip().split(' ')
    datas.append([float(i) for i in x[:-2]])
    if x[-1]=='危险': datas[k].append(0)
    elif x[-1]=='可疑': datas[k].append(1)
    else: datas[k].append(2)
datas = np.array(datas)
np.random.shuffle(datas)

traind= datas[:700, :-1].astype('float32')
# m,n=features.shape
# maxMat=np.max(features,0)
# minMat=np.min(features,0)
# diffMat=maxMat-minMat
# for j in range(n):
#     features[:,j]=(features[:,j]-minMat[j])/diffMat[j]

trainl= datas[:700, -1].reshape(-1, 1).astype('float64')
assert traind.shape[0] == trainl.shape[0]

features_placeholder = tf.placeholder(traind.dtype, traind.shape)
labels_placeholder = tf.placeholder(trainl.dtype, trainl.shape)

trainset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
trainset = trainset.shuffle(buffer_size=10000)
trainset = trainset.batch(20)
trainset = trainset.repeat()

iterator = trainset.make_initializable_iterator()
next_data=iterator.get_next()

sess=tf.Session()
train=sess.run(iterator.initializer, feed_dict={features_placeholder: traind,
                                          labels_placeholder: trainl})
# print(sess.run(next_data))
# print(sess.run(next_data))








