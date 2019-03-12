# from data_load import load_task

# if __name__=='__main__':
#     print(load_task('data/tasks_1-20_v1-2/en',1))
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import numpy as np
from data_load import load_task,vectorize_data
from sklearn import model_selection,metrics
from itertools import chain
from six.moves import range,reduce
from memnn import MemN2N

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_integer('task_id',1,'bAbi task id, 1<=id<=20')
tf.flags.DEFINE_integer('memory_size',50,'Maximum size of Memory')
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learning rate.")
tf.flags.DEFINE_string('data_dir','data/tasks_1-20_v1-2/en','Directory containing bAbI tasks')
tf.flags.DEFINE_integer('random_state',None,'Random_state')
tf.flags.DEFINE_integer('batch_size',32,'batch_size for training')
tf.flags.DEFINE_integer('embedding_size',20,'Embedding size for embedding matrices.')
tf.flags.DEFINE_integer('hops',3,'Number of hops in the Memory Network.')
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer('epochs',100,'Num of epochs to train for')
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")


FLAGS = tf.flags.FLAGS

print('Start task:',FLAGS.task_id)

train,test = load_task(FLAGS.data_dir,FLAGS.task_id)
#print(train[0][2])#(1000-场景个数,3-每个场景包含的内容长度,2-上下文个数,5-第一句话的单词数) [0][0]表示上下文[0][1]表示问题 [0][2]表示答案
#print(len(test))
data = train+test
#print(data)
#print(np.array(data).shape)(2000,3)
#list(chain.from_iterable([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']]))#连成一个iterable对象
#['mary', 'moved', 'to', 'the', 'bathroom', 'john', 'went', 'to', 'the', 'hallway']

vocab = sorted(reduce(lambda x, y: x | y,(set(list(chain.from_iterable(context)) + question + answer) for context,question,answer in data)))
#print(vocab)只有19个单词
word_idx = dict((w,i+1) for i,w in enumerate(vocab))
max_story_size = max(map(len, (s for s,_,_ in data)))#最长的story 10
mean_story_size = int(np.mean([len(s) for s,_,_ in data]))# 6
sentence_size = max(map(len,chain.from_iterable(s for s,_,_ in data)))#6
query_size = max(map(len,(q for _,q,_ in data)))#3
memory_size = min(FLAGS.memory_size,max_story_size)#10

for i in range(memory_size):
	word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

vocab_size = len(word_idx)+1 #+1是为了nil word
sentence_size = max(query_size,sentence_size)
sentence_size = sentence_size+1#+1是为了time标号

# print("Longest sentence length", sentence_size)
# print("Longest story length", max_story_size)
# print("Average story length", mean_story_size)
S,Q,A = vectorize_data(train,word_idx,sentence_size,memory_size)
#划分训练集和验证集
train_S,val_S,train_Q,val_Q,train_A,val_A = model_selection.train_test_split(S,Q,A,test_size=0.1,random_state = FLAGS.random_state)
#测试集
test_S,test_Q,test_A = vectorize_data(test,word_idx,sentence_size,memory_size)

#print(test_S[0])
#print('Training set shape:' , train_S.shape)#(900,10,7)

n_train = train_S.shape[0]
n_test = test_S.shape[0]
n_val = val_S.shape[0]

# print("Training Size", n_train)
# print("Validation Size", n_val)
# print("Testing Size", n_test)

train_labels = np.argmax(train_A,axis = 1)#选择每个答案所处的向量index
test_labels = np.argmax(test_A,axis = 1)
val_labels = np.argmax(val_A,axis = 1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size#设置batch size

batches = zip(range(0,n_train - batch_size,batch_size),range(batch_size, n_train, batch_size))
batches = [(start,end) for start,end in batches]
#[(0, 32), (32, 64), (64, 96), (96, 128), (128, 160), 
#(160, 192), (192, 224), (224, 256), (256, 288), (288, 320),
# (320, 352), (352, 384), (384, 416), (416, 448), (448, 480), 
#(480, 512), (512, 544), (544, 576), (576, 608), (608, 640), (640, 672), (672, 704),
# (704, 736), (736, 768), (768, 800), (800, 832), (832, 864), (864, 896)]
with tf.Session() as sess:
	 model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)
	 for t in range(1,FLAGS.epochs+1):
	 	#learning rate 随epoch改变
	 	if t-1 <= FLAGS.anneal_stop_epoch:
	 		anneal = 2.0 ** ((t-1)//FLAGS.anneal_rate)
	 	else:
	 		anneal = 2.0 ** (FLAGS.anneal_stop_epoch//FLAGS.anneal_rate)
	 	lr = FLAGS.learning_rate / anneal

	 	np.random.shuffle(batches)
	 	total_cost = 0.0
	 	for start,end in batches:
	 		s = train_S[start:end]
	 		q = train_Q[start:end]
	 		a = train_A[start:end]
	 		cost_t = model.batch_fit(s,q,a,lr)
	 		total_cost += cost_t

	 	if t % FLAGS.evaluation_interval == 0:
	 		train_preds = []
	 		for start in range(0,n_train,batch_size):
	 			end = start+batch_size
	 			s = train_S[start:end]
	 			q = train_Q[start:end]
	 			pred = model.predict(s,q)
	 			train_preds += list(pred)

	 		val_pred = model.predict(val_S,val_Q)
	 		train_acc = metrics.accuracy_score(np.array(train_preds),train_labels)
	 		val_acc = metrics.accuracy_score(np.array(val_pred),val_labels)

	 		print('---------------------------')
	 		print('Epoch',t)
	 		print('total loss',total_cost)
	 		print('train accuracy:',train_acc)
	 		print('validation accuracy:',val_acc)
	 		print('---------------------------')
	 test_preds = model.predict(test_S, test_Q)
	 test_acc = metrics.accuracy_score(test_preds, test_labels)
	 print("Testing Accuracy:", test_acc)















