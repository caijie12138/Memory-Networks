from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np

def position_encoding(embedding_size,sentence_size):#参数为每个单词的embedding维度 和 句子的长度
    #首先创造一个全为1的位置编码矩阵 size为（embedding_size,sentence_size）
    #['we','are','family']假设每个单词编码为4维
    #[[0,1,0][2,3,4],[4,5,6],[7,8,9]]
    encoding = np.ones((embedding_size,sentence_size),dtype = np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1,le):
        for j in range(1,ls):
            #实现和论文中的section4.1不太一样 但是都是引入位置信息
            encoding[i-1,j-1] = (i-(embedding_size+1)/2)*(j-(sentence_size+1)/2)#行从60+减小到-60 列也是相同
    encoding = 1+4*encoding/embedding_size/sentence_size #将值进行缩小的操作
    encoding[:,-1] = 1.0#最后一列全部变成1.0
    return np.transpose(encoding)#转置之后相当于最后一个字的词向量变成50维的1 其余单词的词向量逐渐减小 返回维度是（6，50）

def add_gradient_noise(t,stddev=1e-3,name = None):
    #给梯度裁剪之后的值添加噪音
    with tf.op_scope([t,stddev],name,'add_gradient_noise') as name:
        t = tf.convert_to_tensor(t,name='t')
        gn = tf.random_normal(tf.shape(t),stddev=stddev)
        return tf.add(t,gn,name=name)

def zero_nil_slot(t,name=None):
    #将t的第一行用0代替 猜想原因是因为每句话的都加了一个nil符号吧
    with tf.op_scope([t],name,'zero_nil_slot') as name:
        t = tf.convert_to_tensor(t,name = 't')
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1,s]))
        return tf.concat(axis=0,values=[z,tf.slice(t,[1,0],[-1,-1])],name=name)

class MemN2N(object):
    #End-to-End Memory network
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size,
        hops = 3,
        max_grad_norm = 40.0,
        nonlin = None,
        initializer = tf.random_normal_initializer(stddev = 0.1),
        encoding = position_encoding,
        session = tf.Session(),
        name = 'MemN2N'
        ):
        '''
        batch_size 每个batch包含的例子个数
        vocab_size 词汇表中单词的个数
        sentence_size 句子长度
        memory_size 记忆中包含的句子数
        embedding_size 每个单词表示成词向量后 词向量的维数
        hops 一个hop包含一次读取和定位memory（softmax）详见代码
        max_grad_norm l2正则化
        nonlin 非线性标识 ，默认None
        initializer 权重初始化函数
        encoding 位置编码矩阵 详见顶部函数
        '''
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._name = name

        self.build_input()#占位符
        self.build_vars()#AC矩阵

        self._opt = tf.train.GradientDescentOptimizer(learning_rate = self.lr)
        self._encoding = tf.constant(encoding(self._embedding_size,self._sentence_size),name = 'encoding')

        #计算交叉熵
        logits = self._inference(self._stories, self._queries)#(batch_size ? , vocab_size 30)self._answer(?,30)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = tf.cast(self._answers,tf.float32),name = 'cross_entropy')
        loss_op = tf.reduce_sum(cross_entropy)

        #梯度下降
        #grad_and_vars = self._opt.minimize(loss_op)
        #梯度裁剪 主要的作用是防止梯度爆炸 根据类l2正则化项来做 参考博客https://blog.csdn.net/wn87947/article/details/82345537
        grad_and_vars = self._opt.compute_gradients(loss_op)
        grad_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grad_and_vars]
        grad_and_vars = [(add_gradient_noise(g),v) for g,v in grad_and_vars]#加入噪声
        nil_grad_and_vars = []
        for g,v in grad_and_vars:
            if v.name in self._nil_vars:
                nil_grad_and_vars.append((zero_nil_slot(g),v))
            else:
                nil_grad_and_vars.append((g,v))
        train_op = self._opt.apply_gradients(nil_grad_and_vars,name = 'train_op')

        #predict 操作
        predict_op = tf.argmax(logits,1,name = 'predict_op')
        predict_proba_op = tf.nn.softmax(logits,name='predict_proba_op')
        predict_log_prob_op = tf.log(predict_proba_op,name='predict_log_prob_op')

        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_prob_op = predict_log_prob_op
        self.train_op = train_op

        self._sess = session
        self._sess.run(tf.global_variables_initializer())

    def build_input(self):
        #stories queries answers的占位符 以及学习率
        self._stories = tf.placeholder(tf.int32,[None,self._memory_size,self._sentence_size],name='stories')
        self._queries = tf.placeholder(tf.int32,[None,self._sentence_size],name='queries')
        self._answers = tf.placeholder(tf.int32,[None,self._vocab_size],name = 'answers')
        self.lr = tf.placeholder(tf.float32,name='learning_rate')

    def build_vars(self):
        '''
        创建AC矩阵 详见模型图 AC的shape是[vocab_size,embedding_size]
        '''
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1,self._embedding_size])
            A = tf.concat(axis = 0, values = [nil_word_slot, self._init([self._vocab_size-1, self._embedding_size])])#(30,20)
            C = tf.concat(axis = 0, values = [nil_word_slot, self._init([self._vocab_size-1, self._embedding_size])])
            self.A_1 = tf.Variable(A,name = 'A')#重名的话创建一个新的 get_variable()创建一个共享变量
            self.C = []

            for hopn in range(self._hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C,name='C'))
            # Dont use projection for layerwise weight sharing
            # self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")

            # Use final C as replacement for W
            # self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")

            #[x.name for x in self.C] ['MemN2N/hop_0/C:0', 'MemN2N/hop_1/C:0', 'MemN2N/hop_2/C:0']
        self._nil_vars = set([self.A_1.name]+[x.name for x in self.C])

    def _inference(self,stories,queries):
        with tf.variable_scope(self._name):
            #相邻层的AC相等 也就是Ak+1 = Ck W等于顶层的C B等于底层的C
            q_emb = tf.nn.embedding_lookup(self.A_1,queries)#(30,20) (?,7) (?,7,20) 注意embedding_lookup的时候使用list和tf.Variable是不同的结果
            u_0 = tf.reduce_sum(q_emb*self._encoding,1)#(?,20) 7个单词进行求和
            u = [u_0]

            for hopn in range(self._hops):
                if hopn == 0:
                    #stories(?, 10, 7)
                    m_emb_A = tf.nn.embedding_lookup(self.A_1,stories)
                    #m_emb_A(?, 10, 7, 20) #self._encoding (7,20)
                    #print((m_emb_A*self._encoding).shape)#(?, 10, 7, 20)
                    m_A = tf.reduce_sum(m_emb_A*self._encoding , 2)#(?, 10, 20) 和q的操作一样
                else:
                    with tf.variable_scope('hop_{}'.format(hopn-1)):#因为是Ak+1 = Ck
                        m_emb_A = tf.nn.embedding_lookup(self.C[hopn-1],stories)
                        m_A = tf.reduce_sum(m_emb_A*self._encoding , 2) 
                #tf.expand_dims(u[-1],-1)(?,20,1)
                u_temp = tf.transpose(tf.expand_dims(u[-1],-1),[0,2,1])# (?, 1, 20)
                dotted = tf.reduce_sum(m_A * u_temp,2) #(m_A * u_temp) (?, 10, 20)*(?, 1, 20) = (?, 10, 20) (?,10)
                #softmax算概率
                probs = tf.nn.softmax(dotted)
                probs_temp = tf.transpose(tf.expand_dims(probs,-1),[0,2,1])#(?,1,10)
                with tf.variable_scope('hop_{}'.format(hopn)):
                    m_emb_C = tf.nn.embedding_lookup(self.C[hopn],stories)
                    #m_emb_C(?, 10, 7, 20)
                m_C = tf.reduce_sum(m_emb_C*self._encoding,2)#(?, 10, 20)

                c_temp = tf.transpose(m_C,[0,2,1])#(?,20,10)
                o_k = tf.reduce_sum(c_temp * probs_temp,2)#(?,20)

                u_k = u[-1]+o_k#(?,20)

                if self._nonlin:
                    u_k = nonlin(u_k)
                u.append(u_k)
            #将最后一个C作为输出
            # print(self.C[-1].shape)(30,20)
            # print(tf.transpose(self.C[-1], [1,0]).shape)(20,30)
            with tf.variable_scope('hop_{}'.format(self._hops)):
                return tf.matmul(u_k,tf.transpose(self.C[-1],[1,0]))#(?,20) (20,30) 30是vocab_size

    def batch_fit(self,stories,queries,answers,learning_rate):
        feed_dict = {self._stories:stories,self._queries:queries,self._answers:answers,self.lr:learning_rate}
        loss,_ = self._sess.run([self.loss_op,self.train_op],feed_dict = feed_dict)
        return loss

    def predict(self,stories,queries):
        feed_dict = {self._stories:stories,self._queries:queries}
        return self._sess.run(self.predict_op,feed_dict = feed_dict)

    def predict_prob(self,stories,queries):
        feed_dict = {self._stories:stories,self._queries:queries}
        return self._sess.run(self.predict_proba_op,feed_dict = feed_dict)

    def predict_log_prob(self,stories,queries):
        feed_dict = {self._stories:stories,self._queries:queries}
        return self._sess.run(self.predict_log_prob_op,feed_dict = feed_dict)
















