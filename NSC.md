
# train.py


```python

#-*- coding: UTF-8 -*-  
import sys
from Dataset import *
from LSTMModel import LSTMModel

dataname = sys.argv[1]
classes = sys.argv[2]
voc = Wordlist('../data/'+dataname+'/wordlist.txt')

trainset = Dataset('../data/'+dataname+'/train.txt', voc)
devset = Dataset('../data/'+dataname+'/dev.txt', voc)
print 'data loaded.'

model = LSTMModel(voc.size,trainset, devset, dataname, classes, None)
model.train(50)
print '****************************************************************************'
print 'test 1'
result = model.test()
print '****************************************************************************'
print '\n'
for i in xrange(1,10):/Users/is_bada/Downloads/Untitled1.md
	model.train(50)
	print '****************************************************************************'
	print 'test',i+1
	newresult=model.test()
	print '****************************************************************************'
	print '\n'
	if newresult[0]>result[0] :
		result=newresult
		model.save('../model/'+dataname+'/bestmodel')
print 'bestmodel saved!'


```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-2-a0b28bb3d52b> in <module>()
          2 
          3 import sys
    ----> 4 from Dataset import *
          5 from LSTMModel import LSTMModel
          6 


    ImportError: No module named Dataset


# DataSet.py


```python
#-*- coding: UTF-8 -*-  
import numpy
import copy
import theano
import random

def genBatch(data):
    m =0 
    maxsentencenum = len(data[0])
    for doc in data:
        for sentence in doc:
            if len(sentence)>m:
                m = len(sentence)
        for i in xrange(maxsentencenum - len(doc)):
            doc.append([-1])
    tmp = map(lambda doc: numpy.asarray(map(lambda sentence : sentence + [-1]*(m - len(sentence)), doc), dtype = numpy.int32).T, data)                          #[-1]是加在最前面
    tmp = reduce(lambda doc,docs : numpy.concatenate((doc,docs),axis = 1),tmp)
    return tmp 
            
def genLenBatch(lengths,maxsentencenum):
    lengths = map(lambda length : numpy.asarray(length + [1.0]*(maxsentencenum-len(length)), dtype = numpy.float32)+numpy.float32(1e-4),lengths)
    return reduce(lambda x,y : numpy.concatenate((x,y),axis = 0),lengths)

def genwordmask(docsbatch):
    mask = copy.deepcopy(docsbatch)
    mask = map(lambda x : map(lambda y : [1.0 ,0.0][y == -1],x), mask)
    mask = numpy.asarray(mask,dtype=numpy.float32)
    return mask

def gensentencemask(sentencenum):
    maxnum = sentencenum[0]
    mask = numpy.asarray(map(lambda num : [1.0]*num + [0.0]*(maxnum - num),sentencenum), dtype = numpy.float32)
    return mask.T

class Dataset(object):
    def __init__(self, filename, emb,maxbatch = 32,maxword = 500):
        lines = map(lambda x: x.split('\t\t'), open(filename).readlines())           
        label = numpy.asarray(
            map(lambda x: int(x[2])-1, lines),
            dtype = numpy.int32
        )
        docs = map(lambda x: x[3][0:len(x[3])-1], lines) 
        docs = map(lambda x: x.split('<sssss>'), docs) 
        docs = map(lambda doc: map(lambda sentence: sentence.split(' '),doc),docs)
        docs = map(lambda doc: map(lambda sentence: filter(lambda wordid: wordid !=-1,map(lambda word: emb.getID(word),sentence)),doc),docs)
        tmp = zip(docs, label)
        #random.shuffle(tmp)
        tmp.sort(lambda x, y: len(y[0]) - len(x[0]))  
        docs, label = zip(*tmp)

        sentencenum = map(lambda x : len(x),docs)
        length = map(lambda doc : map(lambda sentence : len(sentence), doc), docs)
        self.epoch = len(docs) / maxbatch                                      
        if len(docs) % maxbatch != 0:
            self.epoch += 1
        
        self.docs = []
        self.label = []
        self.length = []
        self.sentencenum = []
        self.wordmask = []
        self.sentencemask = []
        self.maxsentencenum = []

        for i in xrange(self.epoch):
            self.maxsentencenum.append(sentencenum[i*maxbatch])
            self.length.append(genLenBatch(length[i*maxbatch:(i+1)*maxbatch],sentencenum[i*maxbatch])) 
            docsbatch = genBatch(docs[i*maxbatch:(i+1)*maxbatch])
            self.docs.append(docsbatch)
            self.label.append(numpy.asarray(label[i*maxbatch:(i+1)*maxbatch], dtype = numpy.int32))
            self.sentencenum.append(numpy.asarray(sentencenum[i*maxbatch:(i+1)*maxbatch],dtype = numpy.float32)+numpy.float32(1e-4))
            self.wordmask.append(genwordmask(docsbatch))
            self.sentencemask.append(gensentencemask(sentencenum[i*maxbatch:(i+1)*maxbatch]))
        

class Wordlist(object):
    def __init__(self, filename, maxn = 100000):
        lines = map(lambda x: x.split(), open(filename).readlines()[:maxn])
        self.size = len(lines)

        self.voc = [(item[0][0], item[1]) for item in zip(lines, xrange(self.size))]
        self.voc = dict(self.voc)

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return -1

```

# Update.py


```python
#-*- coding: UTF-8 -*-  
import numpy as np
import theano
import theano.tensor as T

def AdaUpdates(parameters, gradients, rho, eps):
	rho = np.float32(rho)
	eps = np.float32(eps)
	
	gradients_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=np.float32), borrow=True) for p in parameters ]
	deltas_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=np.float32), borrow=True) for p in parameters ]

	gradients_sq_new = [ rho*g_sq + (np.float32(1)-rho)*(g*g) for g_sq,g in zip(gradients_sq,gradients) ]
	deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in zip(deltas_sq,gradients_sq_new,gradients) ]

	deltas_sq_new = [ rho*d_sq + (np.float32(1)-rho)*(d*d) for d_sq,d in zip(deltas_sq,deltas) ]

	gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
	deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
	parameters_updates = [ (p,p - d) for p,d in zip(parameters,deltas) ]
	return gradient_sq_updates + deltas_sq_updates + parameters_updates
```

# LSTMModel.py


```python
#-*- coding: UTF-8 -*-  
from EmbLayer import EmbLayer
from LSTMLayer import LSTMLayer
from HiddenLayer import HiddenLayer
from Update import AdaUpdates
from PoolLayer import *
from SentenceSortLayer import *
import theano
import theano.tensor as T
import numpy
import random
import sys

class LSTMModel(object):
    def __init__(self, n_voc, trainset, testset,dataname, classes, prefix):
        if prefix != None:
            prefix += '/'
        self.trainset = trainset
        self.testset = testset

        docs = T.imatrix()
        label = T.ivector()
        length = T.fvector()
        sentencenum = T.fvector()
        wordmask = T.fmatrix()
        sentencemask = T.fmatrix()
        maxsentencenum = T.iscalar()
        isTrain = T.iscalar()

        rng = numpy.random

        layers = []
        layers.append(EmbLayer(rng, docs, n_voc, 200, 'emblayer', dataname, prefix))
        layers.append(LSTMLayer(rng, layers[-1].output, wordmask, 200, 200, 'wordlstmlayer', prefix)) 
        layers.append(MeanPoolLayer(layers[-1].output, length))
        layers.append(SentenceSortLayer(layers[-1].output,maxsentencenum))
        layers.append(LSTMLayer(rng, layers[-1].output, sentencemask, 200, 200, 'sentencelstmlayer', prefix))
        layers.append(MeanPoolLayer(layers[-1].output, sentencenum))
        layers.append(HiddenLayer(rng, layers[-1].output, 200, 200, 'fulllayer', prefix))
        layers.append(HiddenLayer(rng, layers[-1].output, 200, int(classes), 'softmaxlayer', prefix, activation=T.nnet.softmax))
        self.layers = layers
        
        cost = -T.mean(T.log(layers[-1].output)[T.arange(label.shape[0]), label], acc_dtype='float32')
        correct = T.sum(T.eq(T.argmax(layers[-1].output, axis=1), label), acc_dtype='int32')
        err = T.argmax(layers[-1].output, axis=1) - label
        mse = T.sum(err * err)
        
        params = []
        for layer in layers:
            params += layer.params
        L2_rate = numpy.float32(1e-5)
        for param in params[1:]:
            cost += T.sum(L2_rate * (param * param), acc_dtype='float32')
        gparams = [T.grad(cost, param) for param in params]

        updates = AdaUpdates(params, gparams, 0.95, 1e-6)

        self.train_model = theano.function(
            inputs=[docs, label,length,sentencenum,wordmask,sentencemask,maxsentencenum],
            outputs=cost,
            updates=updates,
        )

        self.test_model = theano.function(
            inputs=[docs, label,length,sentencenum,wordmask,sentencemask,maxsentencenum],
            outputs=[correct, mse],
        )

    def train(self, iters):
        lst = numpy.random.randint(self.trainset.epoch, size = iters)
        n = 0
        for i in lst:
            n += 1
            out = self.train_model(self.trainset.docs[i], self.trainset.label[i], self.trainset.length[i],self.trainset.sentencenum[i],self.trainset.wordmask[i],self.trainset.sentencemask[i],self.trainset.maxsentencenum[i])
            print n, 'cost:',out
        
    def test(self):
        cor = 0
        tot = 0
        mis = 0
        for i in xrange(self.testset.epoch):
            tmp = self.test_model(self.testset.docs[i], self.testset.label[i], self.testset.length[i],self.testset.sentencenum[i],self.testset.wordmask[i],self.testset.sentencemask[i],self.testset.maxsentencenum[i])
            cor += tmp[0]
            mis += tmp[1]
            tot += len(self.testset.label[i])
        print 'Accuracy:',float(cor)/float(tot),'RMSE:',numpy.sqrt(float(mis)/float(tot))
        return cor, mis, tot


    def save(self, prefix):
        prefix += '/'
        for layer in self.layers:
            layer.save(prefix)

```

# EMBLayer.py


```python
#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

class EmbLayer(object):
    def __init__(self, rng, inp, n_voc, dim, name, dataname,prefix=None):
        self.input = inp
        self.name = name

        if prefix == None:
            f = file('../data/'+dataname+'/embinit.save', 'rb')
            W = cPickle.load(f)
            f.close()
            W = theano.shared(value=W, name='E', borrow=True)    
        else:
            f = file(prefix + name + '.save', 'rb')
            W = cPickle.load(f)
            f.close()
        self.W = W

        self.output = self.W[inp.flatten()].reshape((inp.shape[0], inp.shape[1], dim))
        self.params = [self.W]

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
```

# HiddenLayer.py


```python
#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, name, prefix=None,
                 activation=T.tanh):
        self.name = name
        self.input = input

        if prefix is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=numpy.float32
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            f = file(prefix + name + '.save', 'rb')
            W = cPickle.load(f)
            b = cPickle.load(f)
            f.close()

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
```

# LSTMLayer.py


```python
#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import cPickle

def randMatrix(rng, shape, lim):
    return numpy.asarray(
        rng.uniform(
            low=-lim,
            high=lim,
            size=shape
        ),
        dtype=numpy.float32
    )

class LSTMLayer(object):
    def __init__(self, rng, input, mask, n_in, n_out, name, prefix=None):
        self.input = input
        self.name = name

        limV = numpy.sqrt(6. / (n_in + n_out * 2))
        limG = limV * 4

        if prefix is None:
            Wi1_values = randMatrix(rng, (n_in, n_out), limG)
            Wi1 = theano.shared(value=Wi1_values, name='Wi1', borrow=True)
            Wi2_values = randMatrix(rng, (n_out, n_out), limG)
            Wi2 = theano.shared(value=Wi2_values, name='Wi2', borrow=True)
            bi_values = numpy.zeros((n_out,), dtype=numpy.float32)
            bi = theano.shared(value=bi_values, name='bi', borrow=True)

            Wo1_values = randMatrix(rng, (n_in, n_out), limG)
            Wo1 = theano.shared(value=Wo1_values, name='Wo1', borrow=True)
            Wo2_values = randMatrix(rng, (n_out, n_out), limG)
            Wo2 = theano.shared(value=Wo2_values, name='Wo2', borrow=True)
            bo_values = numpy.zeros((n_out,), dtype=numpy.float32)
            bo = theano.shared(value=bo_values, name='bo', borrow=True)

            Wf1_values = randMatrix(rng, (n_in, n_out), limG)
            Wf1 = theano.shared(value=Wf1_values, name='Wf1', borrow=True)
            Wf2_values = randMatrix(rng, (n_out, n_out), limG)
            Wf2 = theano.shared(value=Wf2_values, name='Wf2', borrow=True)
            bf_values = numpy.zeros((n_out,), dtype=numpy.float32)
            bf = theano.shared(value=bf_values, name='bf', borrow=True)

            Wc1_values = randMatrix(rng, (n_in, n_out), limV)
            Wc1 = theano.shared(value=Wc1_values, name='Wc1', borrow=True)
            Wc2_values = randMatrix(rng, (n_out, n_out), limV)
            Wc2 = theano.shared(value=Wc2_values, name='Wc2', borrow=True)
            bc_values = numpy.zeros((n_out,), dtype=numpy.float32)
            bc = theano.shared(value=bc_values, name='bc', borrow=True)

        else:
            f = file(prefix + name + '.save', 'rb')
            Wi1 = cPickle.load(f)
            Wi2 = cPickle.load(f)
            bi = cPickle.load(f)

            Wo1 = cPickle.load(f)
            Wo2 = cPickle.load(f)
            bo = cPickle.load(f)

            Wf1 = cPickle.load(f)
            Wf2 = cPickle.load(f)
            bf = cPickle.load(f)

            Wc1 = cPickle.load(f)
            Wc2 = cPickle.load(f)
            bc = cPickle.load(f)

            f.close()

        self.Wi1 = Wi1
        self.Wi2 = Wi2
        self.bi = bi

        self.Wo1 = Wo1
        self.Wo2 = Wo2
        self.bo = bo

        self.Wf1 = Wf1
        self.Wf2 = Wf2
        self.bf = bf

        self.Wc1 = Wc1
        self.Wc2 = Wc2
        self.bc = bc

        def step(emb, mask, C, prev):
            Gi = T.nnet.sigmoid(T.dot(emb, self.Wi1) + T.dot(prev, self.Wi2) + self.bi)
            Go = T.nnet.sigmoid(T.dot(emb, self.Wo1) + T.dot(prev, self.Wo2) + self.bo)
            Gf = T.nnet.sigmoid(T.dot(emb, self.Wf1) + T.dot(prev, self.Wf2) + self.bf)
            Ct = T.tanh(T.dot(emb, self.Wc1) + T.dot(prev, self.Wc2) + self.bc)

            CC = C * Gf + Ct * Gi
            CC = CC * mask.dimshuffle(0,'x') 
            CC = T.cast(CC,'float32')
            h = T.tanh(CC) * Go
            h = h * mask.dimshuffle(0,'x') 
            h = T.cast(h,'float32')
            return [CC, h]

        outs, _ = theano.scan(fn=step,
            outputs_info=[T.zeros_like(T.dot(input[0], self.Wi1)), T.zeros_like(T.dot(input[0], self.Wi1))],
            sequences=[input, mask])

        self.output = outs[1]

        self.params = [self.Wi1, self.Wi2, self.bi, self.Wo1, self.Wo2, self.bo,
            self.Wf1, self.Wf2, self.bf, self.Wc1, self.Wc2, self.bc]

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
```

# PoolLayer.py


```python
#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy

class LastPoolLayer(object):
    def __init__(self, input):
        self.input = input
        self.output = input[-1]
        self.params = []

    def save(self, prefix):
        pass

class MeanPoolLayer(object):
    def __init__(self, input, ll):
        self.input = input
        self.output = T.sum(input, axis=0, acc_dtype='float32') / ll.dimshuffle(0, 'x')          
        self.params = []

    def save(self, prefix):
        pass


class MaxPoolLayer(object):
    def __init__(self, input):
        self.input = input
        self.output = T.max(input, axis = 0)
        self.params = []

    def save(self, prefix):
        pass

class Dropout(object):
    def __init__(self, input, rate, istrain):
        rate = numpy.float32(rate)
        self.input = input
        srng = T.shared_randomstreams.RandomStreams()
        mask = srng.binomial(n=1, p=numpy.float32(1-rate), size=input.shape, dtype='float32')
        self.output = T.switch(istrain, mask*self.input, self.input*numpy.float32(1-rate))
        self.params = []

    def save(self, prefix):
        pass
```

# SentenceSortLayer.py


```python
#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy

class SentenceSortLayer(object):
    def __init__(self, input,maxsentencenum):
        self.input = input
        [sentencelen,emblen] = T.shape(input)
        output = input.reshape((sentencelen / maxsentencenum,maxsentencenum,emblen))
        output = output.dimshuffle(1,0,2)
        self.output = output
        self.params = []
        

    def save(self, prefix):
        pass
```
