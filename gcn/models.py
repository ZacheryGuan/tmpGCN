from gcn.layers import DenseNet,  GraphConvolution_origin, GraphConvolution_case1, GraphConvolution_case2, GraphConvolution_case3, GraphConvolution_case4, Residual, FullyConnected, ConvolutionDenseNet
from gcn.metrics import masked_accuracy, masked_softmax_cross_entropy, weighted_softmax_cross_entropy,triplet_case1_softmax_cross_entropy, triplet_case2_softmax_cross_entropy,large_margin_softmax_cross_entropy, triplet_case3_softmax_cross_entropy
import tensorflow as tf
import numpy as np
from copy import copy
import random

class GCN_MLP_origin(object):
    def __init__(self, model_config, placeholders, input_dim):
        self.model_config = model_config
        self.name = model_config['name']
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self.logging = True if self.model_config['logdir'] else False

        self.vars = {}
        self.layers = []
        self.activations = []
        self.act = tf.nn.relu

        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.inputs._my_input_dim = input_dim
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1] if model_config['Model'] not in [11, 13, 14, 15] else \
        placeholders['label_per_sample'].get_shape().as_list()[1]
        self.outputs = None

        self.global_step = None
        self.loss = 0
        
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.summary = None
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.model_config['learning_rate'],
            # beta2=0.90,
        )

        self.build()
        return

    def build(self):
        layer_type = list(map(
            lambda x: {'c': GraphConvolution_origin, 'd': DenseNet, 'r':Residual, 'f': FullyConnected, 'C': ConvolutionDenseNet}.get(x),
            self.model_config['connection']))
        layer_size = copy(self.model_config['layer_size'])
        layer_size.insert(0, self.input_dim)
        layer_size.append(self.output_dim)
        print('layer_size', layer_size)
        sparse = True
        with tf.name_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.activations.append(self.inputs)
            for i, (output_dim, layer_cls) in enumerate(zip(layer_size[1:], layer_type)):
                # create Variables
                relu_flag = True
                if i==len(layer_size)-2:
                    print('------')
                    assert output_dim == self.output_dim

                    relu_flag = False
                    
                self.layers.append(layer_cls(input=self.activations[-1],
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=self.act,
                                             dropout=True,
                                             sparse_inputs=sparse,
                                             logging=self.logging,
                                             use_theta= self.model_config['conv'] == 'chebytheta', relu_flag = relu_flag))
                sparse = False
                # Build sequential layer model
                if relu_flag==False:                 
                    hidden, self.return_without_w1 = self.layers[-1]()
                else:
                    hidden = self.layers[-1]()
                self.activations.append(hidden)
                

            self.outputs  = self.activations[-1]
                
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars.update({var.op.name: var for var in variables})

        # Build metrics
        with tf.name_scope('predict'):
            self._predict()
        with tf.name_scope('loss'):
            self._loss()
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('accuracy'):
            self._accuracy()
            self._accuracy_of_class()
        tf.summary.scalar('accuracy', self.accuracy)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

    def _predict(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.prediction = tf.nn.softmax(unsoftmaxed)
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.prediction = tf.nn.softmax(unsoftmaxed)

        else:
            self.prediction = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss += self.model_config['weight_decay'] * tf.nn.l2_loss(var)
        # Cross entropy error
        if self.model_config['Model'] in [11, 13]:
            assert(self.model_config['Model11'] in ['nearest', 'weighted'])
            self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['label_per_sample'],
                                                      self.placeholders['labels_mask'])
        elif self.model_config['Model'] in [14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                keep_dims=True)
            self.loss += masked_softmax_cross_entropy(unsoftmaxed, self.placeholders['labels'],
                                                        self.placeholders['labels_mask'])
        elif self.model_config['loss_func']=='large_margin':
            
            self.loss += large_margin_softmax_cross_entropy(self.return_without_w1, self.w1, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'], self.model_config['M_margin'])
            self.loss1=self.loss
            self.loss2=self.loss-self.loss
        else:
            
            self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])
            self.loss1 = self.loss
            self.loss2 = self.loss-self.loss
        # Laplacian regularization
        if self.model_config['lambda'] != 0:
            self._laplacian_regularization(tf.nn.softmax(self.outputs))
            self.loss += self.model_config['lambda'] * self.lapla_reg

    def _laplacian_regularization(self, graph_signals):
        self.lapla_reg = tf.sparse_tensor_dense_matmul(self.placeholders['laplacian'], graph_signals)
        self.lapla_reg = tf.matmul(tf.transpose(graph_signals), self.lapla_reg)
        self.lapla_reg = tf.trace(self.lapla_reg)

    def _accuracy(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            else:
                raise ValueError("model_config['Model11'] should be either 'weighted' or 'nearest'")
        else:
            self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                            self.placeholders['labels_mask'])

    def _accuracy_of_class(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0, keep_dims=True)
            self.accuracy_of_class = [masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]
        else:
            self.accuracy_of_class = [masked_accuracy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
        
class GCN_MLP_case4(object):
    def __init__(self, model_config, placeholders, input_dim):
        self.model_config = model_config
        self.name = model_config['name']
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self.logging = True if self.model_config['logdir'] else False

        self.vars = {}
        self.layers = []
        self.activations = []
        self.act = tf.nn.relu

        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.inputs._my_input_dim = input_dim
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1] if model_config['Model'] not in [11, 13, 14, 15] else \
        placeholders['label_per_sample'].get_shape().as_list()[1]
        self.outputs = None

        self.global_step = None
        self.loss = 0
        
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.summary = None
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.model_config['learning_rate'],
            # beta2=0.90,
        )

        self.build()
        return

    def build(self):
        layer_type = list(map(
            lambda x: {'c': GraphConvolution_case4, 'd': DenseNet, 'r':Residual, 'f': FullyConnected, 'C': ConvolutionDenseNet}.get(x),
            self.model_config['connection']))
        layer_size = copy(self.model_config['layer_size'])
        layer_size.insert(0, self.input_dim)
        layer_size.append(self.output_dim)
        print('layer_size', layer_size)
        sparse = True
        with tf.name_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
#            self.activations.append((self.inputs, self.inputs))
            self.activations.append(self.inputs)
            for i, (output_dim, layer_cls) in enumerate(zip(layer_size[1:], layer_type)):
                # create Variables
                relu_flag = True
                if i==len(layer_size)-2:
                    print('------')
                    assert output_dim == self.output_dim

                    relu_flag = False
                    
                self.layers.append(layer_cls(input=self.activations[-1],
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=self.act,
                                             dropout=True,
                                             sparse_inputs=sparse,
                                             logging=self.logging,
                                             use_theta= self.model_config['conv'] == 'chebytheta', relu_flag = relu_flag))
                sparse = False
                # Build sequential layer model
                if relu_flag==False:                 
                    hidden, self.return_without_w1, self.w1 = self.layers[-1]()
                else:
                    hidden = self.layers[-1]()
                self.activations.append(hidden)
                

 #           self.outputs, self.output_previous = self.activations[-1]
            self.outputs  = self.activations[-1]
                
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars.update({var.op.name: var for var in variables})

        # Build metrics
        with tf.name_scope('predict'):
            self._predict()
        with tf.name_scope('loss'):
            self._loss()
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('accuracy'):
            self._accuracy()
            self._accuracy_of_class()
        tf.summary.scalar('accuracy', self.accuracy)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

    def _predict(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.prediction = tf.nn.softmax(unsoftmaxed)
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.prediction = tf.nn.softmax(unsoftmaxed)

        else:
            self.prediction = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss += self.model_config['weight_decay'] * tf.nn.l2_loss(var)
        # Cross entropy error
        if self.model_config['Model'] in [11, 13]:
            assert(self.model_config['Model11'] in ['nearest', 'weighted'])
            self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['label_per_sample'],
                                                      self.placeholders['labels_mask'])
        elif self.model_config['Model'] in [14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                keep_dims=True)
            self.loss += masked_softmax_cross_entropy(unsoftmaxed, self.placeholders['labels'],
                                                        self.placeholders['labels_mask'])
            
        elif self.model_config['loss_func']=='large_margin':
            
            self.loss += large_margin_softmax_cross_entropy(self.return_without_w1, self.w1, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'], self.model_config['M_margin'])
            self.loss1=self.loss
            self.loss2=self.loss-self.loss
        # Laplacian regularization
        if self.model_config['lambda'] != 0:
            self._laplacian_regularization(tf.nn.softmax(self.outputs))
            self.loss += self.model_config['lambda'] * self.lapla_reg

    def _laplacian_regularization(self, graph_signals):
        self.lapla_reg = tf.sparse_tensor_dense_matmul(self.placeholders['laplacian'], graph_signals)
        self.lapla_reg = tf.matmul(tf.transpose(graph_signals), self.lapla_reg)
        self.lapla_reg = tf.trace(self.lapla_reg)

    def _accuracy(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            else:
                raise ValueError("model_config['Model11'] should be either 'weighted' or 'nearest'")
        else:
            self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                            self.placeholders['labels_mask'])

    def _accuracy_of_class(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0, keep_dims=True)
            self.accuracy_of_class = [masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]
        else:
            self.accuracy_of_class = [masked_accuracy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
        
        
        
class GCN_MLP_case1(object):
    def __init__(self, model_config, placeholders, input_dim):
        self.model_config = model_config
        self.name = model_config['name']
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self.logging = True if self.model_config['logdir'] else False

        self.vars = {}
        self.layers = []
        self.activations = []
        self.act = tf.nn.relu

        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.inputs._my_input_dim = input_dim
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1] if model_config['Model'] not in [11, 13, 14, 15] else \
        placeholders['label_per_sample'].get_shape().as_list()[1]
        self.outputs = None

        self.global_step = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.summary = None
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.model_config['learning_rate'],
            # beta2=0.90,
        )

        self.build()
        return

    def build(self):
        layer_type = list(map(
            lambda x: {'c': GraphConvolution_case1, 'd': DenseNet, 'r':Residual, 'f': FullyConnected, 'C': ConvolutionDenseNet}.get(x),
            self.model_config['connection']))
        layer_size = copy(self.model_config['layer_size'])
        layer_size.insert(0, self.input_dim)
        layer_size.append(self.output_dim)
        print('layer_size', layer_size)
        sparse = True
        with tf.name_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.activations.append(self.inputs)
            for i, (output_dim, layer_cls) in enumerate(zip(layer_size[1:], layer_type)):
                # create Variables
                relu_flag = True
                if i==len(layer_size)-2:
                    print('------')
                    assert output_dim == self.output_dim

                    relu_flag = False
                    
                self.layers.append(layer_cls(input=self.activations[-1],
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=self.act,
                                             dropout=True,
                                             sparse_inputs=sparse,
                                             logging=self.logging,
                                             use_theta= self.model_config['conv'] == 'chebytheta', relu_flag = relu_flag))
                sparse = False
                # Build sequential layer model
                if relu_flag==False:                 
                    hidden, self.return_without_w1 = self.layers[-1]()
                else:
                    hidden = self.layers[-1]()
                self.activations.append(hidden)
                

            self.outputs  = self.activations[-1]
                
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars.update({var.op.name: var for var in variables})

        # Build metrics
        with tf.name_scope('predict'):
            self._predict()
        with tf.name_scope('loss'):
            self._loss()
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('accuracy'):
            self._accuracy()
            self._accuracy_of_class()
        tf.summary.scalar('accuracy', self.accuracy)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

    def _predict(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.prediction = tf.nn.softmax(unsoftmaxed)
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.prediction = tf.nn.softmax(unsoftmaxed)

        else:
            self.prediction = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss += self.model_config['weight_decay'] * tf.nn.l2_loss(var)
        # Cross entropy error
        if self.model_config['Model'] in [11, 13]:
            assert(self.model_config['Model11'] in ['nearest', 'weighted'])
            self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['label_per_sample'],
                                                      self.placeholders['labels_mask'])
        elif self.model_config['Model'] in [14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                keep_dims=True)
            self.loss += masked_softmax_cross_entropy(unsoftmaxed, self.placeholders['labels'],
                                                        self.placeholders['labels_mask'])
        else:
            if self.model_config['loss_func'] == 'imbalance':
                self.loss += weighted_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                            self.model_config['ws_beta'])
            elif self.model_config['loss_func'] == 'triplet': 
                if self.model_config['obj']=='case1':
                    self.loss_all, self.loss1, self.loss2 =  triplet_case1_softmax_cross_entropy(self.outputs, self.return_without_w1, self.placeholders['labels'], self.placeholders['triplet'], self.placeholders['labels_mask'], self.model_config['MARGIN'], self.model_config['triplet_lambda'])
                    self.loss = self.loss+self.loss_all
            else:
                self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])
        # Laplacian regularization
        if self.model_config['lambda'] != 0:
            self._laplacian_regularization(tf.nn.softmax(self.outputs))
            self.loss += self.model_config['lambda'] * self.lapla_reg

    def _laplacian_regularization(self, graph_signals):
        self.lapla_reg = tf.sparse_tensor_dense_matmul(self.placeholders['laplacian'], graph_signals)
        self.lapla_reg = tf.matmul(tf.transpose(graph_signals), self.lapla_reg)
        self.lapla_reg = tf.trace(self.lapla_reg)

    def _accuracy(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            else:
                raise ValueError("model_config['Model11'] should be either 'weighted' or 'nearest'")
        else:
            self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                            self.placeholders['labels_mask'])

    def _accuracy_of_class(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0, keep_dims=True)
            self.accuracy_of_class = [masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]
        else:
            self.accuracy_of_class = [masked_accuracy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)       
        
        
class GCN_MLP_case2(object):
    def __init__(self, model_config, placeholders, input_dim):
        self.model_config = model_config
        self.name = model_config['name']
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self.logging = True if self.model_config['logdir'] else False

        self.vars = {}
        self.layers = []
        self.activations = []
        self.act = tf.nn.relu

        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.inputs._my_input_dim = input_dim
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1] if model_config['Model'] not in [11, 13, 14, 15] else \
        placeholders['label_per_sample'].get_shape().as_list()[1]
        self.outputs = None

        self.global_step = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.summary = None
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.model_config['learning_rate'],
            # beta2=0.90,
        )

        self.build()
        return

    def build(self):
        layer_type = list(map(
            lambda x: {'c': GraphConvolution_case2, 'd': DenseNet, 'r':Residual, 'f': FullyConnected, 'C': ConvolutionDenseNet}.get(x),
            self.model_config['connection']))
        layer_size = copy(self.model_config['layer_size'])
        layer_size.insert(0, self.input_dim)
        layer_size.append(self.output_dim)
        print('layer_size', layer_size)
        sparse = True
        with tf.name_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.activations.append(self.inputs)
            for i, (output_dim, layer_cls) in enumerate(zip(layer_size[1:], layer_type)):
                # create Variables
                relu_flag = True
                if i==len(layer_size)-2:
                    print('------')
                    assert output_dim == self.output_dim

                    relu_flag = False
                    
                self.layers.append(layer_cls(input=self.activations[-1],
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=self.act,
                                             dropout=True,
                                             sparse_inputs=sparse,
                                             logging=self.logging,
                                             use_theta= self.model_config['conv'] == 'chebytheta', relu_flag = relu_flag))
                sparse = False
                # Build sequential layer model
                if relu_flag==False:                 
                    hidden, self.return_without_w1 = self.layers[-1]()
                else:
                    hidden = tf.nn.l2_normalize(self.layers[-1](),1)
                self.activations.append(hidden)
                

            self.outputs  = self.activations[-1]
                
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars.update({var.op.name: var for var in variables})

        # Build metrics
        with tf.name_scope('predict'):
            self._predict()
        with tf.name_scope('loss'):
            self._loss()
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('accuracy'):
            self._accuracy()
            self._accuracy_of_class()
        tf.summary.scalar('accuracy', self.accuracy)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

    def _predict(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.prediction = tf.nn.softmax(unsoftmaxed)
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.prediction = tf.nn.softmax(unsoftmaxed)

        else:
            self.prediction = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss += self.model_config['weight_decay'] * tf.nn.l2_loss(var)
        # Cross entropy error
        if self.model_config['Model'] in [11, 13]:
            assert(self.model_config['Model11'] in ['nearest', 'weighted'])
            self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['label_per_sample'],
                                                      self.placeholders['labels_mask'])
        elif self.model_config['Model'] in [14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                keep_dims=True)
            self.loss += masked_softmax_cross_entropy(unsoftmaxed, self.placeholders['labels'],
                                                        self.placeholders['labels_mask'])
        else:
            if self.model_config['loss_func'] == 'imbalance':
                self.loss += weighted_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                            self.model_config['ws_beta'])
            elif self.model_config['loss_func'] == 'triplet': 
                if self.model_config['obj']=='case2':
                    self.loss += triplet_case2_softmax_cross_entropy(self.outputs, self.return_without_w1, self.placeholders['labels'], self.placeholders['triplet'], self.placeholders['labels_mask'], self.model_config['MARGIN'], self.model_config['triplet_lambda'])
            else:
                self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])
        # Laplacian regularization
        if self.model_config['lambda'] != 0:
            self._laplacian_regularization(tf.nn.softmax(self.outputs))
            self.loss += self.model_config['lambda'] * self.lapla_reg

    def _laplacian_regularization(self, graph_signals):
        self.lapla_reg = tf.sparse_tensor_dense_matmul(self.placeholders['laplacian'], graph_signals)
        self.lapla_reg = tf.matmul(tf.transpose(graph_signals), self.lapla_reg)
        self.lapla_reg = tf.trace(self.lapla_reg)

    def _accuracy(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            else:
                raise ValueError("model_config['Model11'] should be either 'weighted' or 'nearest'")
        else:
            self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                            self.placeholders['labels_mask'])

    def _accuracy_of_class(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0, keep_dims=True)
            self.accuracy_of_class = [masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]
        else:
            self.accuracy_of_class = [masked_accuracy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN_MLP_case3(object):
    def __init__(self, model_config, placeholders, input_dim):
        self.model_config = model_config
        self.name = model_config['name']
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self.logging = True if self.model_config['logdir'] else False

        self.vars = {}
        self.layers = []
        self.activations = []
        self.act = tf.nn.relu

        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.inputs._my_input_dim = input_dim
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1] if model_config['Model'] not in [11, 13, 14, 15] else \
        placeholders['label_per_sample'].get_shape().as_list()[1]
        self.outputs = None

        self.global_step = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.summary = None
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.model_config['learning_rate'],
            # beta2=0.90,
        )

        self.build()
        return

    def build(self):
        layer_type = list(map(
            lambda x: {'c': GraphConvolution_case3, 'd': DenseNet, 'r':Residual, 'f': FullyConnected, 'C': ConvolutionDenseNet}.get(x),
            self.model_config['connection']))
        layer_size = copy(self.model_config['layer_size'])
        layer_size.insert(0, self.input_dim)
        layer_size.append(self.output_dim)
        print('layer_size', layer_size)
        sparse = True
        with tf.name_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.activations.append(self.inputs)
            for i, (output_dim, layer_cls) in enumerate(zip(layer_size[1:], layer_type)):
                # create Variables
                relu_flag = True
                if i==len(layer_size)-2:
                    print('------')
                    assert output_dim == self.output_dim

                    relu_flag = False
                    
                self.layers.append(layer_cls(input=self.activations[-1],
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=self.act,
                                             dropout=True,
                                             sparse_inputs=sparse,
                                             logging=self.logging,
                                             use_theta= self.model_config['conv'] == 'chebytheta', relu_flag = relu_flag, w1_gamma=self.model_config['w1_gamma']))
                sparse = False
                # Build sequential layer model
                if relu_flag==False:                 
                    hidden, self.return_without_w1 = self.layers[-1]()
                else:
                    hidden = tf.nn.l2_normalize(self.layers[-1](),1)
                self.activations.append(hidden)
                

            self.outputs  = self.activations[-1]
                
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars.update({var.op.name: var for var in variables})

        # Build metrics
        with tf.name_scope('predict'):
            self._predict()
        with tf.name_scope('loss'):
            self._loss()
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('accuracy'):
            self._accuracy()
            self._accuracy_of_class()
        tf.summary.scalar('accuracy', self.accuracy)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

    def _predict(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.prediction = tf.nn.softmax(unsoftmaxed)
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.prediction = tf.nn.softmax(unsoftmaxed)

        else:
            self.prediction = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss += self.model_config['weight_decay'] * tf.nn.l2_loss(var)
        # Cross entropy error
        if self.model_config['Model'] in [11, 13]:
            assert(self.model_config['Model11'] in ['nearest', 'weighted'])
            self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['label_per_sample'],
                                                      self.placeholders['labels_mask'])
        elif self.model_config['Model'] in [14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                keep_dims=True)
            self.loss += masked_softmax_cross_entropy(unsoftmaxed, self.placeholders['labels'],
                                                        self.placeholders['labels_mask'])
        else:
            if self.model_config['loss_func'] == 'imbalance':
                self.loss += weighted_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                            self.model_config['ws_beta'])
            elif self.model_config['loss_func'] == 'triplet': 
                if self.model_config['obj']=='case3':
                    self.loss += triplet_case3_softmax_cross_entropy(self.outputs, self.return_without_w1, self.placeholders['labels'], self.placeholders['triplet'], self.placeholders['labels_mask'], self.model_config['MARGIN'], self.model_config['triplet_lambda'])
            else:
                self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])
        # Laplacian regularization
        if self.model_config['lambda'] != 0:
            self._laplacian_regularization(tf.nn.softmax(self.outputs))
            self.loss += self.model_config['lambda'] * self.lapla_reg

    def _laplacian_regularization(self, graph_signals):
        self.lapla_reg = tf.sparse_tensor_dense_matmul(self.placeholders['laplacian'], graph_signals)
        self.lapla_reg = tf.matmul(tf.transpose(graph_signals), self.lapla_reg)
        self.lapla_reg = tf.trace(self.lapla_reg)

    def _accuracy(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            if self.model_config['Model'] in [14, 15] or self.model_config['Model11'] == 'weighted':
                sample2label = self.placeholders['sample2label']
                unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0,
                                                                                    keep_dims=True)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            elif self.model_config['Model11'] == 'nearest':
                sample2label = self.placeholders['sample2label']
                outputs = tf.one_hot(tf.argmax(self.outputs, axis=1),
                                     depth=self.placeholders['label_per_sample'].get_shape().as_list()[1])
                unsoftmaxed = tf.matmul(outputs, sample2label)
                self.accuracy = masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])
            else:
                raise ValueError("model_config['Model11'] should be either 'weighted' or 'nearest'")
        else:
            self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                            self.placeholders['labels_mask'])

    def _accuracy_of_class(self):
        if self.model_config['Model'] in [11, 13, 14, 15]:
            sample2label = self.placeholders['sample2label']
            unsoftmaxed = tf.matmul(self.outputs, sample2label) / tf.reduce_sum(sample2label, axis=0, keep_dims=True)
            self.accuracy_of_class = [masked_accuracy(unsoftmaxed, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]
        else:
            self.accuracy_of_class = [masked_accuracy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:,i])
                                      for i in range(self.placeholders['labels'].shape[1])]

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
