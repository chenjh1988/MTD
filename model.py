# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import pdb


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable('beta', initializer=tf.zeros(params_shape))
        gamma = tf.get_variable('gamma', initializer=tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def position_embedding(max_len, d_emb):
    pos_emb = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0 else np.zeros(d_emb) for pos in range(max_len)]
    )
    pos_emb[1:, 0::2] = np.sin(pos_emb[1:, 0::2])  # dim 2i
    pos_emb[1:, 1::2] = np.cos(pos_emb[1:, 1::2])  # dim 2i+1
    return pos_emb


def feedforward(inputs,
                num_units=[2048, 512],
                scope="feedforward",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs, scope=scope)
    return outputs


def multihead_attention(queries,
                        queries_length,
                        keys,
                        keys_length,
                        weight_init=None,
                        num_units=None,
                        num_heads=4,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None):
    if num_units is None:
        num_units = queries.get_shape().as_list[-1]

    with tf.variable_scope(scope, reuse=reuse):
        # Linear projections, C = # dim or column, T_x = # vectors or actions
        Q = tf.layers.dense(queries, num_units, kernel_initializer=weight_init, use_bias=False)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, kernel_initializer=weight_init, use_bias=False)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, kernel_initializer=weight_init, use_bias=False)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        # query-key score matrix
        # each big score matrix is then split into h score matrix with same size
        # w.r.t. different part of the feature
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

        # Causality, Future blinding
        if True:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)
        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sequence_mask(queries_length, tf.shape(queries)[1], dtype=tf.float32)   # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (h*N, T_q, T_k)
        # Attention vector
        att_vec = outputs
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries
    # Normalize
    outputs = normalize(outputs, scope=scope)  # (N, T_q, C)
    return outputs, att_vec


def ggnn(adj_in, adj_out, embedding, items, out_size, weight_init, step=1):
    batch_size = tf.shape(items)[0]
    W_in = tf.get_variable('W_in', [out_size, out_size], dtype=tf.float32, initializer=weight_init)
    b_in = tf.get_variable('b_in', [out_size], dtype=tf.float32, initializer=weight_init)
    W_out = tf.get_variable('W_out', [out_size, out_size], dtype=tf.float32, initializer=weight_init)
    b_out = tf.get_variable('b_out', [out_size], dtype=tf.float32, initializer=weight_init)

    fin_state = tf.nn.embedding_lookup(embedding, items)
    cell = tf.nn.rnn_cell.GRUCell(out_size)
    with tf.variable_scope('gru'):
        for i in range(step):
            fin_state = tf.reshape(fin_state, [-1, out_size])
            fin_state_in = tf.reshape(tf.matmul(fin_state, W_in) + b_in, [batch_size, -1, out_size])
            fin_state_out = tf.reshape(tf.matmul(fin_state, W_out) + b_out, [batch_size, -1, out_size])
            av = tf.concat([tf.matmul(adj_in, fin_state_in), tf.matmul(adj_out, fin_state_out)], axis=-1)
            state_output, fin_state = tf.nn.dynamic_rnn(
                cell, tf.expand_dims(tf.reshape(av, [-1, 2 * out_size]), axis=1),
                initial_state=tf.reshape(fin_state, [-1, out_size])
            )
    return tf.reshape(fin_state, [batch_size, -1, out_size])


def session_rec(re_embedding, embedding, alias, targets, mask, out_size, weight_init):
    batch_size = tf.shape(alias)[0]
    nasr_w1 = tf.get_variable('nasr_w1', [out_size, out_size], dtype=tf.float32, initializer=weight_init)
    nasr_w2 = tf.get_variable('nasr_w2', [out_size, out_size], dtype=tf.float32, initializer=weight_init)
    nasr_v = tf.get_variable('nasrv', [1, out_size], dtype=tf.float32, initializer=weight_init)
    nasr_b = tf.get_variable('nasr_b', [out_size], dtype=tf.float32, initializer=tf.zeros_initializer())
    B = tf.get_variable('B', [2 * out_size, out_size], initializer=weight_init)

    rm = tf.reduce_sum(mask, 1)
    last_id = tf.gather_nd(alias, tf.stack([tf.range(batch_size), tf.to_int32(rm) - 1], axis=1))
    last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(batch_size), last_id], axis=1))
    # seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], alias[i]) for i in tf.range(batch_size)], axis=0)
    # pdb.set_trace()
    seq_h = tf.map_fn(lambda x: tf.nn.embedding_lookup(re_embedding[x], alias[x]), tf.range(batch_size), dtype=tf.float32)
    last = tf.matmul(last_h, nasr_w1)
    seq = tf.matmul(tf.reshape(seq_h, [-1, out_size]), nasr_w2)
    last = tf.reshape(last, [batch_size, 1, -1])
    m = tf.nn.sigmoid(last + tf.reshape(seq, [batch_size, -1, out_size]) + nasr_b)
    coef = tf.matmul(tf.reshape(m, [-1, out_size]), nasr_v, transpose_b=True) * tf.reshape(mask, [-1, 1])
    coef = tf.reshape(coef, [batch_size, -1, 1])

    b = embedding[1:]
    ma = tf.concat([tf.reduce_sum(coef * seq_h, 1), tf.reshape(last_h, [-1, out_size])], -1)
    y1 = tf.matmul(ma, B)
    logits = tf.matmul(y1, b, transpose_b=True)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets - 1, logits=logits))
    return loss, logits, seq_h


def sa(embedding, inputs, labels, seq_len, hidden_size, emb_size, weight_init,
       pos_table, pos, dropout_rate, is_train, num_head, num_block, nonhybrid):
    batch_size = tf.shape(inputs)[0]
    W_encoder = tf.get_variable('W_encoder', [hidden_size, hidden_size], dtype=tf.float32, initializer=weight_init)
    W_decoder = tf.get_variable('W_decoder', [hidden_size, hidden_size], dtype=tf.float32, initializer=weight_init)
    bl_vector = tf.get_variable('bl_vector', [hidden_size, 1], dtype=tf.float32, initializer=weight_init)

    inputs = tf.nn.embedding_lookup(embedding, inputs)
    pos_emb = tf.nn.embedding_lookup(pos_table, pos)
    inputs += pos_emb
    inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=is_train)

    for i in range(num_block):
        inputs, satt = multihead_attention(inputs, seq_len, inputs, seq_len,
                                           weight_init=weight_init,
                                           num_units=hidden_size,
                                           num_heads=num_head,
                                           dropout_rate=dropout_rate,
                                           is_training=is_train,
                                           scope='block_%d' % i)
        inputs = feedforward(inputs, num_units=[hidden_size, hidden_size], scope='feed_%d' % i,
                             dropout_rate=dropout_rate, is_training=is_train)

    seq_len = tf.cast(seq_len, dtype=tf.int32)
    target = tf.range(tf.shape(seq_len)[0])
    target = tf.stack([target, seq_len - 1], axis=1)
    last = tf.gather_nd(inputs, target)

    if not nonhybrid:
        bili = tf.get_variable('bili', [2 * hidden_size, emb_size], dtype=tf.float32, initializer=weight_init)
        last_ = tf.matmul(last, W_encoder)
        input_ = tf.reshape(inputs, [-1, hidden_size])
        input_ = tf.reshape(tf.matmul(input_, W_decoder), [batch_size, -1, hidden_size])
        att = tf.keras.activations.hard_sigmoid(tf.expand_dims(last_, 1) + input_)
        att = tf.reshape(att, [-1, hidden_size])
        att = tf.reshape(tf.matmul(att, bl_vector), [batch_size, -1])

        mask = tf.sequence_mask(seq_len, dtype=tf.float32)
        att = att * mask
        paddings = tf.ones_like(att) * (-2 ** 32 + 1)
        att = tf.where(tf.equal(att, 0), paddings, att)
        att = tf.nn.softmax(att, dim=1)

        output = tf.reduce_sum(inputs * tf.expand_dims(att, -1), axis=1)
        output = tf.concat([last, output], axis=1)
        output = tf.layers.dropout(output, rate=dropout_rate, training=is_train)
        output = tf.matmul(output, bili)
    else:
        bili = tf.get_variable('bili', [hidden_size, emb_size], dtype=tf.float32, initializer=weight_init)
        output = last
        output = tf.layers.dropout(output, rate=dropout_rate, training=is_train)
        output = tf.matmul(output, bili)

    logits = tf.matmul(output, embedding[1:, :], transpose_b=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels - 1)
    loss = tf.reduce_mean(loss)
    return loss, logits


def gcn(seq, adj, n_h, use_bias=False, num_gcn=1, scope='gcn', reuse=False):
    out = seq
    for i in range(num_gcn):
        with tf.variable_scope(scope + '_' + str(i), reuse=reuse):
            seq_fts = tf.layers.dense(out, n_h, use_bias=False, reuse=tf.AUTO_REUSE)
            out = tf.sparse_tensor_dense_matmul(adj, tf.squeeze(seq_fts, axis=0))
            if use_bias:
                bias = tf.get_variable('gcn_bias', shape=[1, n_h], dtype=tf.float32)
                out += bias
            out = normalize(out, scope='gcn_normal')
            out = tf.nn.tanh(out)
            out = tf.expand_dims(out, axis=0)
    return out


def dgi(seq1, seq2, adj, n_in, n_h, use_bias, num_gcn):
    h_1 = gcn(seq1, adj, n_h, use_bias=use_bias, num_gcn=num_gcn)
    c = tf.sigmoid(tf.reduce_mean(h_1, axis=1))
    h_2 = gcn(seq2, adj, n_h, use_bias=use_bias, num_gcn=num_gcn, reuse=True)

    n_m = tf.shape(h_1)[1]
    c_x = tf.expand_dims(c, axis=1)
    c_x = tf.tile(c_x, [1, n_m, 1])
    A = tf.get_variable('dgi_bilinear', shape=[1, n_h, n_h], dtype=tf.float32)

    sc = tf.matmul(c_x, A)
    sc_1 = tf.reduce_sum(sc * h_1, -1)
    sc_2 = tf.reduce_sum(sc * h_2, -1)
    logits = tf.concat([sc_1, sc_2], 1)
    return logits, h_1


class MTD(object):
    def __init__(self, hidden_size=100,
                 emb_size=100,
                 n_node=None,
                 lr=None,
                 l2=None,
                 decay=None,
                 lr_dc=0.1,
                 kg=0,
                 dropout=0.3,
                 method='sa',
                 num_head=1,
                 num_block=1,
                 nonhybrid=False,
                 num_gcn=1,
                 use_bias=False):
        self.n_node = n_node
        self.l2 = l2
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight_init = tf.random_uniform_initializer(-self.stdv, self.stdv)

        self.dropout_rate = dropout
        self.num_head = num_head
        self.num_block = num_block
        self.nonhybrid = nonhybrid
        self.num_gcn = num_gcn
        self.use_bias = use_bias

        self.pre_embedding = tf.get_variable(
            shape=[n_node, emb_size], name='embedding', dtype=tf.float32, initializer=self.weight_init)
        self.embedding = tf.concat((tf.zeros(shape=[1, emb_size]), self.pre_embedding[1:, :]), 0)

        if method == 'ggnn':
            self.learning_rate = tf.train.exponential_decay(
                lr, global_step=self.global_step, decay_steps=decay, decay_rate=lr_dc, staircase=True)
            self.build_sr_gnn()
            self.run_rec = self.run_sr_gnn
        elif method == 'sa':
            self.learning_rate = tf.train.exponential_decay(
                lr, global_step=self.global_step, decay_steps=decay, decay_rate=lr_dc, staircase=True)
            self.build_sa()
            self.run_rec = self.run_sa

        if kg > 0:
            self.build_dgi()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def build_sr_gnn(self):
        self.mask = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.alias = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.items = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None])
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[None, None, None])

        self.re_embed = ggnn(self.adj_in,
                             self.adj_out,
                             self.embedding,
                             self.items,
                             self.hidden_size,
                             self.weight_init)
        loss, self.logits, self.seq_h = session_rec(self.re_embed,
                                                    self.embedding,
                                                    self.alias,
                                                    self.targets,
                                                    self.mask,
                                                    self.hidden_size,
                                                    self.weight_init)

        params = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in params if 'kg' not in v.name]) * self.l2
        _, self.top_k = tf.nn.top_k(self.logits, k=60)
        self.rec_loss = loss + lossL2
        self.rec_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.rec_loss, global_step=self.global_step)

    def build_sa(self):
        self.seq_len = tf.placeholder(dtype=tf.int32, shape=[None])
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None])
        self.pos = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.is_train = tf.placeholder(dtype=tf.bool, shape=[])
        self.pos_table = tf.get_variable(
            shape=[20, self.emb_size], name='pos_emb', dtype=tf.float32, initializer=self.weight_init)
        self.pos_table = tf.concat((tf.zeros(shape=[1, self.emb_size]), self.pos_table[1:, :]), 0)
        loss, self.logits = sa(self.embedding,
                               self.inputs,
                               self.targets,
                               self.seq_len,
                               self.hidden_size,
                               self.emb_size,
                               self.weight_init,
                               self.pos_table,
                               self.pos,
                               self.dropout_rate,
                               self.is_train,
                               self.num_head,
                               self.num_block,
                               self.nonhybrid)
        params = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in params if 'kg' not in v.name]) * self.l2
        _, self.top_k = tf.nn.top_k(self.logits, k=60)
        self.rec_loss = loss + lossL2
        self.rec_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.rec_loss, global_step=self.global_step)

    def build_dgi(self):
        self.x1 = tf.placeholder(dtype=tf.int64, shape=[1, None])
        self.x2 = tf.placeholder(dtype=tf.int64, shape=[1, None])
        self.adj = tf.sparse_placeholder(dtype=tf.float32)
        seq1 = tf.nn.embedding_lookup(self.embedding, self.x1)
        seq2 = tf.nn.embedding_lookup(self.embedding, self.x2)
        lbl_1 = tf.ones(shape=(1, tf.shape(self.x1)[1]))
        lbl_2 = tf.zeros(shape=(1, tf.shape(self.x2)[1]))
        lbl = tf.concat([lbl_1, lbl_2], axis=1)
        with tf.variable_scope('kg_dgi'):
            logit, hh = dgi(seq1, seq2, self.adj, self.emb_size, self.emb_size, self.use_bias, self.num_gcn)
        self.hh = hh
        params = list(filter(lambda v: 'dgi' in v.name, tf.trainable_variables()))
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in params if 'dgi' in v.name]) * 0.000001
        self.dgi_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=lbl, logits=logit)
        self.dgi_loss = tf.reduce_mean(self.dgi_loss) + lossL2
        self.global_step_dgi = tf.Variable(0, trainable=False, name='global_step_dgi')
        self.dgi_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.dgi_loss, global_step=self.global_step_dgi, var_list=params)

    def run_sr_gnn(self, fetches, batch_input, is_train=True):
        feed_dict = {self.adj_in: batch_input[0],
                     self.adj_out: batch_input[1],
                     self.alias: batch_input[2],
                     self.items: batch_input[3],
                     self.mask: batch_input[4],
                     self.targets: batch_input[5]}
        return self.sess.run(fetches, feed_dict=feed_dict)

    def run_sa(self, fetches, batch_input, is_train=True):
        feed_dict = {self.inputs: batch_input[0],
                     self.seq_len: batch_input[1],
                     self.pos: batch_input[2],
                     self.targets: batch_input[3],
                     self.is_train: is_train}
        return self.sess.run(fetches, feed_dict=feed_dict)

    def run_dgi(self, fetches, x1, x2, adj):
        return self.sess.run(fetches, feed_dict={self.x1: x1, self.x2: x2, self.adj: adj})
