# define the SANs model version 1
import tensorflow as tf
import math
import numpy as np

class Vis_lstm_model:
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, options):
        with tf.device('/gpu:0'):
            self.options = options

            # +1 for zero padding
            self.Wemb = tf.Variable(tf.random_uniform([options['q_vocab_size'] + 1, options['embedding_size']], -1.0, 1.0), name = 'Wemb')
            self.Wimg = self.init_weight(options['cnn7_feature_length'], options['embedding_size'], name = 'Wimg')
            self.bimg = self.init_bias(options['embedding_size'], name = 'bimg')

            self.lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 2,
                                                        num_units = 512,
                                                        input_mode='linear_input',
                                                        direction='unidirectional',
                                                        dropout=0.5,
                                                        seed=0.25,
                                                        dtype=tf.float32,
                                                        kernel_initializer=None,
                                                        bias_initializer=None,
                                                        name='lstm'
                                                    ) 
            
            self.ans_sm_W = self.init_weight(options['rnn_size'], options['ans_vocab_size'], name = 'ans_sm_W')
            self.ans_sm_b = self.init_bias(options['ans_vocab_size'], name = 'ans_sm_b')

            self.lstm_W = []
            self.lstm_U = []
            self.lstm_b = []
            for i in range(options['num_lstm_layers']):
                W = self.init_weight(options['rnn_size'], 4 * options['rnn_size'], name = ('rnnw_' + str(i)))
                U = self.init_weight(options['rnn_size'], 4 * options['rnn_size'], name = ('rnnu_' + str(i)))
                b = self.init_bias(4 * options['rnn_size'], name = ('rnnb_' + str(i)))
                self.lstm_W.append(W)
                self.lstm_U.append(U)
                self.lstm_b.append(b)

            self.W_p_1 = tf.Variable(tf.truncated_normal([options['embedding_size'], 1],\
                                       stddev=1.0/math.sqrt(float(options['embedding_size']))), name='W_p_1')
            self.b_p_1 = tf.Variable(tf.zeros([49,1]), name='b_p_1')

            self.W_p_2 = tf.Variable(tf.truncated_normal([options['embedding_size'], 1],\
                                       stddev=1.0/math.sqrt(float(options['embedding_size']))), name='W_p_2')
            self.b_p_2 = tf.Variable(tf.zeros([49,1]), name='b_p_2')

            self.W_u = tf.Variable(tf.truncated_normal([options['embedding_size'], options['ans_vocab_size']],stddev=1.0/math.sqrt(float(options['embedding_size']))), name='W_u')
            self.b_u = tf.Variable(tf.zeros([options['ans_vocab_size'],1]), name='b_u')

            self.W_i_0 = tf.Variable(tf.truncated_normal([options['cnn7_feature_length'], options['embedding_size']],\
                                     stddev=1.0/math.sqrt(float(options['cnn7_feature_length']))), name='W_i_0')
            self.b_i_0 = tf.Variable(tf.zeros([options['embedding_size'],1]), name='b_i_0')

            self.W_q_1 = tf.Variable(tf.truncated_normal([options['embedding_size'], options['embedding_size']],\
                                     stddev=1.0/math.sqrt(float(options['embedding_size']))), name='W_q_1')
            self.b_q_1 = tf.Variable(tf.zeros([options['embedding_size'],1]), name='b_q_1')

            self.W_q_2 = tf.Variable(tf.truncated_normal([options['embedding_size'], options['embedding_size']],\
                                     stddev=1.0/math.sqrt(float(options['embedding_size']))), name='W_q_2')
            self.b_q_2 = tf.Variable(tf.zeros([options['embedding_size'],1]), name='b_q_2')

            self.W_i_1 = tf.Variable(tf.truncated_normal([options['embedding_size'], options['embedding_size']],\
                                     stddev=1.0/math.sqrt(float(options['embedding_size']))), name='W_i_1')

            self.W_i_2 = tf.Variable(tf.truncated_normal([options['embedding_size'], options['embedding_size']],\
                                     stddev=1.0/math.sqrt(float(options['embedding_size']))), name='W_i_2')

    def forward_pass_lstm(self, word_embeddings):
        x = word_embeddings
        output = None
        for l in range(self.options['num_lstm_layers']):
            h = [None for i in range(self.options['lstm_steps'])]
            c = [None for i in range(self.options['lstm_steps'])]
            layer_output = []
            for lstm_step in range(self.options['lstm_steps']-1):
                if lstm_step == 0:
                    lstm_preactive = tf.matmul(x[lstm_step], self.lstm_W[l]) + self.lstm_b[l]
                else:
                    lstm_preactive = tf.matmul(h[lstm_step-1], self.lstm_U[l]) + tf.matmul(x[lstm_step], self.lstm_W[l]) + self.lstm_b[l]

                i, f, o, new_c = tf.split(lstm_preactive, num_or_size_splits = 4, axis = 1) 
                i = tf.nn.sigmoid(i)
                f = tf.nn.sigmoid(f)
                o = tf.nn.sigmoid(o)
                new_c = tf.nn.tanh(new_c)
                
                if lstm_step == 0:
                    c[lstm_step] = i * new_c
                else:
                    c[lstm_step] = f * c[lstm_step-1] + i * new_c

                h[lstm_step] = o * tf.nn.tanh(c[lstm_step])
                layer_output.append(h[lstm_step])

            x = layer_output
            output = layer_output

        return output

    def build_model(self):
        cnn7_features = tf.placeholder('float32',[ None, self.options['cnn7_feature_length'],49 ], name = 'cnn7')
        sentence = tf.placeholder('int32',[None, self.options['lstm_steps'] - 1], name = "sentence")#is a matrix[ques,512]
        answer = tf.placeholder('float32', [None, self.options['ans_vocab_size']], name = "answer")


        word_embeddings = []
        for i in range(self.options['lstm_steps']-1):
            word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,i])
            word_emb = tf.nn.dropout(word_emb, self.options['word_emb_dropout'], name = "word_emb" + str(i))
            
            word_embeddings.append(word_emb)


        word_embeddings = tf.stack(word_embeddings,0)
        lstm_output = self.forward_pass_lstm(word_embeddings)
        lstm_answer = lstm_output[-1]
        print(lstm_answer)

        
        logits,answer_probab = self.stack_att(cnn7_features,sentence,lstm_answer)

        ce = tf.nn.softmax_cross_entropy_with_logits(labels=answer, logits= logits[:,:,0], name = 'ce')
        # answer_probab = tf.nn.softmax(logits, name='answer_probab')
        
        predictions = tf.argmax(answer_probab,axis=1)
        correct_predictions = tf.equal(tf.argmax(answer_probab,1), tf.argmax(answer,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        loss = tf.reduce_sum(ce, name = 'loss')
        input_tensors = {
            'cnn7' : cnn7_features,
            'sentence' : sentence,
            'answer' : answer
        }
        return input_tensors, loss, accuracy, predictions
    
    def build_generator(self,batch):

        cnn7_features = tf.placeholder('float32',[ None, self.options['cnn7_feature_length'],49 ], name = 'cnn7')
        sentence = tf.placeholder('int32',[None, self.options['lstm_steps'] - 1], name = "sentence")

        word_embeddings = []
        for i in range(self.options['lstm_steps']-1):
            word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,i])
            word_embeddings.append(word_emb)


        lstm_output = self.forward_pass_lstm(word_embeddings)
        lstm_answer = lstm_output[-1]
        
        logits,answer_probab = self.stack_att(cnn7_features,sentence,lstm_answer,batch=batch)

        
        predictions = tf.argmax(answer_probab,1)

        input_tensors = {
            'cnn7' : cnn7_features,
            'sentence' : sentence
        }

        return input_tensors, predictions, answer_probab
    
    def stack_att(self,cnn7_features,sentence,lstm_answer,batch=200):
        # image_embedding = cnn7_features
        W_i_0 = tf.broadcast_to(self.W_i_0, [batch,self.options['cnn7_feature_length'],self.options['embedding_size']])
        b_i_0 = tf.broadcast_to(self.b_i_0, [batch,self.options['embedding_size'],49])
        image_embedding = tf.matmul(W_i_0,cnn7_features) + b_i_0
        image_embedding = tf.nn.tanh(image_embedding)
        image_embedding = tf.nn.dropout(image_embedding, self.options['image_dropout'], name = "vis_features")
        print(f'3======> image_embedding.shape: {image_embedding.shape}')

        W_i_1 = tf.broadcast_to(self.W_i_1, [batch,self.options['embedding_size'],self.options['embedding_size']])
        W_q_1 = tf.broadcast_to(self.W_q_1, [batch,self.options['embedding_size'],self.options['embedding_size']])
        b_q_1 = tf.broadcast_to(self.b_q_1, [batch,self.options['embedding_size'],1])
        u0 = tf.broadcast_to(lstm_answer, [batch,self.options['embedding_size']])
        u0 = tf.expand_dims(u0, 2)
        addto = tf.matmul(W_q_1,u0)+b_q_1
        addto = tf.broadcast_to(addto,[batch,self.options['embedding_size'],49])
        h1 = tf.math.add(tf.matmul(W_i_1,image_embedding), addto)
        h1 = tf.nn.tanh(h1)
        W_p_1 = tf.broadcast_to(tf.transpose(self.W_p_1),[batch,1,self.options['embedding_size']])
        b_p_1 = tf.broadcast_to(tf.transpose(self.b_p_1),[batch,1,49])
        p1 = tf.nn.softmax(tf.matmul(W_p_1,h1)+b_p_1)
        p1 = tf.transpose(p1, [0, 2, 1])

        u1 = u0 + image_embedding @ p1

        W_i_2 = tf.broadcast_to(self.W_i_2, [batch,self.options['embedding_size'],self.options['embedding_size']])
        W_q_2 = tf.broadcast_to(self.W_q_2, [batch,self.options['embedding_size'],self.options['embedding_size']])
        b_q_2 = tf.broadcast_to(self.b_q_2, [batch,self.options['embedding_size'],1])
        W_p_2 = tf.broadcast_to(tf.transpose(self.W_p_2),[batch,1,self.options['embedding_size']])
        b_p_2 = tf.broadcast_to(tf.transpose(self.b_p_2),[batch,1,49])
        h2 = tf.math.add(tf.matmul(W_i_2,image_embedding), (tf.matmul(W_q_2,u1)+b_q_2))
        h2 = tf.nn.tanh(h2)
        p2 = tf.nn.softmax(tf.matmul(W_p_2,h2)+b_p_2)
        p2 = tf.transpose(p2, [0, 2, 1])

        u2 = u1 + tf.matmul(image_embedding,p2) #tf.math.reduce_sum(tf.math.multiply(p2, image_embedding))

        W_u = tf.broadcast_to(tf.transpose(self.W_u),[batch,self.options['ans_vocab_size'],self.options['embedding_size']])
        b_u = tf.broadcast_to(self.b_u,[batch,self.options['ans_vocab_size'],1])
        logits = tf.matmul(W_u,u2)+b_u
        answer_probab = tf.nn.softmax(logits[:,:,0], name='answer_probab')
        
        return logits,answer_probab
