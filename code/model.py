import tensorflow as tf 
import os 
import numpy as np 

def _init_weight_variable(shape,name):
    return tf.get_variable(name, shape=shape, 
        initializer=tf.contrib.layers.xavier_initializer())

def _init_bias_variable(shape, name):
    return tf.get_variable(name,shape=shape, 
        initializer=tf.zeros_initializer())

class qa_model(object):
    def __init__(self,vocab,batch_size,att_size,hidden_size,dropout):
        self.hidden_size = hidden_size
        self.att_size = att_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.vocab = vocab

    def GRU_Q(self,question):
        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
        if self.dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.dropout)
        with tf.variable_scope('embed'):
            embedding_mat = tf.Variable(tf.random_normal([self.vocab, 300]),name='embed')
            embedding_qus = tf.nn.embedding_lookup(embedding_mat, question) 
        with tf.variable_scope('gruq'):
            inputs = tf.transpose(embedding_qus,perm=[1,0,2]) 
            outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, dtype=tf.float32,time_major=True)
        return outputs, state 

    def GRU_A_test(self,answer,initial, mask):
        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
        if self.dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.dropout)
        W = _init_weight_variable([self.hidden_size,self.vocab],name='ans_W')
        b = _init_bias_variable([self.vocab],name='ans_b')
        logit_words = []
        with tf.variable_scope('embed'):
            embedding_mat = tf.Variable(tf.random_normal([self.vocab, 300]),name='embed')
        with tf.variable_scope("grua"):
            for i in range(answer.get_shape().as_list()[1]):
                if i > 0: 
                    tf.get_variable_scope().reuse_variables()
                if i == 0:
                    embedding_ans = tf.nn.embedding_lookup(embedding_mat, tf.ones([1],dtype=tf.int32))
                    outputs, state = cell(embedding_ans, initial)
                else:
                    outputs, state = cell(embedding_ans, state)
                outputs = tf.reshape(outputs,(-1,self.hidden_size)) 
                word = tf.nn.xw_plus_b(outputs, W, b) 
                word = tf.reshape(word,[self.vocab]) 
                max_prob_index = tf.argmax(word)
                logit_words.append(word)
                embedding_ans = tf.nn.embedding_lookup(embedding_mat, max_prob_index) 
                embedding_ans = tf.expand_dims(embedding_ans, 0)
    
        return state, logit_words

    def GRU_A(self,answer,initial, mask):
        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
        state = initial
        if self.dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.dropout)
        W = _init_weight_variable([self.hidden_size,self.vocab],name='ans_W')
        b = _init_bias_variable([self.vocab],name='ans_b')
        with tf.variable_scope('embed'):
            embedding_mat = tf.Variable(tf.random_normal([self.vocab, 300]),name='embed')
        with tf.variable_scope("grua"):
            loss = 0.0
            for i in range(answer.get_shape().as_list()[1]-1):
                if i > 0: 
                    tf.get_variable_scope().reuse_variables()
                embedding_ans = tf.nn.embedding_lookup(embedding_mat, answer[:,i]) 
                outputs, state = cell(embedding_ans, state)
                outputs = tf.reshape(outputs,(-1,self.hidden_size)) 
                word = tf.nn.xw_plus_b(outputs, W, b) 
                labels = tf.expand_dims(answer[:,i+1], 1) 
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) 
                concated = tf.concat([indices, labels], axis=1)
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocab]), 1.0, 0.0)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=word, labels=onehot_labels) 
                current_loss = tf.reduce_sum(cross_entropy) / self.batch_size 
                loss = loss + current_loss
        loss /= (answer.get_shape().as_list()[1]-1)
        return state, loss

    def Vatt(self, v_feature, q_feature): 
        att_size = self.att_size
        max_cnt = v_feature.get_shape().as_list()[1]
        q_size = q_feature.get_shape().as_list()[-1]
        v_size = v_feature.get_shape().as_list()[-1]
        with tf.variable_scope('att_v'):
            Wq = _init_weight_variable([q_size,att_size],name='Wq') 
            Wv = _init_weight_variable([v_size,att_size],name='Wv') 
            Wo = _init_weight_variable([att_size,1],name='Wo') 
            b = _init_bias_variable([att_size], name='b') 
            v = tf.reshape(v_feature,(-1,v_size)) 
            q = tf.tile(q_feature,[1,max_cnt])
            q = tf.reshape(q,(-1,q_size))

            Rq             = tf.matmul(q, Wq)                          
            Rv             = tf.matmul(v, Wv)                        
            hidden         = Rv + Rq + b                            
            act_h          = tf.tanh(hidden)                    
            Ro             = tf.matmul(act_h, Wo)                 
            logits         = tf.reshape(Ro, (-1, max_cnt))         
            weight         = tf.nn.softmax(logits)                
            weight         = tf.expand_dims(weight, 2)           
            frame          = tf.reshape(v, (-1, max_cnt, v_size))     
            weighted_frame = tf.multiply(frame, weight)             
            avg            = tf.reduce_sum(weighted_frame, 1)       
        return avg

    def Qatt(self, v_feature, q_feature):
        q_feature = tf.transpose(q_feature,perm=[1,0,2]) 
        att_size = self.att_size
        max_len = q_feature.get_shape().as_list()[1]
        q_size = q_feature.get_shape().as_list()[-1]
        v_size = v_feature.get_shape().as_list()[-1]
        with tf.variable_scope('att_q'):
            Wq = _init_weight_variable([q_size,att_size],name='Wq') 
            Wv = _init_weight_variable([v_size,att_size],name='Wv') 
            Wo = _init_weight_variable([att_size,1],name='Wo') 
            b = _init_bias_variable([att_size], name='b') 
            q = tf.reshape(q_feature,(-1,q_size)) 
            v = tf.tile(v_feature,[1,max_len]) 
            v = tf.reshape(v,(-1,v_size)) 
            
            Rv           = tf.matmul(v, Wv)
            Rq           = tf.matmul(q, Wq)
            hidden       = Rv + Rq + b
            act_h        = tf.tanh(hidden)
            Ro           = tf.matmul(act_h, Wo)
            logits       = tf.reshape(Ro, (-1, max_len))
            weight       = tf.nn.softmax(logits)
            weight       = tf.expand_dims(weight, 2)
            qus          = tf.reshape(q, (-1, max_len, q_size))
            weighted_qus = tf.multiply(qus, weight)
            avg          = tf.reduce_sum(weighted_qus, 1)
        return avg 

    def cross_modal_comb(self, v_feature, q_feature, comb_size, input_size):
        v_size = v_feature.get_shape().as_list()[-1]
        q_size = q_feature.get_shape().as_list()[-1]
        with tf.variable_scope('comb'):
            Wq = _init_weight_variable([q_size,comb_size],name='Wq') 
            Wv = _init_weight_variable([v_size,comb_size],name='Wv') 
            Wc = _init_weight_variable([2 * comb_size,comb_size],name='Wc')
            Wd = _init_weight_variable([3 * comb_size,input_size],name='Wd')
            bq = _init_bias_variable([comb_size], name='bq') 
            bv = _init_bias_variable([comb_size], name='bv')
            bc = _init_bias_variable([comb_size], name='bc')
            bd = _init_bias_variable([input_size], name='bd')
            v = tf.nn.xw_plus_b(v_feature, Wv, bv) 
            q = tf.nn.xw_plus_b(q_feature, Wq, bq)
            v = tf.nn.l2_normalize(x=v, dim=[1])
            q = tf.nn.l2_normalize(x=q, dim=[1])
            concat = tf.concat([v,q],1)
            concat_feature = tf.nn.xw_plus_b(concat,Wc,bc) 
            mul_feature = tf.multiply(v, q) 
            add_feature = tf.add(v, q)        
        comb_feature = tf.concat([mul_feature, add_feature, concat_feature],1)
        comb_feature = tf.nn.xw_plus_b(comb_feature,Wd,bd) 
        return comb_feature 

    def Ansatt(self, v_feature, ans_feature,q_feature): 
        with tf.variable_scope('att_a'):
            if q_feature.get_shape().as_list()[-1] != ans_feature.get_shape().as_list()[-1]:
                Wq = _init_weight_variable([q_feature.get_shape().as_list()[-1],ans_feature.get_shape().as_list()[-1]],name='Wq') 
                bq = _init_bias_variable([ans_feature.get_shape().as_list()[-1]], name='bq')  
                q_feature = tf.nn.xw_plus_b(q_feature,Wq,bq)
            ans_feature = tf.nn.l2_normalize(x=ans_feature, dim=[1])
            q_feature = tf.nn.l2_normalize(x=q_feature, dim=[1])
            feature = tf.concat([ans_feature,  q_feature],1)
            att_size = self.att_size
            max_cnt = v_feature.get_shape().as_list()[1] 
            ans_size = feature.get_shape().as_list()[-1]
            v_size = v_feature.get_shape().as_list()[-1]
            Wa = _init_weight_variable([ans_size,att_size],name='Wa') 
            Wv = _init_weight_variable([v_size,att_size],name='Wv') 
            Wo = _init_weight_variable([att_size,1],name='Wo') 
            Wd= _init_weight_variable([v_size,100],name='Wd')
            Wy = _init_weight_variable([100,3],name='Wy')
            b = _init_bias_variable([att_size], name='b') 
            bd = _init_bias_variable([100], name='bd')
            by = _init_bias_variable([3], name='by')
            v = tf.reshape(v_feature,(-1,v_size)) 
            ans = tf.tile(feature,[1,max_cnt])
            ans = tf.reshape(ans,(-1,ans_size)) 

            Ra             = tf.matmul(ans, Wa)                     
            Rv             = tf.matmul(v, Wv)                     
            hidden         = Rv + Ra + b                             
            act_h          = tf.tanh(hidden)                         
            Ro             = tf.matmul(act_h, Wo)                   
            logits         = tf.reshape(Ro, (-1, max_cnt))           
            weight         = tf.nn.softmax(logits)                   
            weight         = tf.expand_dims(weight, 2)               
            frame          = tf.reshape(v, (-1, max_cnt, v_size))    
            weighted_frame = tf.multiply(frame, weight)              
            final_v = tf.reshape(weighted_frame,(-1,v_size))
            output = tf.nn.xw_plus_b(final_v,Wd,bd) 
            output = tf.nn.xw_plus_b(output,Wy,by) 
            output = tf.reshape(output,(-1,max_cnt,3))
        return output 


    def compute_loss_reg(self, mask, output, label, offset, ans_loss,lambda_regression, lambda_ans, alpha):
        max_cnt = output.get_shape().as_list()[1]
        score, l_off, r_off = tf.split(output,3,2) 
        score = tf.squeeze(score) 
        score = tf.multiply(score,mask)
        l_off = tf.squeeze(l_off)
        r_off = tf.squeeze(r_off)
        all1 = tf.constant(1.0, shape=[self.batch_size, max_cnt])
        batch_para_mat = tf.constant(alpha, shape=[self.batch_size, max_cnt])
        para_mat = tf.add(label,batch_para_mat)
        loss_mat = tf.log(tf.add(all1, tf.exp(-tf.multiply(label, score)))) 
        loss_mat = tf.multiply(loss_mat, para_mat) * mask
        index = tf.to_float(tf.count_nonzero(mask))
        loss_align = tf.divide(tf.reduce_sum(loss_mat),index)
        l_off = tf.expand_dims(l_off,axis=2)
        r_off = tf.expand_dims(r_off,axis=2)
        offset_pred = tf.concat((l_off, r_off),2) 
        mask = tf.expand_dims(mask,2)
        mask = tf.tile(mask,[1,1,2])
        x = tf.abs(tf.subtract(offset_pred, offset)) * mask
        loss_reg = tf.divide(tf.reduce_sum(x),index)
        loss_reg = tf.multiply(lambda_regression, loss_reg)
        loss = tf.add(loss_reg,loss_align) + tf.multiply(lambda_ans, ans_loss)
        return loss
        
    def forward(self,question,answer,c3d_feature,res_feature,skip_feature,label,offset,skip,lambda_regression,lambda_ans,alpha,qkind,vkind,mask,training):

        if qkind == 'TVQA':
            W = _init_weight_variable([2048,500],name='TVQA_W') 
            b = _init_bias_variable([500], name='TVQA_b')
            v = tf.reshape(c3d_feature,[-1,2048])
            vv = tf.nn.xw_plus_b(v,W,b)
            v_feature = tf.reshape(vv,[-1,c3d_feature.get_shape().as_list()[1],500])
        if vkind == 'combine':
            Wv = _init_weight_variable([2048,500],name='com_W') 
            bv = _init_bias_variable([500], name='com_b')
            res = tf.reshape(res_feature,[-1,2048])
            res_v = tf.nn.xw_plus_b(res,Wv,bv)
            res_feature = tf.reshape(res_v,[-1,c3d_feature.get_shape().as_list()[1],500])
            tf_c3d = tf.nn.l2_normalize(x=c3d_feature, dim=[2])
            tf_resnet = tf.nn.l2_normalize(x=res_feature, dim=[2])
            v_feature = tf.concat([c3d_feature, res_feature], axis=2) 
        if vkind == 'c3d':
            v_feature = c3d_feature
        if vkind == 'res':
            Wv = _init_weight_variable([2048,500],name='res_W') 
            bv = _init_bias_variable([500], name='res_b')
            res = tf.reshape(res_feature,[-1,2048])
            v_feature = tf.nn.xw_plus_b(res,Wv,bv)
            v_feature = tf.reshape(v_feature, [-1,c3d_feature.get_shape().as_list()[1],500])

        q_feature, q_state = self.GRU_Q(question) 
        if skip == 'true':
            q_state = skip_feature 
        att_v = self.Vatt(v_feature,q_state) 
        att_q = self.Qatt(att_v,q_feature) 
        comb_feature = self.cross_modal_comb(att_v,att_q,comb_size=1024,input_size=512) 
        if training == True:
            ans, ans_loss= self.GRU_A(answer,comb_feature, mask)
            output = self.Ansatt(v_feature,ans,q_state) 
            total_loss = self.compute_loss_reg(mask,output, label, offset, ans_loss, lambda_regression, lambda_ans, alpha)
            return total_loss
        else:
            ans, logit_words = self.GRU_A_test(answer,comb_feature, mask)
            output = self.Ansatt(v_feature,ans,q_state) 
            return logit_words, output