import tensorflow as tf 
import argparse
import h5py
import json
import os
import time
import numpy as np 
import random
from model import qa_model
from keras.preprocessing import sequence
from nltk.corpus import wordnet as wn 
import pandas as pd 
from scipy import stats 
import tqdm
import heapq

train_qfeature = ''
test_qfeature = ''
c3d_train_vfeature = ''
res_train_vfeature = ''
c3d_test_vfeature = ''
res_test_vfeature = ''

def comb_WUPS(seq1, seq2, threhold):
    WUPS_score = np.zeros(len(threhold))
    seq1_wup = []
    for a in seq1:
        temp_a = wn.synsets(a)
        if(len(temp_a)) != 0:
            seq1_wup.append(temp_a)

    seq2_wup = []
    for a in seq2:
        temp_a = wn.synsets(a)
        if (len(temp_a)) != 0:
            seq2_wup.append(temp_a)

    if (len(seq1_wup) != 0) and (len(seq2_wup) != 0):
        pair_score = np.zeros((len(seq1_wup), len(seq2_wup)))

        for i, interp_a in enumerate(seq1_wup):
            for j, interp_b in enumerate(seq2_wup):
                # for a pair
                global_max = 0.0
                for x in interp_a:
                    for y in interp_b:
                        local_score = x.wup_similarity(y)
                        if local_score > global_max:
                            global_max = local_score
                pair_score[i][j] = global_max

        score1 = np.prod(np.amax(pair_score, axis=0))
        score2 = np.prod(np.amax(pair_score, axis=1))
        max_cnt = max([score1, score2])
        for idx, _ in enumerate(threhold):
            WUPS_score[idx] = max_cnt
            if max_cnt < threhold[idx]:
                WUPS_score[idx] *= 0.1
    return WUPS_score

def WUPs(groundtruth, prediction, ixtoword, threhold):
    answer = []
    for i in groundtruth:
        answer.append(ixtoword[str(i)])
        if ixtoword[str(i)] == '<eos>':
            break
    probs = [] 
    for i in prediction:
        probs.append(ixtoword[str(i)])
        if ixtoword[str(i)] == '<eos>':
            break
    wups = comb_WUPS(answer, probs, threhold) 
    return wups * 100.0
    
def WUP_IOU(groundtruth, prediction, ixtoword, threhold, iou,iouthrehold,metric):
    wups = WUPs(groundtruth, prediction, ixtoword, threhold) 
    print "wups", wups
    IoU = np.zeros([len(iouthrehold)])
    if metric == 'soft':
        for i in range(len(iouthrehold)):
            IoU[i] = iou
    else:
        for i in range(len(iouthrehold)):
            if iou > iouthrehold[i]:
                IoU[i] = 1.0
    print "iou", IoU
    IoU = np.expand_dims(IoU,0) 
    wups = np.expand_dims(wups,1) 
    result = np.dot(wups,IoU) 
    return result

def acc(groundtruth, prediction, ixtoword, threhold, iou,iouthrehold,metric):
    generated_word_index = []
    for i in range(prediction.shape[0]):
        max_prob_index = np.argmax(prediction[i])
        generated_word_index.append(max_prob_index)
    result = WUP_IOU(groundtruth, generated_word_index, ixtoword, threhold, iou,iouthrehold,metric)
    return result

def temporal_iou(start,end,start_gt,end_gt):
    union = min(start,start_gt), max(end,end_gt)
    inter = max(start,start_gt), min(end,end_gt)
    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1]-inter[0]) / float(union[1] - union[0])

def placeholder_inputs(max_len,max_cnt,kind):
    if kind == 'new':
        c3dfeature_placeholder = tf.placeholder(tf.float32,shape=[None,max_cnt,500])
    else:
        c3dfeature_placeholder = tf.placeholder(tf.float32,shape=[None,max_cnt,2048])
    resfeature_placeholder = tf.placeholder(tf.float32,shape=[None,max_cnt,2048])
    qfeature_placeholder = tf.placeholder(tf.float32,shape=[None,4800])
    label_placeholder = tf.placeholder(tf.float32,shape=[None,max_cnt])
    offset_placeholder = tf.placeholder(tf.float32,shape=[None,max_cnt,2])
    qid_placeholder = tf.placeholder(tf.int32,shape=[None,max_len])
    aid_placeholder = tf.placeholder(tf.int32,shape=[None,max_len])
    mask_placeholder = tf.placeholder(tf.float32,shape=[None,max_cnt])
    return c3dfeature_placeholder,resfeature_placeholder, qfeature_placeholder,label_placeholder,offset_placeholder,qid_placeholder,aid_placeholder,mask_placeholder

def test(ckpt_path, lr,batch_size,final_data,ixtoword,max_ques,max_cnt,att_size,hidden_size,skip,vfeat,lambda_regression, lambda_ans, alpha,kind, metric):
    model_save_dir = ckpt_path
    test_result_output = ckpt_path + '_' + str(metric) + '.txt' 
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    c3dfeature_placeholder,resfeature_placeholder, qfeature_placeholder,label_placeholder,offset_placeholder,qid_placeholder,aid_placeholder,mask_placeholder = placeholder_inputs(max_ques,max_cnt,kind)
    model = qa_model(len(ixtoword.keys()),batch_size,att_size,hidden_size,dropout=0.0)
    logit_words, output = model.forward(qid_placeholder,aid_placeholder,c3dfeature_placeholder,resfeature_placeholder, qfeature_placeholder,label_placeholder,offset_placeholder,skip,lambda_regression, lambda_ans, alpha,kind,vfeat,mask_placeholder, False)
    saver = tf.train.Saver(max_to_keep=20)
    init = tf.global_variables_initializer()
    WUPs_thresh = [0.0, 0.9]
    IOU_thresh = [0.1,0.3,0.5,0.7]
    with tf.Session(config=config) as sess:
        for i in range(0,50,5):
            ckpt = model_save_dir + '/' + 'model_' + str(i) + '.ckpt'
            saver.restore(sess, ckpt)
            tf_acc = np.zeros([len(WUPs_thresh),len(IOU_thresh)])
            for qa in tqdm.tqdm(final_data): 
                vid = qa['num']
                c3d_feature = np.expand_dims(c3d_test_vfeature[vid],0) #(max_cnt,500)
                res_feature = np.zeros([1,max_cnt,2048])
                if vfeat != 'c3d':
                    if kind == 'new':
                        res_feature = np.expand_dims(res_test_vfeature[vid],0)
                qfeature = np.expand_dims(test_qfeature[vid],0) 
                qid = np.expand_dims(np.array(qa['qid']),0) 
                aid = np.expand_dims(np.array(qa['aid']),0) 
                label = np.expand_dims(np.array(qa['label']),0)
                offset = np.expand_dims(np.array(qa['offset']),0) 
                mask = np.expand_dims(np.array(qa['mask']),0)
                start = np.array(qa['start']) 
                end = np.array(qa['end']) 
                frame = qa['frame']
                final_out, answer = sess.run([output,logit_words],feed_dict={
                        c3dfeature_placeholder:c3d_feature,
                        resfeature_placeholder:res_feature,
                        qfeature_placeholder:qfeature,
                        label_placeholder:label,
                        offset_placeholder:offset,
                        qid_placeholder:qid,
                        aid_placeholder:aid,
                        mask_placeholder:mask
                        })
                score, l_offset, r_offset = np.split(final_out,3,2)
                score = -np.squeeze(score) 
                mask = np.squeeze(mask)
                score *= mask
                l_offset = np.squeeze(l_offset)
                r_offset= np.squeeze(r_offset)
                index = np.argmax(score,0)
                reg_start = start[index] + l_offset[index] * frame
                reg_end = end[index] + r_offset[index] * frame 
                aid = np.squeeze(aid)
                aid = aid.tolist()   
                gt_s = qa['timestamps'][0] 
                gt_e = qa['timestamps'][1]  
                IoU = temporal_iou(gt_s, gt_e, reg_start, reg_end)
                if metric == 'hard_five':
                    index = heapq.nlargest(5, range(len(score)), score.take) 
                    IoU = 0.0 
                    for e in index:
                        reg_start = start[e] + l_offset[e] * frame
                        reg_end = end[e] + r_offset[e] * frame
                        gt_s = qa['timestamps'][0]
                        gt_e = qa['timestamps'][1] 
                        maxiouIoU = temporal_iou(gt_s, gt_e, reg_start, reg_end)
                        if maxiouIoU > IoU:
                            IoU = maxiouIoU
                answer = np.squeeze(answer)
                tf_acc += acc(aid,answer,ixtoword,WUPs_thresh,IoU,IOU_thresh,metric)
            tf_acc /= 1.0 * len(final_data)
            for k in range(len(WUPs_thresh)):
                for j in range(len(IOU_thresh)):
                    print "WUPs: "+str(WUPs_thresh[k])+" IOU: "+str(IOU_thresh[j]) + " acc is" + str(tf_acc[k][j])
                    with open(test_result_output,'a+') as f:
                        f.write("step: " +str(i) + " WUPs: "+str(WUPs_thresh[k])+" IOU: "+str(IOU_thresh[j]) + " acc is" + str(tf_acc[k][j])+ "\n")

def read_one_batch(num, batch_size,final_data,max_len,max_cnt,kind):
    
    key = final_data[num*batch_size:(num+1)*batch_size]
    total_qfeature = np.zeros([batch_size,4800])
    total_label = np.zeros([batch_size,max_cnt])
    total_offset = np.zeros([batch_size,max_cnt,2])
    total_qid = np.zeros([batch_size,max_len])
    total_aid = np.zeros([batch_size,max_len])
    total_mask = np.zeros([batch_size,max_cnt])
    total_resfeature = np.zeros([batch_size,max_cnt,2048])
    if kind == 'new':
        total_c3dfeature = np.zeros([batch_size,max_cnt,500])
    else:
        total_c3dfeature = np.zeros([batch_size,max_cnt,2048])
    for i, idx in enumerate(key):
        vid = idx['num']
        total_c3dfeature[i] = c3d_train_vfeature[vid] 
        if kind == 'new':
            total_resfeature[i] = res_train_vfeature[vid] 
        total_qfeature[i] = train_qfeature[vid]
        total_label[i] = np.array(idx['label']) 
        total_offset[i] = np.array(idx['offset']) 
        total_qid[i] = np.array(idx['qid'])
        total_aid[i] = np.array(idx['aid']) 
        total_mask[i] = np.array(idx['mask']) 
    return total_c3dfeature, total_resfeature, total_qfeature, total_label, total_offset, total_qid, total_aid, total_mask


def train(lr,totalepoch,batch_size,final_data,ixtoword,max_ques,max_cnt,att_size,hidden_size,skip,vfeat,lambda_regression, lambda_ans, alpha,kind):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
            )  
        c3dfeature_placeholder,resfeature_placeholder,qfeature_placeholder,label_placeholder,offset_placeholder,qid_placeholder,aid_placeholder,mask_placeholder = placeholder_inputs(max_ques,max_cnt,kind)
        model = qa_model(len(ixtoword.keys()),batch_size,att_size,hidden_size,dropout=0.5)
        loss = model.forward(qid_placeholder,aid_placeholder,c3dfeature_placeholder,resfeature_placeholder,qfeature_placeholder,label_placeholder,offset_placeholder,skip,lambda_regression,lambda_ans,alpha,kind,vfeat,mask_placeholder, True)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        saver = tf.train.Saver(max_to_keep=20)
        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            sess.run(init)
            num_batch = int(np.floor(len(final_data) / batch_size))
            for epoch in range(totalepoch):
                total_loss = 0.0
                random.shuffle(final_data)
                start = time.time()
                for i in tqdm.tqdm(range(num_batch)):
                    c3d, res,qfeature, label, offset, qid, aid, mask = read_one_batch(i,batch_size,final_data,max_ques,max_cnt,kind)
                    _ , tf_loss = sess.run([train_op,loss],feed_dict={
                        c3dfeature_placeholder:c3d,
                        resfeature_placeholder:res,
                        qfeature_placeholder:qfeature,
                        label_placeholder:label,
                        offset_placeholder:offset,
                        qid_placeholder:qid,
                        aid_placeholder:aid,
                        mask_placeholder:mask
                        })
                    total_loss += tf_loss
                duration = float(time.time() - start)
                print('step %d: %.3f sec, loss is %.3f.' % (epoch, duration,total_loss/num_batch))
                
                if epoch % 5 == 0:
                    checkpoint_path = os.path.join('own path', 'model_{0}.ckpt'.format(epoch))
                    if not os.path.exists(os.path.dirname(checkpoint_path)):
                        os.makedirs(os.path.dirname(checkpoint_path))
                    saver.save(sess, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=str,default='1')
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--epoch',type=int,default=50)
    parser.add_argument('--bsz',type=int,default=32)
    parser.add_argument('--att',type=int,default=1024)
    parser.add_argument('--hidden_size',type=int,default=512)
    parser.add_argument('--vfeat',type=str,default='combine',choices=['c3d','res','combine'])
    parser.add_argument('--mode',type=str,default='train',choices=['train','test'])
    parser.add_argument('--skip',type=str,default='false',choices=['false','true'])
    parser.add_argument('--regress',type=float,default=1.0)
    parser.add_argument('--ans',type=float,default=5.0)
    parser.add_argument('--alpha',type=float,default=2.5)
    parser.add_argument('--data',type=str,default='Activity-QA',choices=['Activity-QA','TVQA'])
    parser.add_argument('--ckpt',type=str,default='None')    
    parser.add_argument('--test',type=str,default='soft',choices=['soft','hard_one','hard_five'])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if args.data == 'Activity-QA':
        c3d_train_vfeature = ...
        c3d_test_vfeature = ...
        res_train_vfeature = ...
        res_test_vfeature = ...
        train_qfeature = ...
        test_qfeature = ...
    else:
        c3d_train_vfeature = ...
        c3d_test_vfeature = ...
        train_qfeature = ...
        test_qfeature = ...
    if args.mode == 'train':
        if args.data == 'Activity-QA':
            f = json.load(...)
        else:
            f = json.load(...)
        final_data = f['data']
        ixtoword = f['ixtoword'] 
        max_ques = f['max_ques'] 
        max_cnt= f['max_cnt'] 
        train(args.lr,args.epoch,args.bsz,final_data,ixtoword,max_ques,max_cnt,args.att,args.hidden_size,args.skip,args.vfeat,args.regress,args.ans,args.alpha,args.data)

    if args.mode =='test':
        if args.data == 'Activity-QA':
            f = json.load(...)
        else:
            f = json.load(...)
        final_data = f['data']
        ixtoword = f['ixtoword'] 
        max_ques = f['max_ques'] 
        max_cnt= f['max_cnt'] 
        test(args.ckpt,args.lr,1,final_data,ixtoword,max_ques,max_cnt,args.att,args.hidden_size,args.skip,args.vfeat,args.regress,args.ans,args.alpha,args.data,args.test)

