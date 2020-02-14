
import sys, pdb, os
import numpy as np
import pickle
import tensorflow as tf
import sonnet as snt
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print("Tensorflow verion:",tf.__version__)


def load_data(input_path,min_threshold, max_threshold, seed, seqs_name='.seqs', labels_name='.labels'):
    seqs = pickle.load(open(input_path + seqs_name, 'rb'))
    labels = pickle.load(open(input_path + labels_name, 'rb'))

    new_seqs = []
    new_labels = []
    new_demos= []
    for seq, label in zip(seqs, labels):
        if len(seq) < min_threshold or len(seq) > max_threshold:
            continue
        else:
            new_seqs.append(seq)
            new_labels.append(label)
    
    
    seqs = new_seqs
    labels = new_labels

    temp_seqs, test_seqs, temp_labels, test_labels = train_test_split(seqs, labels, test_size=0.2, random_state=seed)
    train_seqs, valid_seqs, train_labels, valid_labels = train_test_split(temp_seqs, temp_labels, test_size=0.1, random_state=seed)

    train_size = int(len(train_seqs))
    train_seqs = train_seqs[:train_size]
    train_labels = train_labels[:train_size]
    
    #sort patient's sequence by its visit length
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    sorted_index = len_argsort(train_seqs)
    train_seqs = [train_seqs[i] for i in sorted_index]
    train_labels = [train_labels[i] for i in sorted_index]

    sorted_index = len_argsort(valid_seqs)
    valid_seqs = [valid_seqs[i] for i in sorted_index]
    valid_labels = [valid_labels[i] for i in sorted_index]

    sorted_index = len_argsort(test_seqs)
    test_seqs = [test_seqs[i] for i in sorted_index]
    test_labels = [test_labels[i] for i in sorted_index]


    return train_seqs, train_labels, valid_seqs, valid_labels, test_seqs, test_labels

# build the model
def build_model(options):
    #different choice of activation funciton
    if options['emb_activation'] == 'sigmoid':
        emb_activation = tf.nn.sigmoid
    else:
        emb_activation = tf.nn.relu

    if options['visit_activation'] == 'sigmoid':
        visit_activation = tf.nn.sigmoid
    else:
        visit_activation = tf.nn.relu
        
    #embedding matrix, dx for icd code; rx for medication code; pr for procedure code
    #num is the unique number of each code; size is the embedding size to be chosen, 200 as default
    W_emb_icd = tf.get_variable('W_emb_icd', shape=(options['num_icd'], options['icd_emb_size']), dtype=tf.float32)
    W_emb_cpt = tf.get_variable('W_emb_cpt', shape=(options['num_cpt'], options['cpt_emb_size']), dtype=tf.float32)
    W_emb_med = tf.get_variable('W_emb_med', shape=(options['num_med'], options['med_emb_size']), dtype=tf.float32)
    W_emb_demo = tf.get_variable('W_emb_demo', shape=(options['num_demo'], options['demo_emb_size']), dtype=tf.float32)
    W_emb_topic = tf.get_variable('W_emb_topic', shape=(options['num_topic'], options['topic_emb_size']), dtype=tf.float32)

    #input vector for dx,rx,pr 
    icd_var = tf.placeholder(tf.int32, shape=(None, options['batch_size'], options['max_icd_per_visit']), name='icd_var')
    cpt_var = tf.placeholder(tf.int32, shape=(None, options['batch_size'], options['max_cpt_per_visit']), name='cpt_var')
    med_var = tf.placeholder(tf.int32, shape=(None, options['batch_size'], options['max_med_per_visit']), name='med_var')
    demo_var = tf.placeholder(tf.int32, shape=(None, options['batch_size'], options['demo_per_visit']), name='demo_var')
    topic_var = tf.placeholder(tf.int32, shape=(None, options['batch_size'], options['max_topic_per_visit']), name='demo_var')

    #mask for residual connection
    icd_mask = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_icd_per_visit']), name='icd_mask')
    cpt_mask = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_cpt_per_visit']), name='cpt_mask')
    med_mask = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_med_per_visit']), name='med_mask')
    demo_mask = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['demo_per_visit']), name='demo_mask')
    topic_mask = tf.placeholder(tf.float32, shape=(None, options['batch_size'], options['max_topic_per_visit']), name='demo_mask')

    # lookup is the multiplication to retrieve embeddings from embedding matrix
    #second param is the index to retrieve
    icd_visit = tf.nn.embedding_lookup(W_emb_icd, tf.reshape(icd_var, (-1, options['max_icd_per_visit'])))
    icd_visit = emb_activation(icd_visit)
    icd_visit = icd_visit * tf.reshape(icd_mask, (-1, options['max_icd_per_visit']))[:, :, None] ####Masking####
    icd_visit = tf.reduce_sum(icd_visit, axis=1)
    
   # dx_visit = tf.reshape(dx_visit, (-1, options['dx_emb_size'])) # dx_emb_size=200

    cpt_visit = tf.nn.embedding_lookup(W_emb_cpt, tf.reshape(cpt_var, (-1, options['max_cpt_per_visit'])))
    cpt_visit = emb_activation(cpt_visit)
    cpt_visit = cpt_visit * tf.reshape(cpt_mask, (-1, options['max_cpt_per_visit']))[:, :, None] ####Masking####
    cpt_visit = tf.reduce_sum(cpt_visit, axis=1)
    
    med_visit = tf.nn.embedding_lookup(W_emb_med, tf.reshape(med_var, (-1, options['max_med_per_visit'])))
    med_visit = emb_activation(med_visit)
    med_visit = med_visit * tf.reshape(med_mask, (-1, options['max_med_per_visit']))[:, :, None] ####Masking####
    med_visit = tf.reduce_sum(med_visit, axis=1)
    
    demo_visit = tf.nn.embedding_lookup(W_emb_demo, tf.reshape(demo_var, (-1, options['demo_per_visit'])))
    demo_visit = emb_activation(demo_visit)
    demo_visit = demo_visit * tf.reshape(demo_mask, (-1, options['demo_per_visit']))[:, :, None] ####Masking####
    demo_visit = tf.reduce_sum(demo_visit, axis=1)
    
    topic_visit = tf.nn.embedding_lookup(W_emb_topic, tf.reshape(topic_var, (-1, options['max_topic_per_visit'])))
    topic_visit = emb_activation(topic_visit)
    topic_visit = topic_visit * tf.reshape(topic_mask, (-1, options['max_topic_per_visit']))[:, :, None] ####Masking####
    topic_visit = tf.reduce_sum(topic_visit, axis=1)
    
    # adding attention weights for each EHR type
    
    attention = tf.get_variable('attens',shape=(1,5),dtype=tf.float32, 
                                initializer=tf.constant_initializer([[1.0,1.0,1.0,1.0,1.0]]))
    attention_weights = tf.nn.softmax(attention)
    
    EHR_obj = tf.gather_nd(attention_weights,[[0,0]])*icd_visit + tf.gather_nd(attention_weights,[[0,1]])*cpt_visit + tf.gather_nd(attention_weights,[[0,2]])*med_visit + tf.gather_nd(attention_weights,[[0,3]])*demo_visit + tf.gather_nd(attention_weights,[[0,4]])*topic_visit
    W_dx = snt.Sequential([snt.Linear(output_size=options['ehr_emb_size'], name='W_ehr'), visit_activation])
    EHR_obj = W_dx(EHR_obj)
    
#     use the following codes if not use attention weights   
#     # sum of ICD-9,medication and cpt codes
#     EHR_obj = icd_visit + cpt_visit + med_visit + demo_visit + topic_visit
#     W_dx = snt.Sequential([snt.Linear(output_size=options['ehr_emb_size'], name='W_ehr'), visit_activation])
#     EHR_obj = W_dx(EHR_obj)
    
    seq_visit = tf.reshape(EHR_obj, (-1, options['batch_size'], options['visit_emb_size']))     
    seq_length = tf.placeholder(tf.int32, shape=(options['batch_size']), name='seq_length')
    rnn = snt.GRU(options['rnn_size'], name='emb2rnn')
    rnn2pred = snt.Sequential([snt.Linear(output_size=options['output_size'], name='rnn2pred'), tf.nn.sigmoid])
    
    
    #output
    _, final_states = tf.nn.dynamic_rnn(rnn, seq_visit, dtype=tf.float32, time_major=True, sequence_length=seq_length)
    preds = tf.squeeze(rnn2pred(final_states))
    labels = tf.placeholder(tf.float32, shape=(options['batch_size']), name='labels')
    loss = -tf.reduce_mean(labels * tf.log(preds + 1e-10) + (1. - labels) * tf.log(1. - preds + 1e-10))
    
    
    #get tensors for this model
    input_tensors = (icd_var, cpt_var, med_var,demo_var,topic_var)
    label_tensors = labels
    mask_tensors = (icd_mask, cpt_mask, med_mask,demo_mask,topic_mask)
    loss_tensors = loss

    return input_tensors, label_tensors, mask_tensors, loss_tensors, seq_length, preds, attention_weights

#function to run test set
def run_test(seqs, label_seqs, sess, preds_T, input_PHs, label_PHs, mask_PHs, seq_length_PH, loss_T, options):
    all_losses = []
    all_preds = []
    all_labels = []
    batch_size = options['batch_size']
    
    for idx in range(len(label_seqs) // batch_size):
        batch_x = seqs[idx*batch_size:(idx+1)*batch_size]
        batch_y = label_seqs[idx*batch_size:(idx+1)*batch_size]
        inputs, masks, seq_length = preprocess_batch(batch_x, options)
        
        preds, loss = sess.run([preds_T, loss_T],
                feed_dict={
                    input_PHs[0]:inputs[0],
                    input_PHs[1]:inputs[1],
                    input_PHs[2]:inputs[2],
                    input_PHs[3]:inputs[3],
                    input_PHs[4]:inputs[4],
                    mask_PHs[0]:masks[0],
                    mask_PHs[1]:masks[1],
                    mask_PHs[2]:masks[2],
                    mask_PHs[3]:masks[3],
                    mask_PHs[4]:masks[4],
                    label_PHs:batch_y,      # only feed the label for global prediction in our case
                    seq_length_PH:seq_length,
                    }
                )
        
        all_losses.append(loss)
        all_preds.extend(list(preds))
        all_labels.extend(batch_y)
        
    auc = roc_auc_score(all_labels, all_preds)
    aucpr = average_precision_score(all_labels, all_preds)
    accuracy = (np.array(all_labels) == np.squeeze(binarize(np.array(all_preds).reshape(-1, 1), threshold=.5))).mean()
    return np.mean(all_losses), auc, aucpr

def train(input_path,
          output_path,
          batch_size,
          num_iter,
          eval_period,
          rnn_size,
          output_size,
          learning_rate,
          random_seed,
          split_seed,
          emb_activation,
          visit_activation,
          num_icd,
          num_cpt,
          num_med,
          num_demo,
          num_topic,
          icd_emb_size,
          cpt_emb_size,
          med_emb_size,
          demo_emb_size,
          topic_emb_size,
          ehr_emb_size,
          visit_emb_size,
          max_icd_per_visit,
          max_cpt_per_visit,
          max_med_per_visit,
          max_topic_per_visit,
          demo_per_visit,
          regularize,
          min_threshold,
          max_threshold):
            
    #copy local parameters
    options = locals().copy()
    #build the model
    input_PHs, label_PHs, mask_PHs, loss_T, seq_length_PH, preds_T, attention_weights = build_model(options)
    
    #add L2 regularization
    all_vars = tf.trainable_variables()
    L2_loss = tf.constant(0.0, dtype=tf.float32)
    for var in all_vars:
        if len(var.shape) < 2:
            continue
        L2_loss += tf.reduce_sum(var ** 2)

    optimizer = tf.train.AdamOptimizer(learning_rate=options['learning_rate'])
    minimize_op = optimizer.minimize(loss_T + regularize * L2_loss)
    
    train_seqs, train_labels, valid_seqs, valid_labels, test_seqs, test_labels = load_data(
            options['input_path'],
            min_threshold=options['min_threshold'],
            max_threshold=options['max_threshold'],
            seed=options['split_seed'])
    
    #save the file and check point
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_valid_loss = 100000.0
        best_test_loss = 100000.0
        best_valid_auc = 0.0
        best_test_auc = 0.0
        best_valid_aucpr = 0.0
        best_test_aucpr = 0.0
        
        for train_iter in range(options['num_iter']+1):
            batch_x, batch_y = sample_batch(train_seqs, train_labels, options['batch_size'])
            inputs, masks, seq_length = preprocess_batch(batch_x, options)
            
            _, preds, losses = sess.run([minimize_op, preds_T, loss_T],
                    feed_dict={
                        input_PHs[0]:inputs[0],
                        input_PHs[1]:inputs[1],
                        input_PHs[2]:inputs[2],
                        input_PHs[3]:inputs[3],
                        input_PHs[4]:inputs[4],
                        mask_PHs[0]:masks[0],
                        mask_PHs[1]:masks[1],
                        mask_PHs[2]:masks[2],
                        mask_PHs[3]:masks[3],
                        mask_PHs[4]:masks[4],
                        label_PHs:batch_y,
                        seq_length_PH:seq_length,
                        }
                    )

            if train_iter > 0 and train_iter % options['eval_period'] == 0:
                #validation process
                valid_loss, valid_auc, valid_aucpr = run_test(valid_seqs, valid_labels, sess, preds_T, 
                                                              input_PHs, label_PHs, mask_PHs, seq_length_PH, 
                                                              loss_T, options)
                #test the model
                if valid_loss < best_valid_loss:
                    test_loss, test_auc, test_aucpr = run_test(test_seqs, test_labels, sess, preds_T, input_PHs,
                                                               label_PHs, mask_PHs, seq_length_PH, loss_T, options)
                    best_valid_loss = valid_loss
                    best_valid_auc = valid_auc
                    best_valid_aucpr = valid_aucpr
                    best_test_loss = test_loss
                    best_test_auc = test_auc
                    best_test_aucpr = test_aucpr
                    best_attention=attention_weights.eval()
                    print("attention weights:", best_attention)
                    print('\n')
                    
                    savePath = saver.save(sess, output_path + '/r' + str(random_seed) + 's' + str(split_seed) 
                                          + '/model', global_step=train_iter)
                print('Steps: %d, valid_loss:%f, valid_auc:%f, valid_aucpr:%f' % (train_iter, valid_loss, valid_auc,
                                                                                 valid_aucpr))
                
        return best_valid_loss, best_test_loss, best_valid_auc, best_test_auc, best_valid_aucpr, best_test_aucpr, best_attention

def preprocess_batch(patients, options):
         
    lengths = np.array([len(seq) for seq in patients])# visit length for each patient
    max_length = np.max(lengths)# max length of visit
    num_samples = len(patients)
    
    icd = np.zeros((num_samples, max_length, options['max_icd_per_visit'])).astype('int32')
    cpt = np.zeros((num_samples, max_length, options['max_cpt_per_visit'])).astype('int32')
    med = np.zeros((num_samples, max_length, options['max_med_per_visit'])).astype('int32')
    demo = np.zeros((num_samples, max_length, options['demo_per_visit'])).astype('int32')
    topic = np.zeros((num_samples, max_length, options['max_topic_per_visit'])).astype('int32')

    icd_mask = np.zeros((num_samples, max_length, options['max_icd_per_visit'])).astype('float32')
    cpt_mask = np.zeros((num_samples, max_length, options['max_cpt_per_visit'])).astype('float32')
    med_mask = np.zeros((num_samples, max_length, options['max_med_per_visit'])).astype('float32')
    demo_mask = np.zeros((num_samples, max_length, options['demo_per_visit'])).astype('float32')
    topic_mask = np.zeros((num_samples, max_length, options['max_topic_per_visit'])).astype('float32')
            
            
    for i, patient in enumerate(patients):
        for j, visit in enumerate(patient):
            icd_ind=0
            cpt_ind=0
            med_ind=0
            demo_ind=0
            topic_ind=0
            for code in visit:
                if code<85: # code for demo
                    demo[i,j,demo_ind] = code
                    demo_mask[i,j,demo_ind] = 1.
                    demo_ind+=1
                
                elif code<284 and code>=85: # code for icd-9 
                    icd[i,j,icd_ind] = code-85
                    icd_mask[i,j,icd_ind] = 1.
                    icd_ind+=1
                
                elif code>=284 and code<2282:   #code for cpt
                    cpt[i,j,cpt_ind] = code-284
                    cpt_mask[i,j,cpt_ind] = 1.
                    cpt_ind+=1
            
                elif code>=2282 and code<2773:   #code for medication
                    med[i,j,med_ind] = code-2282
                    med_mask[i,j,med_ind] = 1.
                    med_ind+=1
                
                #code for topic feature
                else:
                    topic[i,j,topic_ind] = code-2773
                    topic_mask[i,j,topic_ind] = 1.
                    topic_ind+=1    
                    
    icd = np.transpose(icd, (1, 0, 2)) #time-major RNN
    cpt = np.transpose(cpt, (1, 0, 2))
    med = np.transpose(med, (1, 0, 2))
    demo = np.transpose(demo, (1, 0, 2))
    topic = np.transpose(topic, (1, 0, 2))
    
    icd_mask = np.transpose(icd_mask, (1, 0, 2))
    cpt_mask = np.transpose(cpt_mask, (1, 0, 2))
    med_mask = np.transpose(med_mask, (1, 0, 2))
    demo_mask = np.transpose(demo_mask, (1, 0, 2))
    topic_mask = np.transpose(topic_mask, (1, 0, 2))
    
    lengths = np.array(lengths).astype('int32')

    inputs = (icd, cpt, med, demo, topic)
    masks = (icd_mask, cpt_mask, med_mask, demo_mask, topic_mask)
    
    return inputs, masks, lengths
    
def sample_batch(seqs, labels, batch_size):
    idx = np.random.randint(0, len(seqs) - batch_size + 1)
    return seqs[idx:idx+batch_size], labels[idx:idx+batch_size]


# main function to run the model
if __name__='__main__':

    #set parameters
    input_path='data_oneyear_all'
    output_path = 'outfile_oneyear_all_attent'
    log_path = 'log_file'

    valid_losses = []
    test_losses = []
    valid_aucs = []
    test_aucs = []
    valid_aucprs = []
    test_aucprs = []
    attentions=[]


    for i in range(1):
        tf.set_random_seed(i)
        np.random.seed(i)
        for j in range(10):
            os.makedirs(output_path + '/r' + str(i) + 's' + str(j) + '/')
            tf.reset_default_graph()
            
            valid_loss, test_loss, valid_auc, test_auc, valid_aucpr, test_aucpr, attention_weights = train(
                input_path=input_path,
                output_path=output_path,
                batch_size=50,
                num_iter=2000,
                eval_period=100,
                rnn_size=256,
                output_size=1,
                learning_rate=1e-4,
                random_seed=i,
                split_seed=j,
                emb_activation='relu',
                visit_activation='relu',
                num_icd=199,
                num_cpt=1998,
                num_med=491,
                num_demo=85,
                num_topic=100,
                icd_emb_size=200,
                cpt_emb_size=200,
                med_emb_size=200,
                demo_emb_size=200,
                topic_emb_size=200,
                ehr_emb_size=256,
                visit_emb_size=256,
                max_icd_per_visit=69,
                max_cpt_per_visit=106,
                max_med_per_visit=14,
                max_topic_per_visit=30,
                demo_per_visit=2,
                regularize=1e-4,
                min_threshold=2,
                max_threshold=180)
            
            valid_losses.append(valid_loss)
            test_losses.append(test_loss)
            valid_aucs.append(valid_auc)
            test_aucs.append(test_auc)
            valid_aucprs.append(valid_aucpr)
            test_aucprs.append(test_aucpr)
            attentions.append(attention_weights)
            buf  = "valid_loss:%f, test_loss:%f, valid_auc:%f, test_auc:%f, valid_aucpr:%f, test_aucpr:%f" % (valid_loss, test_loss, valid_auc, test_auc, valid_aucpr, test_aucpr)
            with open(log_path + '.log', 'a') as outfd: outfd.write(buf + '\n')
            print(buf)
            
    buf  = "mean_valid_loss:%f, mean_test_loss:%f, mean_valid_auc:%f, mean_test_auc:%f, mean_valid_aucpr:%f, mean_test_aucpr:%f" % (np.mean(valid_losses), np.mean(test_losses), np.mean(valid_aucs), np.mean(test_aucs), np.mean(valid_aucprs), np.mean(test_aucprs))
    with open(log_path + '.log', 'a') as outfd: outfd.write(buf + '\n')
    print(buf)


    print("Test ROCAUC:",test_aucs)
    print("Test PRAUC:",test_aucprs)
    print("attention weights:",attentions)
    print("mean Test ROCAUC:",np.mean(test_aucs))
    print("mean Test PRCAUC:",np.mean(test_aucprs))

    # print and save attention weights 
    res=[0,0,0,0,0]
    for array in attentions:
        res+=array

    print(res/10)   
    save_attent=np.squeeze(np.array(attentions))
    np.savetxt(output_path+'.csv',save_attent,delimiter=',')

