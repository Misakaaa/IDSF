import numpy as np
import numpy.ma as ma
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics

def get_data_from_sequence_batch(true_batch, pred_batch, padding_token):
    """从序列的batch中提取数据：
    [[3,1,2,0,0,0],[5,2,1,4,0,0]] -> [3,1,2,5,2,1,4]"""
    true_ma = ma.masked_equal(true_batch, padding_token)
    pred_ma = ma.masked_array(pred_batch, true_ma.mask)
    true_ma = true_ma.flatten()
    pred_ma = pred_ma.flatten()
    true_ma = true_ma[~true_ma.mask]
    pred_ma = pred_ma[~pred_ma.mask]
    return true_ma, pred_ma


def f1_for_sequence_batch(true_batch, pred_batch, average="micro", padding_token=0):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, padding_token)
    # labels = list(set(true))
    labels = sorted(
        list(set(true)),
        key=lambda name: (name[1:], name[0])
    )
    return f1_score(true, pred, labels=labels, average=average)

def accuracy_score(true_data, pred_data, true_length=None):
    true_data = np.array(true_data)
    pred_data = np.array(pred_data)
    assert true_data.shape == pred_data.shape
    if true_length is not None:
        val_num = np.sum(true_length)
        # print(val_num)
        assert val_num != 0
        res = 0
        for i in range(true_data.shape[0]):
            # print(true_data[i, :true_length[i]])
            # print(pred_data[i, :true_length[i]])
            # if(str(true_data[i, :true_length[i]]) == str(pred_data[i, :true_length[i]])):
            #     res+=1
            res += np.sum(true_data[i, :true_length[i]] == pred_data[i, :true_length[i]])
            # print(res)
    else:
        val_num = np.prod(true_data.shape)
        assert val_num != 0
        res = np.sum(true_data == pred_data)
    res /= float(val_num)
    return res

def sum():
    num = 0
    len_slot = []
    slot_rea = []
    slot_pre = []
    fp = open("data/rea-labels", "r", encoding='utf-8')
    sentences = fp.readlines()
    for sen in sentences:
        # print(sen)
        slot = sen.replace('\n','').split(' ')
        # print(slot)
        while(slot[-1]=='<pad>'or slot[-1]=='<pad>\n'):
            del slot[-1]

        q = ' '.join(slot)
        len_slot.append(len(slot))
        slot_rea.append(q)

    #     print(q)
    # print(len_slot)

    fr = open("data/pre-labels", "r", encoding='utf-8')
    sen2 = fr.readlines()
    for i in range(len(sen2)):
        slot2 = sen2[i].replace('\n','').split(' ')

        q = ' '.join(slot2[0:len_slot[i]])
        # print(q)
        slot_pre.append(q)


    for i in range(len(slot_pre)):
        if(slot_pre[i] == slot_rea[i]):
            num+=1
    print(num)

    print(num/len(slot_rea))

    count = 0
    total = 0
    for i in range(len(slot_pre)):

        a = slot_pre[i].split(' ')
        b = slot_rea[i].split(' ')
        total += len(a)
        for j in range(len(a)):
            if(a[j]==b[j]):
                count+=1
    print(count/total)

    # f1= open("data/labels-tokens1", "r", encoding='utf-8')
    # f2 = open("data/labels-tokens2", "r", encoding='utf-8')
    #
    # q1 = []
    # q2 = []
    #
    # for i in f1.readlines():
    #     j = i.split(' ')
    #     del j[-1]
    #     # print(j)
    #     a = list(map(lambda x:int(x),j))
    #     # print(a)
    #     q1.append(a)
        # a = []
        # for j in i:
        #     print(j)
        #     if (j != ' ' or '\n'):
        #         a.append(ord(j))
        # print(a)

    # for i in f2.readlines():
    #     j = i.split(' ')
    #     del j[-1]
    #     # print(j)
    #     a = list(map(lambda x: int(x), j))
    #     # print(a)
    #     q2.append(a)
    # # print(q1)
    # print("F1--",f1_for_sequence_batch(q1, q2))
    # print("accuracy--",accuracy_score(q1, q2, len_slot))
# sum()



def geshihua(rea_labels,pre_labels):
    fr= open(rea_labels,'r',encoding="utf-8")
    fp=open(pre_labels,'r',encoding='utf-8')
    fr_re = open("data/rea-labels1",'w',encoding="utf-8")
    fp_re = open("data/pre-labels1",'w',encoding="utf-8")
    fr_lists = fr.readlines()
    fp_lists = fp.readlines()
    for i,j in zip(fr_lists,fp_lists):
         i=i.split(" ")
         j=j.split(" ")

         for r_label,p_label in zip(i,j):
             if r_label!='<pad>' and r_label!='<pad>\n' :
                 fr_re.write(r_label+"\n")
                 fp_re.write(p_label+'\n')

def metrics_result( ):
    # real_label_geshi = "rea-labels"
    # pre_label_geshi = 'pre-labels'
    geshihua("data/rea-labels",'data/pre-labels')

    fr_re = open("data/rea-labels1",'r',encoding="utf-8")
    fp_re = open('data/pre-labels1','r',encoding="utf-8")
    frre_lists = fr_re.readlines()
    fpre_lists = fp_re.readlines()
    print( '精度:{0:.8f}'.format(metrics.precision_score(frre_lists, fpre_lists, average='weighted')))
    print ('召回:{0:0.8f}'.format(metrics.recall_score(frre_lists, fpre_lists, average='weighted')))
    print ('f1-score:{0:.8f}'.format(metrics.f1_score(frre_lists, fpre_lists, average='weighted')))



# real_labels = "rea-labels1"
# pre_labels = 'pre-labels1'
# metrics_result(real_labels,pre_labels)

