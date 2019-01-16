import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_utils import *
from model import SDEN
from sklearn_crfsuite import metrics
import argparse
from data import T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluation(model,dev_data):
    # print(dev_data)   #[tensor([[ 2]])], tensor([[ 71,  16,  28,  70,  68,  70,  61,  23,  60,   6,  73,  72,
    #       53,  20,  45,  29,  70,  19,  25]]), tensor([[ 1,  1,  1,  1,  1,  1,  1,  4,  6,  1,  1,  1,  1,  1,
    #       1,  2,  1,  1,  1]]), tensor([[ 0]])]
    model.eval()
    index2word = {v: k for k, v in model.vocab.items()}
    index2slot = {v:k for k,v in model.slot_vocab.items()}
    index2intent = {v:k for k,v in model.intent_vocab.items()}
    preds=[]
    labels=[]
    hits=0
    len_slot=0
    fp = open("data/rea-labels", "w", encoding='utf-8')
    fr = open("data/pre-labels", "w", encoding='utf-8')

    ff1 = open("data/labels-tokens1", "w", encoding='utf-8')
    ff2= open("data/labels-tokens2", "w", encoding='utf-8')

    current = []
    intent_real =[]
    intent_pre = []
    # label_real = []
    # label_pre = []

    # f1 = open("data/train1.iob", "r", encoding='utf-8')
    f2 = open("data/dev-intent", "w", encoding='utf-8')

    with torch.no_grad():
        for i,batch in enumerate(data_loader(dev_data,32,True)):
            # print(batch)
            h,c,slot,intent = pad_to_batch(batch,model.vocab,model.slot_vocab)
            # print(c)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            # print(slot)

            for s in c:

                a = [index2word[i] for i in s.tolist()]
                b = ' '.join(a).replace('<pad>','').strip()
                current.append(b)
                # print(b)

            for s in slot:
                # print(s)
                for i in s.tolist():
                    ff1.write(str(i)+' ')
                ff1.write('\n')


                a = [index2slot[i] for i in s.tolist()]
                len_slot = len(a)
                # while(a[-1]== '<pad>'):
                #     del a[-1]
                # print([index2slot[i] for i in s.tolist()])
                b = ' '.join(a)
                # label_real.append(b)
                # print(b)
                fp.write(b+'\n')



            intent = intent.to(device)
            slot_p, intent_p = model(h,c)
            # print(slot_p)


            l = slot_p.max(1)[1]
            # print(slot_p.max(1)[1])
            n=len_slot
            m = [l[i:i + n] for i in range(0, len(l), n)]
            # print(n)
            for s in m:
                for i in s.tolist():
                    ff2.write(str(i) + ' ')
                ff2.write('\n')

                a = [index2slot[i] for i in s.tolist()]
                b = ' '.join(a)
                # print(b)
                fr.write(b+'\n')





            # print(slot_p)
            # intent_pre.extend([index2intent[i] for i in intent_p.max(1)[1].tolist()])
            preds.extend([index2slot[i] for i in slot_p.max(1)[1].tolist()])
            labels.extend([index2slot[i] for i in slot.view(-1).tolist()])
            # intent_real.extend([index2intent[i] for i in intent.view(-1).tolist()])
            hits+=torch.eq(intent_p.max(1)[1],intent.view(-1)).sum().item()

        # i = 0
        # for data in current:
        #     f2.write(data+"||||"+intent_real[i]+"||||"+intent_pre[i]+'\n')
        #     i += 1
        #
        # f2.close()


    # print(intents)
    fp.close()
    fr.close()
    ff1.close()
    ff2.close()
    print(hits/len(dev_data))
    
    sorted_labels = sorted(
    list(set(labels) - {'O','<pad>'}),
    key=lambda name: (name[1:], name[0])
    )
    # sorted_labels = sorted(
    # list(set(labels) - {'<pad>'}),
    # key=lambda name: (name[1:], name[0])
    # )
    #
    # this is because sklearn_crfsuite.metrics function flatten inputs
    preds = [[y] for y in preds] 
    labels = [[y] for y in labels]


    # fp = open("data/pre-labels","w",encoding='utf-8')
    # for pre in preds:
    #
    #     print(pre)
    #     fp.write(str(pre)+'\n')
    #
    #
    # fr = open("data/rea-labels","w",encoding='utf-8')
    # for label in labels:
    #     print(label)
    #     fr.write(str(label)+'\n')
    
    print(metrics.flat_classification_report(
    labels, preds, labels = sorted_labels, digits=3
    ))

def save(model,config):
    checkpoint = {
                'model': model.state_dict(),
                'vocab': model.vocab,
                'slot_vocab' : model.slot_vocab,
                'intent_vocab' : model.intent_vocab,
                'config' : config,
            }
    torch.save(checkpoint,config.save_path)
    print("Model saved!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=5,
                        help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning_rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout')
    parser.add_argument('--embed_size', type=int, default=100,
                        help='embed_size')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='hidden_size')
    parser.add_argument('--save_path', type=str, default='weight/model.pkl',
                        help='save_path')
    
    config = parser.parse_args()
    
    train_data, word2index, slot2index, intent2index = prepare_dataset('data/train.iob')
    dev_data = prepare_dataset('data/dev.iob',(word2index,slot2index,intent2index))
    model = SDEN(len(word2index),config.embed_size,config.hidden_size,\
                 len(slot2index),len(intent2index),word2index['<pad>'])
    model.to(device)
    model.vocab = word2index
    model.slot_vocab = slot2index
    model.intent_vocab = intent2index
    
    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    intent_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,milestones=[config.epochs//4,config.epochs//2],optimizer=optimizer)
    
    model.train()
    for epoch in range(config.epochs):
        losses=[]
        scheduler.step()
        for i,batch in enumerate(data_loader(train_data,config.batch_size,True)):
            # print(batch)
            h,c,slot,intent = pad_to_batch(batch,model.vocab,model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            model.zero_grad()
            slot_p, intent_p = model(h,c)

            loss_s = slot_loss_function(slot_p,slot.view(-1))
            loss_i = intent_loss_function(intent_p,intent.view(-1))
            loss = loss_s + loss_i
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f" % \
                      (epoch,config.epochs,i,len(train_data)//config.batch_size,np.mean(losses)))
                losses=[]
                      
        evaluation(model,dev_data)
        # T.sum()
        T.metrics_result()
    save(model,config)