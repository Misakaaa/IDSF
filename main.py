import torch.optim as optim
from data_utils import *
from model import SDEN
from sklearn_crfsuite import metrics
import argparse
from data import T
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluation(model, dev_data):

    model.eval()
    index2word = {v: k for k, v in model.vocab.items()}
    index2slot = {v: k for k, v in model.slot_vocab.items()}
    preds = []
    labels = []
    hits = 0
    len_slot = 0
    fp = open("data/rea-labels", "w", encoding='utf-8')
    fr = open("data/pre-labels", "w", encoding='utf-8')

    ff1 = open("data/labels-tokens1", "w", encoding='utf-8')
    ff2 = open("data/labels-tokens2", "w", encoding='utf-8')

    current = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader(dev_data, 32, True)):
            h, c, slot, intent = pad_to_batch(batch, model.vocab, model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)

            for s in c:
                a = [index2word[i] for i in s.tolist()]
                b = ' '.join(a).replace('<pad>', '').strip()
                current.append(b)

            for s in slot:
                for i in s.tolist():
                    ff1.write(str(i) + ' ')
                ff1.write('\n')

                a = [index2slot[i] for i in s.tolist()]
                len_slot = len(a)

                b = ' '.join(a)

                fp.write(b + '\n')

            intent = intent.to(device)
            slot_p, intent_p = model(h, c)
            l = slot_p.max(1)[1]
            n = len_slot
            m = [l[i:i + n] for i in range(0, len(l), n)]
            for s in m:
                for i in s.tolist():
                    ff2.write(str(i) + ' ')
                ff2.write('\n')

                a = [index2slot[i] for i in s.tolist()]
                b = ' '.join(a)
                fr.write(b + '\n')

            preds.extend([index2slot[i] for i in slot_p.max(1)[1].tolist()])
            labels.extend([index2slot[i] for i in slot.view(-1).tolist()])
            hits += torch.eq(intent_p.max(1)[1], intent.view(-1)).sum().item()

    fp.close()
    fr.close()
    ff1.close()
    ff2.close()
    print(hits / len(dev_data))

    sorted_labels = sorted(
        list(set(labels) - {'O', '<pad>'}),
        key=lambda name: (name[1:], name[0])
    )
    # this is because sklearn_crfsuite.metrics function flatten inputs
    preds = [[y] for y in preds]
    labels = [[y] for y in labels]
    print(metrics.flat_classification_report(
        labels, preds, labels=sorted_labels, digits=3
    ))


def save_checkpoints(model, optimizer, epoch, model_dir, name):
    ckp_path = os.path.join(model_dir, name)
    checkpoint = {
        'model': model.state_dict(),
        'vocab': model.vocab,
        'slot_vocab': model.slot_vocab,
        'intent_vocab': model.intent_vocab,
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, ckp_path)
    torch.save(checkpoint, os.path.join(model_dir, 'idsf_lastest.pth'))
    print("Model saved!")


def load_checkpoints(model, optimizer, model_dir, name='idsf_lastest.pth'):
    ckp_path = os.path.join(model_dir, name)
    try:
        print('Load checkpoint %s' % ckp_path)
        obj = torch.load(ckp_path)
    except FileNotFoundError:
        print('No checkpoint %s!!' % ckp_path)
        return False, None, None, None, None
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    vocab = obj['vocab']
    slot_vocab = obj['slot_vocab']
    intent_vocab = obj['intent_vocab']
    epoch = obj['epoch']

    return True, epoch, vocab, slot_vocab, intent_vocab


def train(config):
    train_data, word2index, slot2index, intent2index = prepare_dataset('data/train.iob')
    dev_data = prepare_dataset('data/dev.iob', (word2index, slot2index, intent2index))

    print('vocab_size : ',len(word2index))
    print('slot2index : ',len(slot2index))
    print('intent2index : ',len(intent2index))

    model = SDEN(len(word2index), config.embed_size, config.hidden_size, \
                 len(slot2index), len(intent2index), word2index['<pad>'])
    model.to(device)
    model.vocab = word2index
    model.slot_vocab = slot2index
    model.intent_vocab = intent2index

    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    intent_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[config.epochs // 4, config.epochs // 2],
                                               optimizer=optimizer)

    epoches = range(config.epochs)

    # _, start, vocab, slot_vocab, intent_vocab = load_checkpoints(model, optimizer, config.save_path)
    #
    # if _ is True:
    #    epoches = range(start, config.epochs)

    for epoch in epoches:
        losses = []
        model.train()
        scheduler.step()
        for i, batch in enumerate(data_loader(train_data, config.batch_size, True)):
            # print(batch)
            h, c, slot, intent = pad_to_batch(batch, model.vocab, model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            model.zero_grad()
            slot_p, intent_p = model(h, c)

            loss_s = slot_loss_function(slot_p, slot.view(-1))
            loss_i = intent_loss_function(intent_p, intent.view(-1))
            loss = loss_s + loss_i
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f" % \
                      (epoch, config.epochs, i, len(train_data) // config.batch_size, np.mean(losses)))
                losses = []

        evaluation(model, dev_data)
        T.metrics_result()
        save_checkpoints(model, optimizer, epoch, config.save_path, name='idsf_%s.pth' % (epoch + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='save_path')

    config = parser.parse_args()

    train(config)
