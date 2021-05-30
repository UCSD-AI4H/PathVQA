import argparse
import builtins
import os
import pickle
from sklearn.metrics import f1_score
import datetime

import torch

from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from dataset import Dictionary
# , VQAFeatureDataset, VisualGenomeFeatureDataset, Flickr30kFeatureDataset
from dataset import PVQAFeatureDataset, PretrainDataset, _load_dataset_pvqa
from dataset import question_types, get_q_type
from modeling import BanModel, instance_bce_with_logits, compute_score_with_logits

from tqdm import tqdm
import utils
from dataset import tfidf_from_questions
from nltk.translate.bleu_score import sentence_bleu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pvqa', help='vqa or pvqa')

    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--img_v', type=str, default='', help='pvqa img feature version')
    parser.add_argument('--use_vg', action='store_true', help='use visual genome dataset to train?')
    parser.add_argument('--tfidf', action='store_false', help='tfidf word embedding?')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/ban')
    parser.add_argument('--batch_size', type=int, default=256)  # batch_size per gpu
    parser.add_argument('--seed', type=int, default=1204, help='random seed')

    parser.add_argument('--qa_bl', action='store_true', help='qa without image for baseline')

    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use in single-gpu mode')
    parser.add_argument('--workers', type=int, default=0)

    args = parser.parse_args()
    return args


# DDP
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '51243'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main_worker(gpu, args):
    args.gpu = gpu

    if args.multiGPUs and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))

    if args.multiGPUs:
        args.rank = gpu
        setup(args.rank, args.world_size)

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            args.workers = int((args.workers + args.world_size - 1) / args.world_size)

    # prepare data
    if args.task == 'pvqa':
        dict_path = 'data/pvqa/pvqa_dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        test_dset = PVQAFeatureDataset(args.data_split, dictionary, adaptive=False)
        w_emb_path = 'data/pvqa/glove_pvqa_300d.npy'
    else:
        raise Exception('%s not implemented yet' % args.task)

    if args.task == 'pvqa':
        test_loader = DataLoader(test_dset, args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    # prepare model

    model = BanModel(ntoken=test_dset.dictionary.ntoken,
                     num_ans_candidates=test_dset.num_ans_candidates,
                     num_hid=args.num_hid, v_dim=test_dset.v_dim,
                     op=args.op,
                     gamma=args.gamma, qa_bl=args.qa_bl)

    tfidf = None
    weights = None
    model.w_emb.init_embedding(w_emb_path, tfidf, weights)

    if args.multiGPUs:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.workers = int((args.workers + args.world_size - 1) / args.world_size)
            model = DDP(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = DDP(model)
    else:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    # load snapshot
    if args.input is not None:
        print('#8')
        print('loading %s' % args.input)
        if args.gpu is None:
            model_data = torch.load(args.input)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            model_data = torch.load(args.input, map_location=loc)
        model_data_sd = model_data.get('model_state', model_data)

        model.load_state_dict(model_data_sd)

    res = evaluate(test_loader, model, args)
    eval_score = res['eval_score']
    preds = res['preds']
    anss = res['anss']
    b_scores = []
    b_scores_1 = []
    b_scores_2 = []
    b_scores_3 = []
    assert len(preds) == len(anss), 'len(preds)=%d, len(anss)=%d' % (len(preds), len(anss))
    for i in range(len(preds)):
        pred_ans = test_dset.label2ans[preds[i]]
        gt_ans = test_dset.entries[i]['ans_sent']
        b_score = sentence_bleu(references=[str(gt_ans).lower().split()],
                                hypothesis=str(pred_ans).lower().split())
        b_score_1 = sentence_bleu(references=[str(gt_ans).lower().split()],
                                hypothesis=str(pred_ans).lower().split(), weights=(1, 0, 0, 0))
        b_score_2 = sentence_bleu(references=[str(gt_ans).lower().split()],
                                hypothesis=str(pred_ans).lower().split(), weights=(0, 1, 0, 0))
        b_score_3 = sentence_bleu(references=[str(gt_ans).lower().split()],
                                hypothesis=str(pred_ans).lower().split(), weights=(0, 0, 1, 0))
        b_scores.append(b_score)
        b_scores_1.append(b_score_1)
        b_scores_2.append(b_score_2)
        b_scores_3.append(b_score_3)

    b_score_m = np.mean(b_scores)
    b_score_m_1 = np.mean(b_scores_1)
    b_score_m_2 = np.mean(b_scores_2)
    b_score_m_3 = np.mean(b_scores_3)
    b_score_info = 'bleu score=%.4f\n' % b_score_m
    b_score_info_1 = 'bleu1 score=%.4f\n' % b_score_m_1
    b_score_info_2 = 'bleu2 score=%.4f\n' % b_score_m_2
    b_score_info_3 = 'bleu3 score=%.4f' % b_score_m_3
    print(b_score_info)
    print(b_score_info_1)
    print(b_score_info_2)
    print(b_score_info_3)
    with open(os.path.join(args.output, 'type_result.txt'), 'a') as f:
        f.write(b_score_info)
        f.write(b_score_info_1)
        f.write(b_score_info_2)
        f.write(b_score_info_3)


@torch.no_grad()
def evaluate(eval_loader: DataLoader, model, args):
    model.eval()
    val_score = 0

    scores = []
    preds = []
    anss = []
    for (v, b, q, a) in tqdm(eval_loader):
        v = v.cuda(args.gpu)
        b = b.cuda(args.gpu)
        q = q.cuda(args.gpu)
        a = a.cuda(args.gpu)

        pred, att = model(v, b, q, a)
        #print('pred.shape=', pred.shape)
        preds.append(np.argmax(pred.detach().cpu().numpy(), axis=1))
        #print('preds[-1].shape=', preds[-1].shape)
        anss.append(a.detach().cpu().numpy())
        #print('a.shape=', anss[-1].shape)
        base_scores = compute_score_with_logits(pred, a.data)
        batch_score = base_scores.sum()
        scores.append(base_scores.detach().cpu().numpy().sum(-1))
        val_score += batch_score.item()

    val_score /= len(eval_loader.dataset)
    scores = np.concatenate(scores).ravel()
    preds = np.concatenate(preds).reshape(-1)
    anss = np.concatenate(anss, axis=0)
    #print('anss.shape=',anss.shape)
    anss = np.concatenate((anss, 0.2 * np.ones((anss.shape[0], 1))), axis=1)
    #print('anss.shape=', anss.shape)
    anss = np.argmax(anss, axis=1)
    #print('anss.shape=', anss.shape)
    anss = np.ravel(anss)
    #print('anss.shape=',anss.shape)
    #print('preds.shape=', preds.shape)
    f1_val = f1_score(anss, preds, average='macro')
    if args.task == 'pvqa':
        dataroot = 'data/pvqa'
        name = 'test'
        ans2label_path = os.path.join(dataroot, 'qas', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'qas', 'trainval_label2ans.pkl')
        ans2label = pickle.load(open(ans2label_path, 'rb'))
        label2ans = pickle.load(open(label2ans_path, 'rb'))
        img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s_img_id2idx.pkl' % name), 'rb'))
        entries = _load_dataset_pvqa(dataroot, name, img_id2idx, label2ans, ans2label)
        # print('entries: ', len(entries))
        qtype_score = {qtype: 0. for qtype in question_types}
        qtype_cnt = {qtype: 0 for qtype in question_types}
        for i in range(len(entries)):
            entry = entries[i]
            qtype = get_q_type(entry['question'])
            qtype_cnt[qtype] += 1
            qtype_score[qtype] += scores[i]

        with open(os.path.join(args.output, 'type_result.txt'), 'w') as f:
            info = str(datetime.datetime.now())
            info += 'Overall score=%.4f\n' % val_score
            info += 'F1 score=%.4f\n' % f1_val
            for t in question_types:
                if qtype_cnt[t] > 0:
                    info += 'type %s:\tcnt=%d\tacc=%.4f\n' % (t, qtype_cnt[t], qtype_score[t] / qtype_cnt[t])
            f.write(info)
            print(info)
        return {'eval_score': val_score, 'preds': preds, 'anss': anss}
    else:
        raise Exception('Not implemented other than PVQA')


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.world_size = torch.cuda.device_count()
    print('Found %d devices, world_size=%d' % (args.world_size, args.world_size))
    args.multiGPUs = args.world_size > 1

    if args.multiGPUs:
        mp.spawn(main_worker,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True)
    else:
        main_worker(args.gpu, args)


if __name__ == '__main__':
    main()
