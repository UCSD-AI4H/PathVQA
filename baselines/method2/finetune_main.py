
import argparse
import builtins
import os
import pickle
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pvqa', help='vqa or pvqa')
    parser.add_argument('--epochs', type=int, default=13)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--cos', action='store_true', help='cosine learning rate')

    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    parser.add_argument('--train', type=str, default='train')
    parser.add_argument('--val', type=str, default='val')
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
        train_dset = PVQAFeatureDataset(args.train, dictionary, adaptive=False)
        val_dset = PVQAFeatureDataset(args.val, dictionary, adaptive=False)
        w_emb_path = 'data/pvqa/glove_pvqa_300d.npy'
    else:
        raise Exception('%s not implemented yet' % args.task)

    if args.multiGPUs:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset)
    else:
        train_sampler = None

    if args.task == 'pvqa':
        train_loader = DataLoader(train_dset, args.batch_size, shuffle=(train_sampler is None),
                                  num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        eval_loader = DataLoader(val_dset, args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    # prepare model

    model = BanModel(ntoken=train_dset.dictionary.ntoken,
                     num_ans_candidates=train_dset.num_ans_candidates,
                     num_hid=args.num_hid, v_dim=train_dset.v_dim,
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

        for name, param in model.named_parameters():
            if name in model_data_sd:
                param.data = model_data_sd[name]

        # optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        # optimizer.load_state_dict(model_data.get('optimizer_state', model_data))
        args.start_epoch = model_data['epoch'] + 1

    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))

    best_eval_score = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.multiGPUs:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)

        eval_score = evaluate(eval_loader, model, args)

        with open(os.path.join(args.output, 'log.log'), 'a') as f:
            f.write(str(datetime.datetime.now()))
            f.write('epoch=%d' % epoch)
            f.write('eval_score=%.4f' % eval_score)

        print('eval_score=', eval_score)
        print('best eval_score = ', best_eval_score)

        if not args.multiGPUs or (args.multiGPUs and args.gpu == 0):
            if eval_score > best_eval_score:
                model_path = os.path.join(args.output, 'model_best.pth')
                utils.save_model(model_path, model, epoch, optimizer)
                best_eval_score = eval_score


def train(train_loader: DataLoader, model, optimizer, epoch, args):
    model.train()
    total_loss = 0.0
    train_score = 0
    total_norm = 0
    count_norm = 0
    grad_clip = .25
    for (v, b, q, a) in tqdm(train_loader):
        v = v.cuda(args.gpu)
        b = b.cuda(args.gpu)
        q = q.cuda(args.gpu)
        a = a.cuda(args.gpu)

        pred, att = model(v, b, q, a)
        loss = instance_bce_with_logits(pred, a)
        optimizer.zero_grad()
        loss.backward()

        total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        count_norm += 1

        total_loss += loss.item()

        optimizer.step()

        batch_score = compute_score_with_logits(pred, a.data).sum()
        train_score += batch_score.item()
    total_loss /= len(train_loader)
    train_score /= len(train_loader.dataset)
    print('total_loss=', total_loss, '; train_score=', train_score)


@torch.no_grad()
def evaluate(eval_loader: DataLoader, model, args):
    model.eval()
    val_score = 0

    scores = []
    for (v, b, q, a) in tqdm(eval_loader):
        v = v.cuda(args.gpu)
        b = b.cuda(args.gpu)
        q = q.cuda(args.gpu)
        a = a.cuda(args.gpu)

        pred, att = model(v, b, q, a)

        base_scores = compute_score_with_logits(pred, a.data)
        batch_score = base_scores.sum()
        scores.append(base_scores.detach().cpu().numpy().sum(-1))
        val_score += batch_score.item()

    val_score /= len(eval_loader.dataset)
    scores = np.concatenate(scores).ravel()
    return val_score


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1. + np.cos(np.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.world_size = torch.cuda.device_count()
    print('Found %d devices, world_size=%d' % (args.world_size, args.world_size))
    args.multiGPUs = args.world_size > 1
    if not os.path.exists(args.output):
        utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'args.txt'))
    logger.write(args.__repr__())

    if args.multiGPUs:
        mp.spawn(main_worker,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True)
    else:
        main_worker(args.gpu, args)


if __name__ == '__main__':
    main()
