import os
import argparse
import builtins

import torch

from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from dataset import Dictionary
# , VQAFeatureDataset, VisualGenomeFeatureDataset, Flickr30kFeatureDataset

from dataset import PVQAFeatureDataset, PretrainDataset
from modeling import BanPreModel

from tqdm import tqdm

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pvqa', help='vqa or pvqa')
    parser.add_argument('--epochs', type=int, default=13)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--cos', action='store_true', help='cosine learning rate')
    # parser.add_argument('--schedule')

    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    parser.add_argument('--train', type=str, default='train')
    parser.add_argument('--val', type=str, default='')
    parser.add_argument('--img_v', type=str, default='', help='pvqa img feature version')
    parser.add_argument('--use_vg', action='store_true', help='use visual genome dataset to train?')
    parser.add_argument('--tfidf', action='store_false', help='tfidf word embedding?')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/ban_pre')
    parser.add_argument('--batch_size', type=int, default=256)  # batch_size per gpu
    parser.add_argument('--seed', type=int, default=1204, help='random seed')

    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use in single-gpu mode')
    parser.add_argument('--pretrain_tasks', type=str, default='', help='pretrain tasks, separated by ,')
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
    print('prepare dataset')
    if args.task == 'pvqa':
        dict_path = 'data/pvqa/pvqa_dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        train_dset = PVQAFeatureDataset(args.train, dictionary, adaptive=False)
        train_pre_dset = PretrainDataset(train_dset, args.task)
        if args.val:
            val_dset = PVQAFeatureDataset(args.val, dictionary, adaptive=False)
            val_pre_dset = PretrainDataset(val_dset, args.task)
        w_emb_path = 'data/pvqa/glove_pvqa_300d.npy'
    else:
        raise Exception('%s not implemented yet' % args.task)

    if args.multiGPUs:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_pre_dset)
    else:
        train_sampler = None

    if args.task == 'pvqa':
        train_loader = DataLoader(train_pre_dset, args.batch_size, shuffle=(train_sampler is None),
                                  num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        # eval_loader = DataLoader(val_pre_dset, args.batch_size, shuffle=False,
        #                          num_workers=args.workers, pin_memory=True)

    # prepare model
    print('building BanPreModel')
    model = BanPreModel(ntoken=train_dset.dictionary.ntoken,
                        num_ans_candidates=train_dset.num_ans_candidates,
                        num_hid=args.num_hid, v_dim=train_dset.v_dim,
                        op=args.op,
                        gamma=args.gamma,
                        pretrain_tasks=args.pretrain_tasks.split(','))

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
        print('loading %s' % args.input)
        if args.gpu is None:
            model_data = torch.load(args.input)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            model_data = torch.load(args.input, map_location=loc)
        model.load_state_dict(model_data.get('model_state', model_data))
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer.load_state_dict(model_data.get('optimizer_state', model_data))
        args.start_epoch = model_data['epoch'] + 1

    else:
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(args.start_epoch, args.epochs):
        print('training epoch: %d' % epoch)
        if args.multiGPUs:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)

        if not args.multiGPUs or (args.multiGPUs and args.gpu == 0):
            model_path = os.path.join(args.output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optimizer)


def train(train_loader, model, optimizer, epoch, args):
    model.train()
    total_loss = 0.0
    for examples in tqdm(train_loader):
        (uid, question, (feats, spatials),
         vq_matched, match_question, va_matched, answer_rps, label,
         ans_valid, ans_rps_valid) = examples
        uid = uid.cuda(args.gpu)
        question = question.cuda(args.gpu)
        feats = feats.cuda(args.gpu)
        spatials = spatials.cuda(args.gpu)
        vq_matched = vq_matched.cuda(args.gpu)
        match_question = match_question.cuda(args.gpu)
        va_matched = va_matched.cuda(args.gpu)
        answer_rps = answer_rps.cuda(args.gpu)
        label = label.cuda(args.gpu)
        ans_valid = ans_valid.cuda(args.gpu)
        ans_rps_valid = ans_rps_valid.cuda(args.gpu)

        # print('question.shape', question.shape)
        # print('feats.shape', feats.shape)
        # print('spatials.shape', spatials.shape)
        # print('vq_matched.shape', vq_matched.shape)
        # print('match_question.shape', match_question.shape)
        # print('va_matched.shape', va_matched.shape)
        # print('answer_rps.shape', answer_rps.shape)
        # print('label.shape', label.shape)

        loss = model(question, feats, spatials,
                     vq_matched, match_question, va_matched, answer_rps, label,
                     ans_valid, ans_rps_valid)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('total_loss=', total_loss)


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

    args.scheduler = [2, 6, 10]   # not used

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.world_size = torch.cuda.device_count()
    print('Found %d devices, world_size=%d' % (args.world_size, args.world_size))
    args.multiGPUs = args.world_size > 1

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'args.txt'))
    logger.write(args.__repr__())

    if args.multiGPUs:
        mp.spawn(main_worker,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True)
        cleanup()
    else:
        main_worker(0, args)


if __name__ == '__main__':
    main()
