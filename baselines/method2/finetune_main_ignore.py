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
import torch.nn as nn
import numpy as np

from dataset import Dictionary
# , VQAFeatureDataset, VisualGenomeFeatureDataset, Flickr30kFeatureDataset
from dataset import PVQAFeatureDataset, PretrainDataset, _load_dataset_pvqa
from dataset import question_types, get_q_type
from modeling import BanModel, instance_bce_with_logits, compute_score_with_logits

import copy
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
        train_loader = DataLoader(train_dset, args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        eval_loader = DataLoader(val_dset, args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    # prepare model

    model = BanModel(ntoken=train_dset.dictionary.ntoken,
                     num_ans_candidates=train_dset.num_ans_candidates,
                     num_hid=args.num_hid, v_dim=train_dset.v_dim,
                     op=args.op,
                     gamma=args.gamma)

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
        train(train_loader, eval_loader, train_dset, model, optimizer, epoch, args)

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

                

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = net
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, w_optim, Likelihood, step, args):
        """
        Compute unrolled weight w' (virtual step)
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        dataIndex = len(trn_y)+step*args.batch_size
        
        v, b, q = trn_X
        a = trn_y
           
        # forward
        pred, att = self.v_net(v, b, q, a)
        
        
        print('len bacth', a.size())
        print('len bacth2', pred.size())
        print('a size', v.size())
        
        # sigmoid loss
        first = torch.sigmoid(Likelihood[step*args.batch_size:dataIndex])
        second = instance_bce_with_logits(pred, a, reduction='none').mean(1).cuda() 
        print(first.size())
        print(second.size())
        lossup = torch.dot(first, second)
        lossdiv =(torch.sigmoid(Likelihood[step*args.batch_size:dataIndex]).sum())
        loss = lossup/lossdiv
        
#         loss = torch.dot(torch.sigmoid(Likelihood[step*batch_size:dataIndex]), ignore_crit(logits, trn_y))/(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        
        # compute gradient of train loss towards likelihhod
        loss.backward()

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw in zip(self.net.parameters(), self.v_net.parameters()):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                
                if w.grad is not None:
                    vw.copy_(w - args.lr * (m + w.grad + self.w_weight_decay*w))


    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, w_optim, Likelihood, Likelihood_optim, step, args):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        crit = nn.CrossEntropyLoss().cuda()
        
        xi = 0.01
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, w_optim, Likelihood, step, args)
        
        
        vv, vb, vq = val_X
        va = trn_y
        # calc val prediction
        pred, att = self.v_net(vv, vb, vq, va)
        # calc unrolled validation loss
        loss = instance_bce_with_logits(pred, va) # L_val(w`)
        
        # compute gradient of validation loss towards weights
        v_weights = tuple(self.v_net.parameters())
        # some weights not used return none
        
        dw = []
        for w in v_weights:  
            if w.requires_grad:
                dw.append(torch.autograd.grad(loss, w, allow_unused=True, retain_graph=True))
            else:
                dw.append(None)
        hessian = self.compute_hessian(dw, trn_X, trn_y, Likelihood, args.batch_size, step)

        
        Likelihood_optim.zero_grad()
        # update final gradient = - xi*hessian
#         with torch.no_grad():
#             for likelihood, h in zip(Likelihood, hessian):
#                 print(len(hessian))
#                 likelihood.grad = - xi*h
        with torch.no_grad():
            Likelihood.grad = - xi*hessian[0]         
        Likelihood_optim.step()
        return Likelihood, Likelihood_optim, loss

    def compute_hessian(self, dw, trn_X, trn_y, Likelihood, batch_size, step):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
                
        norm = torch.cat([w[0].view(-1) for w in dw if ((w != None) and (w[0] != None))]).norm()
        
        eps = 0.01 / norm
        
        v, b, q = trn_X
        a = trn_y
        
        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                if d!= None and d[0] != None:
                    pp = eps * d[0]
                    p += eps * d[0]
        
        
        # forward & calc loss
        dataIndex = len(trn_y)+step*batch_size 
        # forward
        logits, att = self.net(v, b, q, a)
        # sigmoid loss
        first = torch.sigmoid(Likelihood[step*batch_size:dataIndex])
        second = instance_bce_with_logits(logits, a, reduction='none').mean(1).cuda() 
        lossup = torch.dot(first, second)
        lossdiv =(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        loss = lossup/lossdiv
        
        
        dalpha_pos = torch.autograd.grad(loss, Likelihood) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                if d != None and d[0] != None:
                    p -= 2. * eps * d[0]
        # forward
        logits, att = self.net(v, b, q, a)       
        # sigmoid loss
        first = torch.sigmoid(Likelihood[step*batch_size:dataIndex])
        second = instance_bce_with_logits(logits, a, reduction='none').mean(1).cuda() 
        lossup = torch.dot(first, second)
        lossdiv =(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        loss = lossup/lossdiv
        dalpha_neg = torch.autograd.grad(loss, Likelihood) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                if d != None and d[0] != None:
                    p += eps * d[0]

        hessian = [(p-n) / (2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian



def train(train_loader: DataLoader, eval_loader, train_dset, model, optimizer, epoch, args):
    model.train()
    total_loss = 0.0
    train_score = 0
    total_norm = 0
    count_norm = 0
    grad_clip = .25
    
    architect = Architect(model, 0.9, 3e-4)
    Likelihood = torch.nn.Parameter(torch.ones(len(train_dset)).cuda(),requires_grad=True).cuda()
    Likelihood_optim = torch.optim.Adam({Likelihood}, 0.1, betas=(0.5, 0.999))
    
    for step, ((v, b, q, a), (vv, vb, vq, va)) in enumerate(zip(train_loader, eval_loader)):
        v = v.cuda(args.gpu)
        b = b.cuda(args.gpu)
        q = q.cuda(args.gpu)
        a = a.cuda(args.gpu)
        vv = vv.cuda(args.gpu)
        vb = vb.cuda(args.gpu)
        vq = vq.cuda(args.gpu)
        va = va.cuda(args.gpu)                                         
    
    
        Likelihood_optim.zero_grad()
        Likelihood, Likelihood_optim, valid_loss= architect.unrolled_backward((v, b, q), a, (vv, vb, vq), va, optimizer, Likelihood, Likelihood_optim, step, args)
        
        
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

    if args.multiGPUs:
        mp.spawn(main_worker,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True)
    else:
        main_worker(args.gpu, args)


if __name__ == '__main__':
    main()
