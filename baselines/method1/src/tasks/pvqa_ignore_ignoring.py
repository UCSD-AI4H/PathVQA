import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args

from pretrain.qa_answer_table import load_lxmert_qa
from tasks.pvqa_model import PVQAModel
from tasks.pvqa_data import PVQADataset, PVQATorchDataset, PVQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
valid_bs = 16


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = PVQADataset(splits)
    tset = PVQATorchDataset(dset)
    evaluator = PVQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs, shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )
#   return ques_id, feats, boxes, ques, target
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)




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
        
        feats, boxes, sent = trn_X
        target = trn_y
        
           
        # forward
        logit = self.v_net(feats, boxes, sent)
        assert logit.dim() == target.dim() == 2
        
        
        
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # sigmoid loss
        first = torch.sigmoid(Likelihood[step*args.batch_size:dataIndex])
        second = bce_loss(logit, target).mean(1).cuda() 
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
        
        
        vfeats, vboxes, vsent = val_X
        vtarget = val_y
        # calc val prediction
        logit = self.v_net(vfeats, vboxes, vsent)
        # calc unrolled validation loss
        
        assert logit.dim() == vtarget.dim() == 2
        bce_loss = nn.BCEWithLogitsLoss()
        loss = bce_loss(logit, vtarget)
         # L_val(w`)
        
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
        
        feats, boxes, sent = trn_X
        target = trn_y
        
        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                if d!= None and d[0] != None:
                    pp = eps * d[0]
                    p += eps * d[0]
        
        
        # forward & calc loss
        dataIndex = len(trn_y)+step*batch_size 
        # forward
        logits = self.net(feats, boxes, sent)
        # sigmoid loss
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        first = torch.sigmoid(Likelihood[step*batch_size:dataIndex])
        second = bce_loss(logits, target).mean(1).cuda() 
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
        logits = self.net(feats, boxes, sent)       
        # sigmoid loss
        first = torch.sigmoid(Likelihood[step*batch_size:dataIndex])
        second = bce_loss(logits, target).mean(1).cuda()
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
    
    
    
class PVQA:
    def __init__(self):
        # datasets

        self.train_tuple = get_data_tuple(
            splits=args.train, bs=args.batch_size, shuffle=False, drop_last=False
        )
#         DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                splits=args.valid, bs=valid_bs,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # Model
        self.model = PVQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple, args):
        dset, loader, evaluator = train_tuple
        vdset, vloader, vevaluator = eval_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        
        
        
        architect = Architect(self.model, 0.9, 3e-4)
        Likelihood = torch.nn.Parameter(torch.ones(len(dset)).cuda(),requires_grad=True).cuda()
        Likelihood_optim = torch.optim.Adam({Likelihood}, 0.1, betas=(0.5, 0.999))
    
    
        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, ((ques_id, feats, boxes, sent, target),(vques_id, vfeats, vboxes, vsent, vtarget)) in iter_wrapper(enumerate(zip(loader, vloader))):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                vfeats, vboxes, vtarget = vfeats.cuda(), vboxes.cuda(), vtarget.cuda()
                
                
                Likelihood_optim.zero_grad()
                Likelihood, Likelihood_optim, valid_loss= architect.unrolled_backward((feats, boxes, sent), target, (vfeats, vboxes, vsent), vtarget, self.optim, Likelihood, Likelihood_optim, i, args)
        
        
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]  # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == '__main__':
    pvqa = PVQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        pvqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False  # Always loading all data in test
        if 'test' in args.test:
            result = pvqa.evaluate(
                get_data_tuple(args.test, bs=valid_bs,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
            print(result)
            with open(args.output + "/log.log", 'a') as f:
                f.write('test result=' + str(result))
                f.flush()

        elif 'val' in args.test:
            ## NOT USED
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = pvqa.evaluate(
                get_data_tuple('test', bs=valid_bs,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
            print(result)
            with open(args.output + "/log.log", 'a') as f:
                f.write('test result=' + str(result))
                f.flush()


        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', pvqa.train_tuple.dataset.splits)
        if pvqa.valid_tuple is not None:
            print('Splits in Valid data:', pvqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (pvqa.oracle_score(pvqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        pvqa.train(pvqa.train_tuple, pvqa.valid_tuple, args)
