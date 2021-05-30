# coding=utf-8

import base64
import json
import os
import pandas as pd
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

import re
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def load_tsv(split: str):
    tsv_file = 'data/pvqa/images/%s%s.csv' % (split, args.pvqaimgv)
    df = pd.read_csv(tsv_file, delimiter='\t', names=FIELDNAMES)

    data = []
    for i in range(df.shape[0]):
        datum = {}
        datum['img_id'] = '%s_%04d' % (split, df['image_id'][i])
        datum['img_w'] = df['image_w'][i]
        datum['img_h'] = df['image_h'][i]
        datum['num_boxes'] = df['num_boxes'][i]

        boxes = df['boxes'][i]
        buf = base64.b64decode(boxes[1:])
        temp = np.frombuffer(buf, dtype=np.float64).astype(np.float32)
        datum['boxes'] = temp.reshape(datum['num_boxes'], -1)

        features = df['features'][i]
        buf = base64.b64decode(features[1:])
        temp = np.frombuffer(buf, dtype=np.float32)
        datum['features'] = temp.reshape(datum['num_boxes'], -1)

        data.append(datum)

    return data


class PVQADataset:

    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # loading dataset
        self.data = []
        for split in self.splits:
            self.data.extend(pickle.load(open('data/pvqa/qas/%s_vqa.pkl' % split, 'rb')))
        print('Load %d data from splits %s' % (len(self.data), self.name))
        # Convert list to dict for evaluation
        self.id2datum = {datum['question_id']: datum for datum in self.data}

        # Answers
        # self.q2a = pickle.load(open('data/pvqa/qas/q2a.pkl', 'rb'))
        # self.qid2a = pickle.load(open('data/pvqa/qas/qid2a.pkl', 'rb'))
        # self.qid2q = pickle.load(open('data/pvqa/qas/qid2q.pkl', 'rb'))
        self.ans2label = pickle.load(open('data/pvqa/qas/trainval_ans2label.pkl', 'rb'))
        self.label2ans = pickle.load(open('data/pvqa/qas/trainval_label2ans.pkl', 'rb'))

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class PVQATorchDataset(Dataset):
    def __init__(self, dataset: PVQADataset):
        super(PVQATorchDataset, self).__init__()
        self.raw_dataset = dataset

        # loading detection features to img_data
        self.imgid2img = {}
        for split in dataset.splits:
            data = load_tsv(split)

            for datum in data:
                self.imgid2img[datum['img_id']] = datum

        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print('use %d data in torch dataset' % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()

        assert obj_num == len(boxes) == len(feats)

        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


question_types = ('where', 'what', 'how', 'how many/how much', 'when', 'why', 'who/whose', 'other', 'yes/no')


def get_q_type(q: str):
    q = q.lower()
    if q.startswith('how many') or q.startswith('how much'):
        return 'how many/how much'
    first_w = q.split()[0]
    if first_w in ('who', 'whose'):
        return 'who/whose'
    for q_type in ('where', 'what', 'how', 'when', 'why'):
        if first_w == q_type:
            return q_type
    if first_w in ['am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'does', 'do', 'did', 'can', 'could']:
        return 'yes/no'
    if 'whose' in q:
        return 'who/whose'
    if 'how many' in q or 'how much' in q:
        return 'how many/how much'
    for q_type in ('where', 'what', 'how', 'when', 'why'):
        if q_type in q:
            return q_type
    print(q)
    return 'other'


class PVQAEvaluator:
    def __init__(self, dataset: PVQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        qtype_score = {qtype: 0. for qtype in question_types}
        qtype_cnt = {qtype: 0 for qtype in question_types}
        preds = []
        anss = []
        b_scores = []
        b_scores1 = []
        b_scores2 = []
        b_scores3 = []
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            quest = datum['sent']

            q_type = get_q_type(quest)
            qtype_cnt[q_type] += 1

            hypo = str(ans).lower().split()
            refs = []
            preds.append(self.dataset.ans2label[ans])
            if ans in label:
                score += label[ans]
                qtype_score[q_type] += label[ans]
            ans_flag = True
            for gt_ans in label:
                refs.append(str(gt_ans).lower().split())
                if ans_flag:
                    anss.append(self.dataset.ans2label[gt_ans] if gt_ans in self.dataset.ans2label else -1)
                    ans_flag = False
            b_score = sentence_bleu(references=refs, hypothesis=hypo)
            b_score1 = sentence_bleu(references=refs, hypothesis=hypo,
                                     weights=[1, 0, 0, 0])
            b_score2 = sentence_bleu(references=refs, hypothesis=hypo,
                                     weights=[0, 1, 0, 0])
            b_score3 = sentence_bleu(references=refs, hypothesis=hypo,
                                     weights=[0, 0, 1, 0])

            b_scores.append(b_score)
            b_scores1.append(b_score1)
            b_scores2.append(b_score2)
            b_scores3.append(b_score3)
        b_score_m = np.mean(b_scores)
        b_score_m1 = np.mean(b_scores1)
        b_score_m2 = np.mean(b_scores2)
        b_score_m3 = np.mean(b_scores3)
        info = 'b_score=%.4f\n' % b_score_m
        info += 'b_score1 = %.4f\n' % b_score_m1
        info += 'b_score2 = %.4f\n' % b_score_m2
        info += 'b_score3 = %.4f\n' % b_score_m3

        info += 'f1_score=%.4f\n' % f1_score(anss, preds, average='macro')
        info += 'score = %.4f\n' % (score / len(quesid2ans))
        for q_type in question_types:
            if qtype_cnt[q_type] > 0:
                qtype_score[q_type] /= qtype_cnt[q_type]
        info += 'Overall score: %.4f\n' % (score / len(quesid2ans))
        for q_type in question_types:
            info += 'qtype: %s\t score=%.4f\n' % (q_type, qtype_score[q_type])

        with open(os.path.join(args.output, 'result_by_type.txt'), 'a') as f:
            f.write(info)
        print(info)
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)
