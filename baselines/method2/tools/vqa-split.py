"""

Split VQA COCO2014_Val dataset

"""

import os
import pickle
import json
import h5py
import numpy as np

if __name__ == '__main__':

    imgid2idx = pickle.load(open('../data/val_imgid2idx.pkl', 'rb'))

    hf = h5py.File('../data/val.hdf5', 'r')

    qs = json.load(open('../data/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))

    target = pickle.load(open('../data/cache/val_target.pkl', 'rb'))

    quests = qs['questions']
    quests = sorted(quests, key=lambda x: x['question_id'])

    th = 20000
    # 0~19999: valval
    # 20000~40503: valtest

    idx_s = {'val': range(0, th), 'test': range(th, len(imgid2idx))}
    st_s = {'val': 0, 'test': th}
    for s in ['val', 'test']:
        # imgid2idx
        s_imgid2idx = {}
        for imgid in imgid2idx:
            idx = imgid2idx[imgid]
            if idx in idx_s[s]:
                s_imgid2idx[imgid] = idx - st_s[s]

        pickle.dump(s_imgid2idx, open('../data/val%s_imgid2idx.pkl' % s, 'wb'))

        # questions
        # ['info', 'task_type', 'data_type', 'license', 'data_subtype',
        s_qs = {}
        for k in ['info', 'task_type', 'data_type', 'license']:
            s_qs[k] = qs[k]
        s_qs['data_subtype'] = 'val%s2014' % s

        s_quests = []
        for qa in quests:
            img_id = qa['image_id']
            idx = imgid2idx[img_id]
            if idx in idx_s[s]:
                s_quests.append(qa)
        s_qs['questions'] = s_quests
        json.dump(s_qs, open('../data/v2_OpenEnded_mscoco_val%s2014_questions.json' % s, 'w'))

        # target
        s_target = []
        for t in target:
            img_id = t['image_id']
            idx = imgid2idx[img_id]
            if idx in idx_s[s]:
                s_target.append(t)
        pickle.dump(s_target, open('../data/cache/val%s_target.pkl' % s, 'wb'))

    fval = h5py.File('../data/valval.hdf5', 'w')
    ftest = h5py.File('../data/valtest.hdf5', 'w')
    pos_boxes = np.array(hf['pos_boxes'])
    box_th = pos_boxes[th, 0]

    for att in ['image_bb', 'image_features', 'spatial_features']:
        fval.create_dataset(att, data=np.array(hf.get(att))[:box_th])
        ftest.create_dataset(att, data=np.array(hf.get(att))[box_th:])

    fval.create_dataset('pos_boxes', data=pos_boxes[:th])
    ftest.create_dataset('pos_boxes', data=pos_boxes[th:]-box_th)

    fval.close()
    ftest.close()
    hf.close()