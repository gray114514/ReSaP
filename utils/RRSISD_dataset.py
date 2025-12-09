#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import sys
import os.path as osp
import json
import pickle as pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np 
from pycocotools import mask
from .prompt import get_qwen3_box_prompt , get_qwen3_point_prompt , sample_points_adaptive_grid
from .prompt import get_GRPO_point_prompt , get_GRPO_box_prompt , sample_points_and_labels_json 

class REFER:

    def __init__(self, data_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        if dataset == 'refcocog':
            print('Split by {}!'.format(splitBy))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        elif dataset == 'rrsisd':
            self.IMAGE_DIR = osp.join(data_root, 'images/rrsisd/JPEGImages')
        elif dataset == 'ris_lad':
            self.IMAGE_DIR = osp.join(data_root, 'images/ris_lad/JPEGImages')
        else: 
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        f = open(ref_file, 'r')
        self.data['refs'] = pickle.load(open(ref_file, 'rb'))

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if
                         image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == unicode:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann['segmentation']
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref):
        # return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        image = self.Imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']

        m = mask.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']

        return {'mask': m, 'area': area}


    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)


# In[7]:


import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random



def add_random_boxes(img, min_num=20, max_num=60, size=32):
    h,w = size, size
    img = np.asarray(img).copy()
    img_size = img.shape[1]
    boxes = []
    num = random.randint(min_num, max_num)
    for k in range(num):
        y, x = random.randint(0, img_size-w), random.randint(0, img_size-h)
        img[y:y+h, x: x+w] = 0
        boxes. append((x,y,h,w) )
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


class ReferDataset(data.Dataset):

    def __init__(self,
                 refer_data_root="./dataset/RRSISD",
                 dataset='rrsisd',
                 splitBy='unc',
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False,):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(refer_data_root, dataset, splitBy)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        num_images_to_mask = int(len(ref_ids) * 0.2)
        self.images_to_mask = random.sample(ref_ids, num_images_to_mask)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.sentences = []

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                sentences_for_ref.append(sentence_raw)

            self.sentences.append(sentences_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name']))

        ref = self.refer.loadRefs(this_ref_id)

        if len(ref) != 1:
            print("error !!!!")
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
#        ref_box = self.refer.getRefBox(ref[0]['ref_id'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization\
            #H , W = img.size
            img, target = self.image_transforms(img, annot)
            rows, cols = np.nonzero(target)
            if len(rows) > 0 and len(cols) > 0:
                y_min = np.min(rows)
                y_max = np.max(rows)
                x_min = np.min(cols)
                x_max = np.max(cols)
                #H1 , W1 = img.size
                #ref_box = [round(i * H1 / H)  for i in ref_box]
                ref_box = [int(x_min),int(y_min),int(x_max),int(y_max)]
            else:
                ref_box = self.refer.getRefBox(ref[0]['ref_id'])
                H , W = img.size
                H1 , W1 = img.size
                ref_box = [round(i * H1 / H)  for i in ref_box]
        else: target = annot
        if self.eval_mode:
            sentences = []
            for s in range(len(self.sentences[index])):
                e = self.sentences[index][s]
                sentences.append(e)

        else:
            choice_sent = np.random.choice(len(self.sentences[index]))
            sentences = self.sentences[index][choice_sent]

        return img, target, sentences , ref_box
    

class RRSISDDataCollator:
    def __init__(self, processor,use_think=False):
        self.processor = processor
        self.use_think = use_think
    def __call__(self, features):
        
        texts = []
        images = []
        masks = []
        sentences = []
        boxes = []
        for image ,mask, expression , ref_box in features:
            if self.use_think:
                text , _ = get_GRPO_box_prompt(self.processor,expression)
            else:
                text , _ = get_qwen3_box_prompt(self.processor,expression)
            texts.append(text)
            sentences.append(expression)
            images.append(image)
            mask = np.array(mask)
            mask = torch.from_numpy(mask).long()                     
            masks.append(mask)
            boxes.append(ref_box)
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        return images , masks , batch , sentences , boxes
    

class GRPO_DataCollator:
    def __init__(self, processor, point_task_ratio=0.2,is_random=False):
        """
        processor: Qwen 图像+文本预处理器
        point_task_ratio: float, 随机采样中用于点判断任务的比例 (0~1)
        """
        self.processor = processor
        self.point_task_ratio = point_task_ratio
        self.is_random = is_random
    def __call__(self, features):
        images, masks, sentences, ref_boxes = zip(*features)

        # 将 mask 转为 tensor
        masks = [torch.from_numpy(np.array(m)).long() for m in masks]
        masks = torch.stack(masks,dim=0)
        is_point_task = random.random() < self.point_task_ratio
        if is_point_task:
        
            if self.is_random:
                points_json_list, answer_json_list = sample_points_and_labels_json(masks)
                answers = answer_json_list
            else:
                _ , H , W = masks.shape
                points_list, points_json, answer_list = sample_points_adaptive_grid(ref_boxes,W=W,H=H,return_answer=True,mask=masks)
                points_json_list = [points_json]
                answers = [answer_list]
            texts = [get_GRPO_point_prompt(self.processor, sents, points_json)[0]
                     for sents, points_json in zip(sentences, points_json_list)]
           

        else:
            texts = []
            answers = []
            for sents, box in zip(sentences, ref_boxes):
                text, _ = get_GRPO_box_prompt(self.processor, sents)
                texts.append(text)
                answers.append(box)

        batch = self.processor(
            text=texts,
            images=list(images),
            return_tensors="pt",
            padding=True,
        )

        return images, masks, batch, answers , is_point_task
    
    

class SFT_DataCollator:
    def __init__(self, processor,point_task_ratio=0.5):
        self.processor = processor
        self.point_task_ratio = point_task_ratio
    def __call__(self, features):
        images, masks, sentences, ref_boxes = zip(*features)
        # 将 mask 转为 tensor
        masks = [torch.from_numpy(np.array(m)).long() for m in masks]
        masks = torch.stack(masks,dim=0)

        texts , prompt = [] , []
        is_point_task = random.random() < self.point_task_ratio
            
        for sents, box in zip(sentences, ref_boxes):
            if is_point_task:
                _ , H , W = masks.shape
                points_list , points_json_list , answer = sample_points_adaptive_grid(boxes=[box],mask=masks,W=W,H=H,return_answer=True)
                text , _ = get_qwen3_point_prompt(self.processor,sents,points_json_list,answer)
                text1 , _ = get_qwen3_point_prompt(self.processor,sents,points_json_list)
            else:
                text, _ = get_qwen3_box_prompt(self.processor, sents,box)
                text1 , _  = get_qwen3_box_prompt(self.processor,sents)
            texts.append(text.rstrip())
            prompt.append(text1)

        batch = self.processor(
            text=texts,
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        prompt_batch = self.processor(
            text=prompt,
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        labels = []
        for i in range(prompt_batch.input_ids.shape[0]):
            L1 = prompt_batch.input_ids[i].shape[0]
            L2 = batch.input_ids[i].shape[0]
            labels.append(torch.cat((torch.full(size=(L1,),fill_value=-100) , batch.input_ids[i,L1:]) , dim = 0))
        labels = torch.stack(labels,dim=0)
        return images, masks, batch , labels