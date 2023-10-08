import math
import os.path
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from beit3_tools.beit3_datasets import get_sentencepiece_model_for_beit3
import numpy as np
from beit3_tools import utils
from tqdm import tqdm
import os
from beit3_tools.engine_for_finetuning import TaskHandler

class AtomicHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.image_feats = []
        self.text_feats = []
        self.image_ids = []
        self.text_ids = []
        self.store_pointer = 0
        self.metric_logger = None
        self.store_freq = 200


    def train_batch(self, model, image, language_tokens, padding_mask, image_id,  **kwargs):
        loss, vision_cls, language_cls = model(
            image=image, text_description=language_tokens, padding_mask=padding_mask)
        return {
            "loss": loss,
        }

    def before_eval(self, metric_logger, **kwargs):
        self.image_feats.clear()
        self.text_feats.clear()
        self.image_ids.clear()
        self.text_ids.clear()
        self.metric_logger = metric_logger
        self.store_pointer = 0

    def eval_batch(self, model, image, language_tokens, padding_mask, image_id):
        vision_cls, _ = model(image=image, only_infer=True)
        _, language_cls = model(
            text_description=language_tokens, padding_mask=padding_mask, only_infer=True)

        self.image_feats.append(vision_cls.clone())
        self.text_feats.append(language_cls.clone())
        self.image_ids.append(image_id.clone())

    def build_rank(self, query_dataloader, answer_dataloader, topk, values, query_data_type, k=1000):
        all_rank = {}
        if query_data_type == 'img':
            retrieval_type = 'text'
        elif query_data_type == 'text':
            retrieval_type = 'img'

        print(f"Build rank list for {query_data_type} to {retrieval_type}")
        topk = topk.detach().cpu() #.reshape(-1, k)
        values = values.detach().cpu() #.reshape(-1, k)

        for idx in tqdm(range(topk.shape[0])):
            if query_data_type == 'img':
                item_id = query_dataloader.dataset._get_img_id(idx)
            elif query_data_type == 'text':
                item_id = query_dataloader.dataset._get_text_id(idx)

            rank_list = topk[idx].tolist()
            # transfer rank idx to item id
            if retrieval_type == 'img':
                rank_list = answer_dataloader.dataset._get_img_id(rank_list)
            elif retrieval_type == 'text':
                rank_list = answer_dataloader.dataset._get_text_id(rank_list)

            all_rank[item_id] = {'rank': rank_list,
                                'scores': values[idx].tolist()}

        return all_rank

    def after_eval(self, data_loader, build_ranking=False, **kwargs):
        image_feats = {}
        for feats, ids in zip(self.image_feats, self.image_ids):
            for i, _idx in enumerate(ids):
                idx = _idx.item()
                if idx not in image_feats:
                    image_feats[idx] = feats[i]

        tiids = torch.cat(self.image_ids, dim=0)
        iids = []
        sorted_tensors = []
        for key in sorted(image_feats.keys()):
            sorted_tensors.append(image_feats[key].view(1, -1))
            iids.append(key)

        image_cls_feats = torch.cat(sorted_tensors, dim=0)
        text_cls_feats = torch.cat(self.text_feats, dim=0)

        scores = image_cls_feats @ text_cls_feats.t()
        iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("iids: {}".format(iids.size()))
        print("tiids: {}".format(tiids.size()))

        topk1000 = scores.topk(1000, dim=1)
        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk1 = scores.topk(1, dim=1)

        topk10_iids = tiids[topk10.indices]
        topk5_iids = tiids[topk5.indices]
        topk1_iids = tiids[topk1.indices]
        topk1000_iids = tiids[topk1000.indices]


        tr_r1000 = (iids.unsqueeze(1) == topk1000_iids).float().max(dim=1)[0].mean()
        tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

        np.save('topk5_iids.npy', topk5_iids.cpu().numpy())
        np.save('tiids.npy', iids.cpu().numpy())

        image_to_text_rank = self.build_rank(data_loader, topk1000_iids, topk1000.values,
                                             query_data_type='img')


        scores = scores.t()
        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk1 = scores.topk(1, dim=1)
        topk1000 = scores.topk(1000, dim=1)

        topk1000_iids = iids[topk1000.indices]
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        topk1_iids = iids[topk1.indices]




        text_to_image_rank = self.build_rank(data_loader, topk1000_iids, topk1000.values,
                                             query_data_type='text')

        ir_r1000 = (tiids.unsqueeze(1) == topk1000_iids).float().max(dim=1)[0].mean()
        ir_r10 = (tiids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        ir_r5 = (tiids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        ir_r1 = (tiids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()



        eval_result = {
            "tr_r1": tr_r1.item() * 100.0,
            "tr_r5": tr_r5.item() * 100.0,
            "tr_r10": tr_r10.item() * 100.0,
            "tr_r1000": tr_r1000.item() * 100.0,

            "ir_r1": ir_r1.item() * 100.0,
            "ir_r5": ir_r5.item() * 100.0,
            "ir_r10": ir_r10.item() * 100.0,
            "ir_r1000": ir_r1000.item() * 100.0,

            "average_score": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0,
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        if build_ranking:
            return eval_result, "average_score", text_to_image_rank, image_to_text_rank

        else:
            return eval_result, "average_score"

class AtomicAllretHandler(AtomicHandler):
    def __init__(self) -> None:
        super().__init__()

        self.use_gpu = False

    def store_feats(self, mode, tag, gpu_id):
        if not os.path.exists(f"embeddings/{mode}/{tag}"):
            os.makedirs(f"embeddings/{mode}/{tag}")

        if mode == 'image':
            np.save( 'embeddings/image/{}/image_feats_{}_freq_{}_gpu_{}.npy'.format(tag, self.store_pointer, self.store_freq, gpu_id), torch.cat(self.image_feats[self.store_pointer:self.store_pointer+self.store_freq], dim=0), allow_pickle=True)
            print('save embeddings to embeddings/image/{}/image_feats_{}_freq_{}_gpu_{}.npy'.format(tag, self.store_pointer, self.store_freq, gpu_id))
        elif mode == 'text':
            np.save( 'embeddings/text/{}/text_feats_{}_freq_{}_gpu_{}.npy'.format(tag, self.store_pointer, self.store_freq, gpu_id), torch.cat(self.text_feats[self.store_pointer:self.store_pointer+self.store_freq], dim=0), allow_pickle=True)
            print('save embeddings to embeddings/text/{}/text_feats_{}_freq_{}_gpu_{}.npy'.format(tag, self.store_pointer, self.store_freq, gpu_id))


    def eval_batch(self, model, mode='image', image=None, language_tokens=None, padding_mask=None, image_id=None, text_id=None):
        if mode == 'image':
            vision_cls, _ = model(image=image, only_infer=True)
            self.image_feats.append(vision_cls.detach().cpu())
            self.image_ids.append(image_id.detach().cpu())
        elif mode == 'text':
            _, language_cls = model(
            text_description=language_tokens, padding_mask=padding_mask, only_infer=True)
            self.text_feats.append(language_cls.detach().cpu())
            self.text_ids.append(text_id.detach().cpu())
        else:
            raise ValueError("mode should be either image or text")

    def after_eval(self, query_dataloader, answer_dataloader, retrieval_mode, args, build_ranking=False, **kwargs):

        if args.load_image_from_precomputed_npy:
            # image_cls_feats = self.image_feats
            tiids = torch.cat(self.text_ids, dim=0)  #
            for i in range(len(self.image_feats)):
                self.image_feats[i] = self.image_feats[i].view(1, -1)

            image_cls_feats = torch.cat(self.image_feats, dim=0)
            text_cls_feats = torch.cat(self.text_feats, dim=0)
            iids = self.image_ids
            # iids = torch.from_numpy(np.array())self.image_ids
            # image_cls_feats = torch.tensor(image_cls_feats)

        else:
            image_feats = {}
            for feats, ids in zip(self.image_feats, self.image_ids):
                for i, _idx in enumerate(ids):
                    idx = _idx.item()
                    if idx not in image_feats:
                        image_feats[idx] = feats[i]

            tiids = torch.cat(self.text_ids, dim=0)  #
            iids = []
            sorted_tensors = []
            for key in sorted(image_feats.keys()):
                sorted_tensors.append(image_feats[key].view(1, -1))
                iids.append(key)

            image_cls_feats = torch.cat(sorted_tensors, dim=0)
            text_cls_feats = torch.cat(self.text_feats, dim=0)

        if args.eval:
            # clear GPU memory
            del self.image_feats
            del self.text_feats
            torch.cuda.empty_cache()
        else:
            self.image_feats = []
            self.text_feats = []
            torch.cuda.empty_cache()


        iids = torch.LongTensor(iids).to(image_cls_feats.device)
        tiids = torch.LongTensor(tiids).to(image_cls_feats.device)

        print("iids: {}".format(iids.size()))
        print("tiids: {}".format(tiids.size()))
        print("image_cls_feats: {}".format(image_cls_feats.size()))
        print("text_cls_feats: {}".format(text_cls_feats.size()))

        print("calculate scores")
        scores_values = []

        ir_r1000 = 0
        ir_r10 = 0
        ir_r5 = 0
        ir_r1 = 0
        mrr_10 = 0


        top_10_hit_index = []
        if self.use_gpu:
            iids = iids.to('cuda')
            tiids = tiids.to('cuda')


        freq = 1000
        for i in tqdm(range(0, text_cls_feats.shape[0], freq)):
            scores = []
            if self.use_gpu:
                last = 0
                freq = 1280000
                for bound in range(freq, image_cls_feats.shape[0], freq):
                    image_cls_feats_i = image_cls_feats[bound - freq : bound].to('cuda')
                    text_cls_feats_i = text_cls_feats[i].to('cuda')
                    score = image_cls_feats_i @ text_cls_feats_i.t()
                    scores.append(score)
                    last = bound


                image_cls_feats_i = image_cls_feats[last:].to('cuda')
                text_cls_feats_i = text_cls_feats[i].to('cuda')
                score = image_cls_feats_i @ text_cls_feats_i.t()
                scores.append(score)
                score = torch.cat(scores, dim=0)

            else:
                print(f"calculate scores from {i} to {i+freq}")
                scores = image_cls_feats @ text_cls_feats[i:i+freq].t()


                score = scores.t()
                # scores_values.append(score)
                topk10 = score.topk(10, dim=1)
                topk5 = score.topk(5, dim=1)
                topk1 = score.topk(1, dim=1)
                topk1000 = score.topk(1000, dim=1)

                topk1000_iids = iids[topk1000.indices]
                topk10_iids = iids[topk10.indices]
                topk5_iids = iids[topk5.indices]
                topk1_iids = iids[topk1.indices]

                #

                ir_r1000_ = (tiids[i:i+freq].unsqueeze(1) == topk1000_iids).float().max(dim=1)[0].mean()
                ir_r10_ = (tiids[i:i+freq].unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
                ir_r5_ = (tiids[i:i+freq].unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
                ir_r1_ = (tiids[i:i+freq].unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()


                mrr_10_ = 0
                for j in range(topk10_iids.shape[0]):
                    if tiids[i+j] in topk10_iids[j]:
                        mrr_10_ += 1 / (topk10_iids[j].tolist().index(tiids[i+j]) + 1)
                mrr_10_ /= topk10_iids.shape[0]
                print(f"mrr_10: {mrr_10_}")
                ir_r1000 += ir_r1000_
                ir_r10 += ir_r10_
                ir_r5 += ir_r5_
                ir_r1 += ir_r1_

                mrr_10 += mrr_10_




        total = text_cls_feats.shape[0]//freq + 1
        print(f"total: {total}")
        ir_r1000 /= total
        ir_r10 /= total
        ir_r5 /= total
        ir_r1 /= total
        mrr_10 /= total


        if build_ranking:
            text_to_image_rank = self.build_rank(query_dataloader, answer_dataloader, topk1000_iids, scores_values,
                                                 query_data_type='text')


        eval_result = {
            "ir_r1": ir_r1.item() * 100.0,
            "ir_r5": ir_r5.item() * 100.0,
            "ir_r10": ir_r10.item() * 100.0,
            "ir_r1000": ir_r1000.item() * 100.0,
            "mrr_10": mrr_10,
            "average_score": 100.0 * (ir_r10 + ir_r1000).item() / 2.0,
        }
        print('* Eval result = ', eval_result)
        print('* Eval result = %s' % json.dumps(eval_result))
        if build_ranking:
            return eval_result, "average_score", text_to_image_rank,

        else:
            return eval_result, "average_score"

class AtomicSubmissionHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.image_feats = []
        self.text_feats = []
        self.image_ids = []
        self.text_ids = []
        self.store_feq = 100
        # self.store_pointer = 51
        self.metric_logger = None

    def store_feats(self, mode, tag, gpu_id):
        if not os.path.exists(f"embeddings/{mode}/{tag}"):
            os.makedirs(f"embeddings/{mode}/{tag}")

        if mode == 'image':
            np.save( 'embeddings/image/{}/image_feats_{}_freq_{}_gpu_{}.npy'.format(tag, self.store_pointer, self.store_feq, gpu_id), torch.cat(self.image_feats[self.store_pointer:self.store_pointer+self.store_feq], dim=0), allow_pickle=True)
            print('save embeddings to embeddings/image/{}/image_feats_{}_freq_{}_gpu_{}.npy'.format(tag, self.store_pointer, self.store_feq, gpu_id))
        elif mode == 'text':
            np.save( 'embeddings/text/{}/text_feats_{}_freq_{}_gpu_{}.npy'.format(tag, self.store_pointer, self.store_feq, gpu_id), torch.cat(self.text_feats[self.store_pointer:self.store_pointer+self.store_feq], dim=0), allow_pickle=True)
            print('save embeddings to embeddings/text/{}/text_feats_{}_freq_{}_gpu_{}.npy'.format(tag, self.store_pointer, self.store_feq, gpu_id))

    def before_eval(self, metric_logger, **kwargs):
        self.image_feats.clear()
        self.text_feats.clear()
        self.image_ids.clear()
        self.text_ids.clear()
        self.store_pointer = 0
        self.metric_logger = metric_logger

    def eval_batch(self, model, mode='image', image=None, language_tokens=None, padding_mask=None, image_id=None, text_id=None):
        if mode == 'image':
            vision_cls, _ = model(image=image, only_infer=True)
            self.image_feats.append(vision_cls.detach().cpu())
            self.image_ids.append(image_id.detach().cpu())
        elif mode == 'text':
            _, language_cls = model(
            text_description=language_tokens, padding_mask=padding_mask, only_infer=True)
            self.text_feats.append(language_cls.detach().cpu())
            self.text_ids.append(text_id.detach().cpu())
        else:
            raise ValueError("mode should be either image or text")

    def build_rank(self, query_dataloader, answer_dataloader, topk, values, query_data_type, k=1000):
        all_rank = {}
        if query_data_type == 'img':
            retrieval_type = 'text'
        elif query_data_type == 'text':
            retrieval_type = 'img'

        print(f"Build rank list for {query_data_type} to {retrieval_type}")
        topk = topk.detach().cpu() #.reshape(-1, k)
        values = values.detach().cpu() #.reshape(-1, k)

        for idx in tqdm(range(topk.shape[0])):
            if query_data_type == 'img':
                item_id = query_dataloader.dataset._get_img_id(idx)
            elif query_data_type == 'text':
                item_id = query_dataloader.dataset._get_text_id(idx)

            rank_list = topk[idx].tolist()
            # transfer rank idx to item id
            if retrieval_type == 'img':
                rank_list = answer_dataloader.dataset._get_img_id(rank_list)
            elif retrieval_type == 'text':
                rank_list = answer_dataloader.dataset._get_text_id(rank_list)

            all_rank[item_id] = {'rank': rank_list,
                                'scores': values[idx].tolist()}

        return all_rank

    def after_eval(self, query_dataloader, answer_dataloader, mode, args, **kwargs):

        if args.load_embeddings_from_npy:
            if mode == 'text_to_image':
                tiids = torch.cat(self.text_ids, dim=0)
                text_cls_feats = torch.cat(self.text_feats, dim=0)

                image_feats = {}
                for feats, ids in zip(self.image_feats, self.image_ids):
                    for i, _idx in enumerate(ids):
                        idx = _idx.item()
                        if idx not in image_feats:
                            image_feats[idx] = feats[i]
                iids = []
                sorted_tensors = []
                for key in sorted(image_feats.keys()):
                    sorted_tensors.append(image_feats[key].view(1, -1))
                    iids.append(key)
                image_cls_feats = torch.cat(sorted_tensors, dim=0)

                # sorted_tensors = self.image_feats
                # iids = self.image_ids

                # image_cls_feats = self.image_feats

            elif mode == 'image_to_text':
                image_feats = {}
                for feats, ids in zip(self.image_feats, self.image_ids):
                    for i, _idx in enumerate(ids):
                        idx = _idx.item()
                        if idx not in image_feats:
                            image_feats[idx] = feats[i]
                iids = []
                sorted_tensors = []
                for key in sorted(image_feats.keys()):
                    sorted_tensors.append(image_feats[key].view(1, -1))
                    iids.append(key)

                tiids = self.text_ids
                image_cls_feats = torch.cat(sorted_tensors, dim=0)
                text_cls_feats = self.text_feats
            else:
                raise ValueError("mode should be either text_to_image or image_to_text")

        else:
            image_feats = {}
            for feats, ids in zip(self.image_feats, self.image_ids):
                for i, _idx in enumerate(ids):
                    idx = _idx.item()
                    if idx not in image_feats:
                        image_feats[idx] = feats[i]



            tiids = torch.cat(self.text_ids, dim=0) #
            iids = []

            sorted_tensors = []
            for key in sorted(image_feats.keys()):
                sorted_tensors.append(image_feats[key].view(1, -1))
                iids.append(key)


            image_cls_feats = torch.cat(sorted_tensors, dim=0)
            text_cls_feats = torch.cat(self.text_feats, dim=0) # torch.cat(text_sorted_tensors, dim=0)

        scores = image_cls_feats @ text_cls_feats.t()
        iids = torch.LongTensor(iids).to(scores.device)
        tiids = torch.LongTensor(tiids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("iids: {}".format(iids.size()))
        print("tiids: {}".format(tiids.size()))

        if mode == 'text_to_image':
            scores = scores.t()
            topk1000 = scores.topk(1000, dim=1)
            scores_values = topk1000.values
            topk1000_iids = iids[topk1000.indices]

            text_to_image_rank = self.build_rank(query_dataloader, answer_dataloader, topk1000_iids, scores_values,
                                                 query_data_type='text')

            return text_to_image_rank

        elif mode == 'image_to_text':

            topk1000 = scores.topk(1000, dim=1)
            topk1000_iids = tiids[topk1000.indices]
            image_to_text_rank = self.build_rank(query_dataloader, answer_dataloader, topk1000_iids, topk1000.values,
                                                    query_data_type='img')

            return image_to_text_rank
        else:
            raise ValueError("mode should be either text_to_image or image_to_text")


class Atomicstage1Handler(AtomicAllretHandler):
    def __init__(self) -> None:
        super().__init__()
    def after_eval(self, query_dataloader, answer_dataloader, retrieval_mode, args, build_ranking=False, **kwargs):
        if args.load_image_from_precomputed_npy:
            # image_cls_feats = self.image_feats
            tiids = torch.cat(self.text_ids, dim=0)  #
            for i in range(len(self.image_feats)):
                self.image_feats[i] = self.image_feats[i].view(1, -1)

            image_cls_feats = torch.cat(self.image_feats, dim=0)
            text_cls_feats = torch.cat(self.text_feats, dim=0)
            iids = self.image_ids
            # iids = torch.from_numpy(np.array())self.image_ids
            # image_cls_feats = torch.tensor(image_cls_feats)

        else:
            image_feats = {}
            for feats, ids in zip(self.image_feats, self.image_ids):
                for i, _idx in enumerate(ids):
                    idx = _idx.item()
                    if idx not in image_feats:
                        image_feats[idx] = feats[i]

            tiids = torch.cat(self.text_ids, dim=0)  #
            iids = []
            sorted_tensors = []
            for key in sorted(image_feats.keys()):
                sorted_tensors.append(image_feats[key].view(1, -1))
                iids.append(key)

            image_cls_feats = torch.cat(sorted_tensors, dim=0)
            text_cls_feats = torch.cat(self.text_feats, dim=0)

        if args.eval:
            # clear GPU memory
            del self.image_feats
            del self.text_feats
            torch.cuda.empty_cache()
        else:
            self.image_feats = []
            self.text_feats = []
            torch.cuda.empty_cache()

        iids = torch.LongTensor(iids).to(image_cls_feats.device)
        tiids = torch.LongTensor(tiids).to(image_cls_feats.device)

        print("iids: {}".format(iids.size()))
        print("tiids: {}".format(tiids.size()))
        print("image_cls_feats: {}".format(image_cls_feats.size()))
        print("text_cls_feats: {}".format(text_cls_feats.size()))

        print("calculate scores")
        if self.use_gpu:
            iids = iids.to('cuda')
            tiids = tiids.to('cuda')

        freq = 500
        for i in tqdm(range(206500 , text_cls_feats.shape[0], freq)):
            print(f"calculate scores from {i} to {i + freq}")
            scores = image_cls_feats @ text_cls_feats[i:i + freq].t()
            score = scores.t()
            # scores_values.append(score)
            topk100000 = score.topk(100000, dim=1)
            topk100000_iids = iids[topk100000.indices]
            topk100000_iids = topk100000_iids.tolist()
            with open(f'stage1/candidates_{i}_{i + freq}.json', 'w') as f:
                json.dump(topk100000_iids, f)
                print(f"save candidates to stage1/candidates_{i}_{i + freq}.json")

            del topk100000_iids, topk100000, score, scores

        eval_result = {

        }

        return eval_result, "average_score"


class Atomicstage2Handler(AtomicAllretHandler):
    def __init__(self) -> None:
        super().__init__()

    def after_eval(self, query_dataloader, answer_dataloader, retrieval_mode, args, build_ranking=False, **kwargs):
        if args.load_image_from_precomputed_npy:
            # image_cls_feats = self.image_feats
            tiids = torch.cat(self.text_ids, dim=0)  #
            for i in range(len(self.image_feats)):
                self.image_feats[i] = self.image_feats[i].view(1, -1)

            image_cls_feats = torch.cat(self.image_feats, dim=0)
            text_cls_feats = torch.cat(self.text_feats, dim=0)
            iids = self.image_ids

        else:
            image_feats = {}
            for feats, ids in zip(self.image_feats, self.image_ids):
                for i, _idx in enumerate(ids):
                    idx = _idx.item()
                    if idx not in image_feats:
                        image_feats[idx] = feats[i]
            tiids = torch.cat(self.text_ids, dim=0)  #
            iids = []
            sorted_tensors = []
            for key in sorted(image_feats.keys()):
                sorted_tensors.append(image_feats[key].view(1, -1))
                iids.append(key)

            image_cls_feats = torch.cat(sorted_tensors, dim=0)
            text_cls_feats = torch.cat(self.text_feats, dim=0)

        if args.eval:
            # clear GPU memory
            del self.image_feats
            del self.text_feats
            torch.cuda.empty_cache()
        else:
            self.image_feats = []
            self.text_feats = []
            torch.cuda.empty_cache()

        iids = torch.LongTensor(iids).to(image_cls_feats.device)
        tiids = torch.LongTensor(tiids).to(image_cls_feats.device)

        print("iids: {}".format(iids.size()))
        print("tiids: {}".format(tiids.size()))
        print("image_cls_feats: {}".format(image_cls_feats.size()))
        print("text_cls_feats: {}".format(text_cls_feats.size()))

        # load candidates
        for i in range(0, 10000, 1000):
            candidates = np.load(f'stage1/candidates_{i}_{i+1000}.npy')
            print(f"load candidates from stage1/candidates_{i}_{i+1000}.npy")
            if i == 0:
                candidates_all = candidates
            else:
                candidates_all = np.concatenate((candidates_all, candidates), axis=0)

        candidates_all = torch.from_numpy(candidates_all).to(image_cls_feats.device)
        print(f"candidates_all: {candidates_all.size()}")

        #build mask matrix on candidates
        mask_matrix = torch.zeros((candidates_all.size(0), image_cls_feats.size(0))).to(image_cls_feats.device)
        for i in range(candidates_all.size(0)):
            mask_matrix[i][candidates_all[i]] = 1
        print(f"mask_matrix: {mask_matrix.size()}")


        # calculate scores
        print("calculate scores")
        scores_values = []
        ir_r1000 = 0
        ir_r10 = 0
        ir_r5 = 0
        ir_r1 = 0
        mrr_10 = 0

        freq = 1000
        for i in tqdm(range(0, text_cls_feats.shape[0], freq)):

            score =  image_cls_feats @ text_cls_feats[i:i+freq].t()
            score  = score * mask_matrix[i:i+freq]
            score = score.t()

            topk10 = score.topk(10, dim=1)
            topk5 = score.topk(5, dim=1)
            topk1 = score.topk(1, dim=1)
            topk1000 = score.topk(1000, dim=1)

            topk1000_iids = iids[topk1000.indices]
            topk10_iids = iids[topk10.indices]
            topk5_iids = iids[topk5.indices]
            topk1_iids = iids[topk1.indices]

            ir_r1000_ = (tiids[i:i+freq].unsqueeze(1) == topk1000_iids).float().max(dim=1)[0].mean()
            ir_r10_ = (tiids[i:i+freq].unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
            ir_r5_ = (tiids[i:i+freq].unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
            ir_r1_ = (tiids[i:i+freq].unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

            ir_r1000 += ir_r1000_
            ir_r10 += ir_r10_
            ir_r5 += ir_r5_
            ir_r1 += ir_r1_
            mrr_10_ = 0
            for i in range(topk10_iids.shape[0]):
                if tiids[i] in topk10_iids[i]:
                    mrr_10_ += 1 / (topk10_iids[i].tolist().index(tiids[i]) + 1)
            mrr_10_ /= topk10_iids.shape[0]
            mrr_10 += mrr_10_

        total = text_cls_feats.shape[0]//freq + 1
        print(f"total: {total}")
        ir_r1000 /= total
        ir_r10 /= total
        ir_r5 /= total
        ir_r1 /= total
        mrr_10 /= total

        eval_result = {
            "ir_r1": ir_r1.item() * 100.0,
            "ir_r5": ir_r5.item() * 100.0,
            "ir_r10": ir_r10.item() * 100.0,
            "ir_r1000": ir_r1000.item() * 100.0,
            "mrr_10": mrr_10 * 100.0,
            "average_score": 100.0 * (ir_r10 + ir_r1000).item() / 2.0,
        }
        print('* Eval result = ', eval_result)
        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"



def get_handler(args):
    if args.task == "atomic":
        return AtomicHandler()
    elif args.task in ("atomic_submission"):
        return AtomicSubmissionHandler()
    elif args.task in ("atomic_allret_adapter"):
        return AtomicAllretHandler()
    elif args.task in ("atomic_stage1"):
        # return AtomicAllretHandler()
        return Atomicstage1Handler()
    else:
        raise NotImplementedError("Sorry, %s is not support." % args.task)


def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable,
        optimizer: torch.optim.Optimizer, device: torch.device,
        handler: TaskHandler, epoch: int, start_steps: int,
        lr_schedule_values: list, loss_scaler, max_norm: float = 0,
        update_freq: int = 1, model_ema: Optional[ModelEma] = None,
        log_writer: Optional[utils.TensorboardLogger] = None,
        task=None, mixup_fn=None, **kwargs
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and tensor_key.startswith("image"):
                data[tensor_key] = data[tensor_key].half()

        # mixup for imagenet finetuning
        if mixup_fn is not None:
            data["image"], data["label"] = mixup_fn(data["image"], data["label"])

        if task in ["coco_captioning", "nocaps"]:
            data["global_step"] = global_step

        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model, **data)

        loss = results.pop("loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value,
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value,
                "lr": max_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay_value,
                "grad_norm": grad_norm,
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, handler, build_ranking=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval(data_loader, build_ranking=build_ranking)

@torch.no_grad()
def evaluate_onalldata(query_dataloader, answer_dataloader, model, device, handler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger)

    # build query embeddings
    if args.retrieval_mode == 'text_to_image':




        for data in tqdm(query_dataloader):
            for tensor_key in data.keys():
                data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                handler.eval_batch(model=model, mode='text', **data)


        if args.load_image_from_precomputed_npy:
            freq = 200
            if 'base' in args.model:
                embeddings_path = f"embeddings/image/beit3_large_patch16_224_checkpoint-5.pth"
                for id in range(0, 10600, freq):
                    print(f"load image embeddings from {id} to {id + freq}")
                    file_name = os.path.join(embeddings_path, f"image_feats_{id}_freq_100_gpu_0.npy")
                    embeddings = np.load(file_name, allow_pickle=True)
                    handler.image_feats.extend(torch.from_numpy(embeddings))


                # handler.image_feats = torch.from_numpy(handler.image_feats)
                handler.image_ids.extend(torch.arange(len(handler.image_feats)).tolist())  # handler.image_feats.shape[0]

            elif 'large' in args.model:
                embeddings_path = f"embeddings/image/beit3_large_patch16_224_checkpoint-5.pth"
                for id in range(0, 1800, freq): #13000
                    print(f"load image embeddings from {id} to {id + freq}")
                    file_name = os.path.join(embeddings_path, f"image_feats_{id}_freq_200_gpu_0.npy")
                    embeddings = np.load(file_name, allow_pickle=True)
                    handler.image_feats.extend(torch.from_numpy(embeddings))
                handler.image_ids.extend(torch.arange(len(handler.image_feats)).tolist()) # handler.image_feats.shape[0]



        else:
            for data in tqdm(answer_dataloader):
                for tensor_key in data.keys():
                    data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    handler.eval_batch(model=model, mode='image', **data)
                if args.save_embeddings_to_npy:
                    if len(handler.image_feats) % handler.store_freq == 0:
                        handler.store_feats(mode='image', tag=args.model + '_' + args.finetune.split('/')[-1], gpu_id=0)
                        handler.store_pointer += handler.store_freq



    elif args.retrieval_mode == 'image_to_text':
        raise NotImplementedError

    # build query embeddings

    if args.eval:
        # clear GPU memory
        del model
        query_dataloader = None
        answer_dataloader = None
        torch.cuda.empty_cache()
    return handler.after_eval(query_dataloader, answer_dataloader, args.retrieval_mode, args)

@torch.no_grad()
def evaluate_submission(query_dataloader, answer_dataloader, model, device, handler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    handler.before_eval(metric_logger=metric_logger)

    # build query embeddings
    if args.retrieval_mode == 'text_to_image':
        for data in tqdm(query_dataloader):
            for tensor_key in data.keys():
                data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                handler.eval_batch(model=model, mode='text', **data)

        if args.load_embeddings_from_npy:
            # print(sorted(os.listdir(args.embeddings_file_path)))
            freq = 3000
            paths = [f'image_feats_{pointer}_freq_3000_gpu_0.npy' for pointer in range(0, 150001, freq)]
            for file in paths:
                if file.endswith('.npy'):
                    handler.image_feats.append(np.load(os.path.join(args.embeddings_file_path, file)))


            # temoproary
            for data in tqdm(answer_dataloader):
                for tensor_key in data.keys():
                    data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    handler.eval_batch(model=model, mode='image', **data)

                if len(handler.image_feats) % handler.store_feq == 0:
                    if args.dist_eval:
                        handler.store_feats(mode='image', tag=args.model + '_' + args.finetune.split('/')[-1],
                                            gpu_id=args.gpu)
                    else:
                        handler.store_feats(mode='image', tag=args.model + '_' + args.finetune.split('/')[-1], gpu_id=0)
                    handler.store_pointer += handler.store_feq

        else:
            for data in tqdm(answer_dataloader):
                for tensor_key in data.keys():
                    data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    handler.eval_batch(model=model, mode='image', **data)

                if len(handler.image_feats) % handler.store_feq == 0:
                    if args.dist_eval:
                        handler.store_feats(mode='image', tag=args.model + '_' + args.finetune.split('/')[-1],
                                            gpu_id=args.gpu)
                    else:
                        handler.store_feats(mode='image', tag=args.model + '_' + args.finetune.split('/')[-1], gpu_id=0)
                    handler.store_pointer += handler.store_feq

    elif args.retrieval_mode == 'image_to_text':
        for data in tqdm(query_dataloader):
            for tensor_key in data.keys():
                data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                handler.eval_batch(model=model, mode='image', **data)

        if args.load_embeddings_from_npy:
            # laod text embeddings from npy files
            for file in os.listdir(args.embeddings_file_path):
                if file.endswith('.npy'):
                    handler.text_feats.append(np.load(os.path.join(args.embeddings_file_path, file)))
            handler.text_feats = np.concatenate(handler.text_feats, axis=0)
            handler.text_feats = torch.from_numpy(handler.text_feats)
            handler.text_ids = torch.arange(handler.text_feats.shape[0])
        else:
            for data in tqdm(answer_dataloader):
                for tensor_key in data.keys():
                    data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    handler.eval_batch(model=model, mode='text', **data)
                if len(handler.text_feats) % handler.store_feq == 0:
                    if args.dist_eval:
                        handler.store_feats(mode='text',
                                            tag=args.model + '_' + args.finetune.split('/')[-1], gpu_id=args.gpu)
                    else:
                        handler.store_feats(mode='text', tag=args.model + '_' + args.finetune.split('/')[-1], gpu_id=0)
                    handler.store_pointer += handler.store_feq
    else:
        raise NotImplementedError

    # build query embeddings

    return handler.after_eval(query_dataloader, answer_dataloader, args.retrieval_mode, args)
