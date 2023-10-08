import os
import json
import random

import torch
import glob
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, \
    IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform

from beit3_tools import utils
from beit3_tools.glossary import normalize_word
from beit3_tools.randaug import RandomAugment

# AtoMiC
from datasets import load_dataset
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

class BaseAtoMicDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path=None, load_tokenized_text=False,
                  load_image_from_huggingface_hub=True, ):


        self.load_image_from_huggingface_hub = load_image_from_huggingface_hub
        if self.load_image_from_huggingface_hub:
            print("load images data from huggingface hub: ")
            if args.cluster:
                self.images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train', cache_dir='../../datasets/ATOMIC/',
                                           num_proc=4)
            else:
                self.images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train',
                                           num_proc=4)
            print("build index to map image_id to index in self.images: ")
            image_ids = self.images['image_id']
            self.image_id2row_dict = self._getRowIdx(image_ids)
        else:
            print("load images data from local file: ------------ ")
        # else:
        #     image_ids = json.loads(open("../datasets/ATOMIC/all_image_ids.json").read())
        #     self.image_id2row_dict = self._getRowIdx(image_ids)

        if not load_tokenized_text:
            if not args.use_entity and not args.use_summary:
                print("load texts data from huggingface hub: ")
                if args.cluster:
                    self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train', cache_dir='../../datasets/ATOMIC/',
                                              num_proc=4)
                else:
                    self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train',
                                              num_proc=4)
                print("build index to map text_id to index in self.texts: ")
                text_ids = self.texts['text_id']
                self.text_id2row_dict = self._getRowIdx(text_ids)



    def _getRowIdx(self, id_list: list) -> dict:
        id2pos = {}
        for pos, _id in tqdm(enumerate(id_list), total=len(id_list)):
            id2pos[_id] = pos
        return id2pos


class AtoMicDataset(BaseAtoMicDataset):
    def __init__(self, args, split, transform,
                 tokenizer, num_max_bpe_tokens, text_features=None,
                 data_path=None, load_tokenized_text=True):
        # load dataset from huggingface
        super().__init__(args, data_path, load_tokenized_text=args.load_tokenized_text,
                         load_image_from_huggingface_hub=args.load_image_from_huggingface_hub)

        self.args = args
        print("load qrels data from huggingface hub: ")
        # split = 'test'
        # self.qrels = load_dataset("TREC-AToMiC/AToMiC-Qrels-v0.2", split=split, cache_dir='../datasets/ATOMIC/', )
        self.qrels = load_dataset("TREC-AToMiC/AToMiC-Qrels-v0.2", split=split, )

        if text_features is None:
            # text_features = ["page_title", "section_title",
            #                  "context_sec_des", ] #"context_page_des",
            text_features = ["page_title", "section_title", "context_section_description",
                              ] #"context_page_description"
        self.text_features = text_features
        self.split = split
        self.data_path = data_path
        self.transform = transform

        if args.load_tokenized_text:
            print("load tokenized text data from json file: ")
            self.texts = []
            if not args.eval:
                if data_path is None:
                    data_path = f"../datasets/ATOMIC/Atomic_text_tokenized_{split}.json"
                else:
                    data_path = os.path.join(data_path, f"Atomic_text_tokenized_{split}.json")
            else:
                data_path = f"../datasets/ATOMIC/Atomic_text_tokenized_{split}.json" #_all_fileds

            with open(data_path, "r") as reader:
                for line in tqdm(reader):
                    data = json.loads(line)
                    self.texts.append(data)


        elif args.use_entity:
            print("load entity data from json file: ")
            with open(f"datasets/atomic_{split}_entities.json", "r") as reader:
                self.entities = json.load(reader)
            with open(f"datasets/atomic_{split}_summaries.json", "r") as reader:
                self.summaries = json.load(reader)

        elif args.use_summary:
            print("load summary data from json file: ")
            with open(f"datasets/atomic_{split}_summaries.json", "r") as reader:
                self.summaries = json.load(reader)


        self.load_tokenized_text = args.load_tokenized_text

        # BEIT3 parameters
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = args.captioning_mask_prob

    #   self.check_dataset_intergrity()

    def check_dataset_intergrity(self):
        print("check dataset intergrity: ")
        for image_id in self.qrels['image_id']:
            idx = self.image_id2row_dict[image_id]
            try:
                iamge = self.images[idx]['image']
                # image = self.transform(image)
            except:
                print(f"image_id: {image_id} does not have image")

        sys.exit()
        return True

    def _getRowIdx(self, id_list: list) -> dict:
        id2pos = {}
        for pos, _id in tqdm(enumerate(id_list), total=len(id_list)):
            id2pos[_id] = pos
        return id2pos

    def _split_images(self, images, qrels):
        image_ids = set(qrels["image_id"])
        valid_image_ids = []
        for idx, image_id in tqdm(enumerate(images["image_id"]), total=len(images)):
            if image_id in image_ids:
                valid_image_ids.append(images[idx])
        return valid_image_ids

    def _get_image(self, image_id):
        if self.load_image_from_huggingface_hub:
            idx = self.image_id2row_dict[image_id]
            image = self.images[idx]['image']
            image = self.transform(image)
            return image
        else:

            # image = np.load(os.path.join('../datasets/ATOMIC/images', self.split, f"{image_id}.npy"))
            if self.args.cluster:
                img_path = os.path.join('../datasets/ATOMIC/images', self.split, f"{image_id}.jpg")
            else:
                img_path = os.path.join('\\\\longkukunas\\PhD data\\datasets\\ATOMIC\\images', self.split, f"{image_id}.jpg")

            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image

    def _tokenize_text(self, new_dict):
        result = []
        if self.args.use_entity:
            content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(new_dict)))
            result += content


        elif self.args.use_summary:
            content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(new_dict))
            result += content

        else:
            for key in self.text_features:
                content = new_dict[key]
                content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))
                if len(content) > self.num_max_bpe_tokens:
                    content = content[:self.num_max_bpe_tokens]

                result += content
                if len(result) >= (self.num_max_bpe_tokens - 2) :
                    break
        return result

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        # retreive image and text pair from qrels
        item = self.qrels[index]
        img_id = item['image_id']
        text_id = item['text_id']
        # data["image_id"] = self.image_id2row_dict[img_id]
        data["image_id"] = index
        # data["text_id"] = self.text_id2row_dict[text_id]
        data["image"] = self._get_image(img_id)
        data["text_id"] = index

        if self.args.use_entity:
            text_dict = self.entities[index]
            if len(text_dict) == 0:
                text_dict = self.summaries[index]

        elif self.args.use_summary:
            text_dict = self.summaries[index]
        else:
            text_dict = self.texts[self.text_id2row_dict[text_id]]

        if not self.load_tokenized_text:
            text_segment = self._tokenize_text(text_dict)
            # text_segment = [token for feature in self.text_features for token in text_dict[feature]]
        else:
            text_segment = [token for feature in self.text_features for token in text_dict[f"{feature}_tokens_ids"]]

        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

        return data

    def _get_img_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.qrels[idx]['image_id'])
            return id_list
        elif isinstance(index, int):
            return self.qrels[index]['image_id']
        else:
            raise ValueError("index should be either list or int")

    def _get_text_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.qrels[idx]['text_id'])
            return id_list

        elif isinstance(index, int):
            return self.qrels[index]['text_id']
        else:
            raise ValueError("index should be either list or int")

    def __getitem__(self, index):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self):
        # return 1000
        # print("The length of qrels is",len(self.qrels))
        return len(self.qrels)


class AtoMicDatasetanswer(torch.utils.data.Dataset):
    def __init__(self, args, transform,
                 tokenizer, num_max_bpe_tokens, text_features=None,
                 data_path=None, load_tokenized_text=True):

        # super().__init__(data_path, load_tokenized_text=args.load_tokenized_text,
        #                  load_image_from_huggingface_hub=args.load_image_from_huggingface_hub)
        super().__init__()

        self.args = args
        self.load_image_from_huggingface_hub = args.load_image_from_huggingface_hub
        self.load_tokenized_text = args.load_tokenized_text

        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = args.captioning_mask_prob
        if text_features is None:
            # text_features = ["page_title", "section_title",
            #                  "context_sec_des", ] #"context_page_des",
            text_features = ["page_title", "section_title", "context_section_description",
                              ] #"context_page_description"

        self.text_features = text_features

        self.transform = transform

        self.retrieval_mode = args.retrieval_mode

        if self.retrieval_mode == "text_to_image":
            if args.load_image_from_huggingface_hub:
                # load all images
                print("load images data from huggingface hub: ")
                # self.images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train', cache_dir='../datasets/ATOMIC/',
                #                            num_proc=4)
                self.images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train',
                                           num_proc=4)
                print("build index to map image_id to index in self.images: ")
                self.image_ids = self.images['image_id']
                self.image_id2row_dict = self._getRowIdx(self.image_ids)

                with open(f"yang_datasets/retrieval_image_ids.json", "r") as reader:
                    self.retrieval_image_ids = json.load(reader)

            elif args.load_image_from_precomputed_npy:
                print("load images data from precomputed numpy file: ")

            else:
                print("load images data from local: ")
                self.image_ids = np.load('../datasets/ATOMIC/all_image_ids.npy')

        elif self.retrieval_mode == "image_to_text":
            if args.cluster:
                self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train', cache_dir='../../datasets/ATOMIC/',
                                          num_proc=4)
            else:
                self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train',
                                          num_proc=4)
            self.text_ids = self.texts['text_id']

    def _getRowIdx(self, id_list: list) -> dict:
        id2pos = {}
        for pos, _id in tqdm(enumerate(id_list), total=len(id_list)):
            id2pos[_id] = pos
        return id2pos

    def _get_img_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.image_ids[idx])
            return id_list
        elif isinstance(index, int):
            return self.image_ids[index]
        else:
            raise ValueError("index should be either list or int")

    def _get_text_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.texts[idx]['text_id'])
            return id_list

        elif isinstance(index, int):
            return self.texts[index]['text_id']
        else:
            raise ValueError("index should be either list or int")

    def _get_image(self, idx):
        if self.load_image_from_huggingface_hub:
            idx = self.image_id2row_dict[self.retrieval_image_ids[idx]]
            image = self.images[idx]['image']
            image = self.transform(image)
            return image
        else:
            image_id = self.image_ids[idx]
            if self.args.cluster:
                img_path = os.path.join(f"../datasets/ATOMIC/all_images\\{image_id}.jpg")
            else:
                img_path = f"..\\datasets\\ATOMIC\\all_images\\{image_id}"#.jpg
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image

    def _tokenize_text(self, new_dict):
        result = []
        for key in self.text_features:
            content = new_dict[key]
            content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))
            if len(content) > self.num_max_bpe_tokens:
                content = content[:self.num_max_bpe_tokens]

            result += content
            if len(result) >= (self.num_max_bpe_tokens - 2) :
                break
        return result

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens


    def __getitem__(self, index):
        if self.retrieval_mode == "text_to_image":
            data = dict()
            if not self.args.load_image_from_precomputed_npy:
                data["image_id"] = index
                data["image"] = self._get_image(index)

            return data

        else:
            data = dict()
            data["text_id"] = index
            text_dict = self.texts[index]
            # if not self.load_tokenized_text:
            #     text_dict = self._tokenize_text(text_dict)
            if not self.load_tokenized_text:
                text_segment = self._tokenize_text(text_dict)
                # text_segment = [token for feature in self.text_features for token in text_dict[feature]]
            else:
                text_segment = [token for feature in self.text_features for token in text_dict[f"{feature}_tokens_ids"]]

            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["language_tokens"] = language_tokens
            data["padding_mask"] = padding_mask
            return data

    def __len__(self):
        # return 300000
        if self.retrieval_mode == "text_to_image":
            return len(self.retrieval_image_ids)
            # return len(self.image_ids)
        else:
            return len(self.text_ids)

class AtoMicDatasetquery(torch.utils.data.Dataset):
    def __init__(self, args, transform,
                 tokenizer, num_max_bpe_tokens, text_features=None,
                 data_path=None, load_tokenized_text=True):

        # super().__init__(data_path, load_tokenized_text=args.load_tokenized_text,
        #                  load_image_from_huggingface_hub=args.load_image_from_huggingface_hub)
        super().__init__()
        self.args = args
        self.load_tokenized_text = args.load_tokenized_text
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = args.captioning_mask_prob
        if text_features is None:
            # text_features = ["page_title", "section_title",
            #                  "context_sec_des", ] #"context_page_des",
            text_features = ["page_title",
                              ] #"context_page_description" "section_title", "context_section_description",

        self.text_features = text_features

        self.transform = transform

        self.retrieval_mode = args.retrieval_mode

        split = 'test'

        if args.use_entity:
            print("load entity data from json file: ")
            with open(f"datasets/atomic_{split}_entities_precise.json", "r") as reader:
                self.entities = json.load(reader)
            with open(f"datasets/atomic_{split}_summaries.json", "r") as reader:
                self.summaries = json.load(reader)
            if args.produce_stage1_candidates:
                single_entity = []
                for group_entity in self.entities:
                    for entity in group_entity:
                        single_entity.append(entity)
                self.entities = single_entity

        elif args.use_summary:
            print("load summary data from json file: ")
            with open(f"datasets/atomic_{split}_summaries.json", "r") as reader:
                self.summaries = json.load(reader)

        # elif args.produce_stage1_candidates:
        #     with open('summary/atomic_test_entities_set.json', 'r') as f:
        #         self.entities = json.load(f)

        if args.query_url is None:
            self.querys = load_dataset("TREC-AToMiC/AToMiC-Qrels-v0.2", split='test', )
            if args.cluster:
                self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train',
                                          cache_dir='../../datasets/ATOMIC/',
                                          num_proc=4)
            else:
                self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train', )

            text_ids = self.texts['text_id']
            self.text_id2row_dict = self._getRowIdx(text_ids)
        else:
            self.querys = load_dataset(args.query_url, split='train',
                                       num_proc=4)

        if self.retrieval_mode == "text_to_image":
            if args.cluster:
                with open('datasets/imageids2pos.json', 'r') as f:
                    self.imageid2pos = json.load(f)
            else:
                with open('../datasets/ATOMIC/imageids2pos.json', 'r') as f:
                    self.imageid2pos = json.load(f)

        with open(f"yang_datasets/retrieval_image_ids.json", "r") as reader:
            self.retrieval_image_ids = json.load(reader)
        self.retrieval_image_ids2pos = self._getRowIdx(self.retrieval_image_ids)

    def _get_img_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.querys[idx]['image_id'])
            return id_list
        elif isinstance(index, int):
            return self.querys[index]['image_id']
        else:
            raise ValueError("index should be either list or int")

    def _get_text_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.querys[idx]['text_id'])
            return id_list

        elif isinstance(index, int):
            return self.querys[index]['text_id']
        else:
            raise ValueError("index should be either list or int")

    def _getRowIdx(self, id_list: list) -> dict:
        id2pos = {}
        for pos, _id in tqdm(enumerate(id_list), total=len(id_list)):
            id2pos[_id] = pos
        return id2pos

    def _get_image(self, idx):
        image = self.querys[idx]['image']
        image = self.transform(image)
        return image

    def _tokenize_text(self, new_dict):
        result = []
        if self.args.use_entity :
            content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(new_dict)))
            result += content

        elif self.args.produce_stage1_candidates:
            content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(new_dict))
            result += content

        elif self.args.use_summary:
            content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(new_dict))
            result += content

        else:
            for key in self.text_features:
                content = new_dict[key]
                content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))
                if len(content) > self.num_max_bpe_tokens:
                    content = content[:self.num_max_bpe_tokens]

                result += content
                if len(result) >= (self.num_max_bpe_tokens - 2) :
                    break
        return result

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens


    def __getitem__(self, index):
        # index = 1727 + index
        if self.retrieval_mode == "text_to_image":
            # query is text, return text
            data = dict()

            if self.args.produce_stage1_candidates:
                data["text_id"] = index
            else:
                image_id = self.querys[index]['image_id']
                # image_pos = self.imageid2pos[image_id]
                image_pos = self.retrieval_image_ids2pos[image_id]
                data["text_id"] = image_pos


            if self.args.query_url is None:
                if self.args.use_entity or self.args.produce_stage1_candidates:
                    text_dict = self.entities[index]
                    if len(text_dict) == 0:
                        text_dict = self.summaries[index]

                elif self.args.use_summary:
                    text_dict = self.summaries[index]
                else:
                    text_id = self.querys[index]['text_id']
                    text_dict = self.texts[self.text_id2row_dict[text_id]]
            else:
                text_dict = self.querys[index]
            # if not self.load_tokenized_text:
            #     text_dict = self._tokenize_text(text_dict)
            # print(index, text_dict)

            # if index == 0:
            #     text_dict = 'clear nights'  #'Tribute in Light is an art installation created in remembrance of the September 11 attacks. It consists of 88 vertical searchlights arranged in two columns of light to represent the Twin Towers. It stands six blocks south of the World Trade Center on top of the Battery Parking Garage in New York City.' #'New York City'
            # elif index == 1:
            #     text_dict = 'Northern New Jersey' #It stands six blocks south of the World Trade Center'
            # elif index == 2:
            #     text_dict = 'Searchlights' #'Tribute in Light is an art installation created in remembrance of the September 11 attacks.'
            # elif index == 3:
            #     text_dict = 'Space Cannon' #'It stands six blocks south of the World Trade Center on top of the Battery Parking Garage in New York City.'
            #


            if not self.load_tokenized_text:
                text_segment = self._tokenize_text(text_dict)
                # text_segment = [token for feature in self.text_features for token in text_dict[feature]]
            else:
                text_segment = [token for feature in self.text_features for token in text_dict[f"{feature}_tokens_ids"]]

            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["language_tokens"] = language_tokens
            data["padding_mask"] = padding_mask
            return data


        else:
            data = dict()
            data["image_id"] = index
            data["image"] = self._get_image(index)
            return data


    def __len__(self):
        if self.args.produce_stage1_candidates:
            return len(self.entities)
        else:
            return len(self.querys)



task2dataset = {
    "atomic": AtoMicDataset,
    "atomic_allret_adapter": AtoMicDataset,
    'atomic_stage1': AtoMicDataset,

}

def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)


def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )

    else:
        sampler = torch.utils.data.SequentialSampler(dataset,)

    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(is_train, args):
    if args.task in ["imagenet"]:
        return build_imagenet_transform(is_train, args)

    if is_train:
        t = [
            RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0),
                                              interpolation=args.train_interpolation),
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t



def create_dataset_by_split(args, split, is_train=True):
    transform = build_transform(is_train=is_train, args=args)
    dataset_class = task2dataset[args.task]
    tokenizer = get_sentencepiece_model_for_beit3(args)

    opt_kwargs = {}
    if args.task in ["coco_captioning", "nocaps"]:
        opt_kwargs["mask_prob"] = args.captioning_mask_prob

    if args.task in ['atomic', "atomic_allret_adapter", 'atomic_stage1']:
        dataset = dataset_class(args=args,
                                data_path=args.data_path, split=split,
                                transform=transform, tokenizer=tokenizer,
                                num_max_bpe_tokens=args.num_max_bpe_tokens,
                                **opt_kwargs,
                                )
    else:
        dataset = dataset_class(
            data_path=args.data_path, split=split,
            transform=transform, tokenizer=tokenizer,
            num_max_bpe_tokens=args.num_max_bpe_tokens,
            task=args.task, **opt_kwargs,
        )

    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )




def create_query_answer_dataset(args):
    is_train = False
    transform = build_transform(is_train=is_train, args=args)

    tokenizer = get_sentencepiece_model_for_beit3(args)
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    # create query dataset
    dataset_class = AtoMicDatasetquery

    query_dataset = dataset_class(args=args,
                                data_path=args.data_path,
                                transform=transform, tokenizer=tokenizer,
                                num_max_bpe_tokens=args.num_max_bpe_tokens, )



    query_dataloader = create_dataloader(
        query_dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )

    # create answer dataset
    dataset_class = AtoMicDatasetanswer

    answer_dataset = dataset_class(args=args,
                                data_path=args.data_path,
                                transform=transform, tokenizer=tokenizer,
                                num_max_bpe_tokens=args.num_max_bpe_tokens, )

    answer_dataloader = create_dataloader(
        answer_dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )


    return query_dataloader, answer_dataloader


def create_answer_dataset(args):
    is_train = False
    transform = build_transform(is_train=is_train, args=args)

    tokenizer = get_sentencepiece_model_for_beit3(args)
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)


    # create answer dataset
    dataset_class = AtoMicDatasetanswer

    answer_dataset = dataset_class(args=args,
                                data_path=args.data_path,
                                transform=transform, tokenizer=tokenizer,
                                num_max_bpe_tokens=args.num_max_bpe_tokens, )

    answer_dataloader = create_dataloader(
        answer_dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )


    return  answer_dataloader


def create_downstream_dataset(args, is_eval=False, eval_on_test_set=False):
    if is_eval:
        return create_dataset_by_split(args, split="test", is_train=False)

    elif eval_on_test_set:
        return create_dataset_by_split(args, split="submission", is_train=False)

    else:
        return \
            create_dataset_by_split(args, split="train", is_train=True), \
                create_dataset_by_split(args, split="validation", is_train=False)
