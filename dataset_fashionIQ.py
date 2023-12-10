#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import itertools
import json as jsonmod
import os

import torch

from config import FASHIONIQ_ANNOTATION_DIR, FASHIONIQ_IMAGE_DIR
from dataset import MyDataset


class FashionIQDataset(MyDataset):
	"""
	FashionIQ dataset, introduced in "Fashion IQ: A new dataset towards
	retrieving images by natural language feedback"; Hui Wu, Yupeng Gao,
	Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio
	Feris; Proceedings of CVPR, pp. 11307–11317, 2021.
	"""

	def __init__(self, split, vocab, transform, what_elements='triplet', load_image_feature=0,
					fashion_categories='all', ** kw):
		"""
		Args:
			fashion_categories: fashion_categories to consider. Expected to be a string such as : "dress toptee".
		"""
		MyDataset.__init__(self, split, FASHIONIQ_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, ** kw)

		fashion_categories = ['dress', 'shirt', 'toptee'] if fashion_categories=='all' else sorted(fashion_categories.split())

		# concatenate in one list the image identifiers of the fashion categories to consider
		image_id2name_files = [os.path.join(FASHIONIQ_ANNOTATION_DIR, 'image_splits', f'split.{fc}.{split}.json') for fc in fashion_categories]
		image_id2name = [self.load_file(a) for a in image_id2name_files]
		self.image_id2name = list(itertools.chain.from_iterable(image_id2name))

		# if necessary, load triplet annotations of the fashion categories to consider
		if self.what_elements in ["query", "triplet"]:
			prefix = 'pair2cap' if split=='test' else 'cap'
			annotations_files = [os.path.join(FASHIONIQ_ANNOTATION_DIR, 'captions', f'{prefix}.{fc}.{split}.json') for fc in fashion_categories]
			annotations = [self.load_file(a) for a in annotations_files]
			self.annotations = list(itertools.chain.from_iterable(annotations))
		
			# DEBUG: We only use a small part of dataset to search hyper-parameters!!!
			# if self.what_elements == "triplet":
			# 	import random
			# 	ratio = 0.2
			# 	num_samples = round(ratio * len(self.annotations))
			# 	self.annotations = random.choices(self.annotations, k=num_samples)

		# ADD: build the noun classifier weights
		if self.what_elements == "triplet":
			noun_tokens = []
			for annotation in self.annotations:
				noun_tokens.append(self.get_transformed_nouns(self.get_token(annotation)))
			FashionIQDataset.noun_tokens = torch.cat(noun_tokens).unique().long()
			# Construct word_to_idx for generate one-hot labels
			FashionIQDataset.noun_token_to_idx = {}
			for i, token in enumerate(self.noun_tokens):
				FashionIQDataset.noun_token_to_idx[token.item()] = i

		if self.what_elements != "target":
			self.mil_labels = torch.zeros([len(self.annotations), len(FashionIQDataset.noun_tokens)])
			for i, annotation in enumerate(self.annotations):
				mil_label_tokens = self.get_transformed_nouns(self.get_token(annotation))
				for token in mil_label_tokens:
					k = FashionIQDataset.noun_token_to_idx.get(token.item())
					if k is not None:
						#self.mil_labels[i, CIRRDataset.noun_token_to_idx[token.item()]] = 1
						self.mil_labels[i, k] = 1

	def get_token(self, ann):
		res = ann["noun"]
		res += ann["verb"]
		res += ann["adj"]
		res += ann["adv"]
		return res

	def __len__(self):
		if self.what_elements=='target':
			return len(self.image_id2name)
		return 2*len(self.annotations) # 1 annotation = 2 captions = 2 queries/triplets


	def load_file(self, f):
		"""
		Depending on the file, returns:
			- a list of dictionaries with the following format:
				{'target': 'B001AS562I', 'candidate': 'B0088WRQVS', 'captions': ['i taank top', 'has spaghetti straps']}
			- a list of image identifiers
		"""
		with open(f, "r") as jsonfile:
			ann = jsonmod.loads(jsonfile.read())
		return ann


	############################################################################
	# *** GET ITEM METHODS
	############################################################################

	def get_triplet(self, idx):

		# NOTE: following CoSMo (Lee et. al, 2021), we consider the two captions
		# of each reference-target pair separately, doubling the number of
		# queries
		index = idx // 2 # get the annotation index
		cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)

		# get data
		ann = self.annotations[index]

		capts = ann['captions'][cap_slice]
		text, real_text = self.get_transformed_captions(capts)

		path_src = ann['candidate'] + ".png"
		img_src = self.get_transformed_image(path_src)

		path_trg = ann['target'] + ".png"
		img_trg = self.get_transformed_image(path_trg)

		noun = self.get_transformed_nouns(self.get_token(ann))
		mil_labels = self.mil_labels[index]

		return img_src, text, img_trg, real_text, idx, noun, mil_labels


	def get_query(self, idx):

		# NOTE: following CoSMo (Lee et. al, 2021), we consider the two captions
		# of each reference-target pair separately, doubling the number of
		# queries
		index = idx // 2 # get the annotation index
		cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)

		# get data
		ann = self.annotations[index]

		capts = ann['captions'][cap_slice]
		text, real_text = self.get_transformed_captions(capts)

		path_src = ann['candidate'] + ".png"
		img_src = self.get_transformed_image(path_src)
		img_src_id = self.image_id2name.index(ann['candidate'])

		img_trg_id = [self.image_id2name.index(ann['target'])]

		noun = self.get_transformed_nouns(self.get_token(ann))
		mil_labels = self.mil_labels[index]

		return img_src, text, img_src_id, img_trg_id, real_text, idx, noun, mil_labels


	def get_target(self, index):

		img_id = index
		path_img = self.image_id2name[index] + ".png"
		img = self.get_transformed_image(path_img)

		return img, img_id, index


	############################################################################
	# *** FORMATTING INFORMATION FOR VISUALIZATION PURPOSES
	############################################################################

	def get_triplet_info(self, index):
		"""
		Should return 3 strings:
			- the text modifier
			- an identification code (name, relative path...) for the reference image
			- an identification code (name, relative path...) for the target image
		"""
		ann = self.annotations[index]
		return " [and] ".join(ann["captions"]), ann["candidate"], ann["target"]