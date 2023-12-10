#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import itertools
import json as jsonmod
import os

import torch

from config import SHOES_ANNOTATION_DIR, SHOES_IMAGE_DIR
from dataset import MyDataset


class ShoesDataset(MyDataset):
	"""
    Shoes dataset, introduced with "Dialog-based interactive image retrieval";
    Xiaoxiao Guo, Hui Wu, Yu Cheng, Steven Rennie, Gerald Tesauro, and Rogerio
    Feris; Proceedings of NeurIPS, pp. 676–686, 2018.

	Images are extracted from "Automatic attribute discovery and
	characterization from noisy web data"; Tamara L Berg, Alexander C Berg, and
	Jonathan Shih; Proc. ECCV, pp. 663–676, 2010.
	"""

	def __init__(self, split, vocab, transform, what_elements='triplet', load_image_feature=0, ** kw):
		"""
		Args:
			split: either "train", "val" : used to know if to do text augmentation
			vocab: vocabulary wrapper.
			transform: tuple (transformer for image, opt.img_transform_version)
			what_elements: specifies what to provide when iterating over the dataset (queries, targets ?)
			load_image_feature: whether to load raw images (if 0, default) or pretrained image feature (if > 0, size of the feature)
		"""

		MyDataset.__init__(self, split, SHOES_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, ** kw)

		# load the paths of the images involved in the split
		self.image_id2name = self.load_file(os.path.join(SHOES_ANNOTATION_DIR, f'split.{split}.json'))

		# if necessary, load triplet annotations
		if self.what_elements in ["query", "triplet"]:
			self.annotations = self.load_file(os.path.join(SHOES_ANNOTATION_DIR, f'triplet.{split}.json'))

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
			ShoesDataset.noun_tokens = torch.cat(noun_tokens).unique().long()
			# Construct word_to_idx for generate one-hot labels
			ShoesDataset.noun_token_to_idx = {}
			for i, token in enumerate(self.noun_tokens):
				ShoesDataset.noun_token_to_idx[token.item()] = i

		if self.what_elements != "target":
			self.mil_labels = torch.zeros([len(self.annotations), len(ShoesDataset.noun_tokens)])
			for i, annotation in enumerate(self.annotations):
				mil_label_tokens = self.get_transformed_nouns(self.get_token(annotation))
				for token in mil_label_tokens:
					k = ShoesDataset.noun_token_to_idx.get(token.item())
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
		return len(self.annotations)


	def load_file(self, f):
		"""
		Depending on the file, returns:
			- a list of dictionaries with the following format:
				{'ImageName': '{path_from_{img_dir}}/img_womens_clogs_851.jpg', 'ReferenceImageName': '{path_from_{img_dir}}/img_womens_clogs_512.jpg', 'RelativeCaption': 'is more of a textured material'}
			- a list of image identifiers/paths
		"""
		with open(f, "r") as jsonfile:
			ann = jsonmod.loads(jsonfile.read())
		return ann


	############################################################################
	# *** GET ITEM METHODS
	############################################################################

	def get_triplet(self, index):

		ann = self.annotations[index]

		capts = ann['RelativeCaption']
		text, real_text = self.get_transformed_captions([capts])

		path_src = ann['ReferenceImageName']
		img_src = self.get_transformed_image(path_src)

		path_trg = ann['ImageName']
		img_trg = self.get_transformed_image(path_trg)
		
		noun = self.get_transformed_nouns(self.get_token(ann))
		mil_labels = self.mil_labels[index]

		return img_src, text, img_trg, real_text, index, noun, mil_labels


	def get_query(self, index):

		ann = self.annotations[index]

		capts = ann['RelativeCaption']
		text, real_text = self.get_transformed_captions([capts])

		path_src = ann['ReferenceImageName']
		img_src = self.get_transformed_image(path_src)
		img_src_id = self.image_id2name.index(ann['ReferenceImageName'])

		img_trg_id = [self.image_id2name.index(ann['ImageName'])]

		noun = self.get_transformed_nouns(self.get_token(ann))
		mil_labels = self.mil_labels[index]

		return img_src, text, img_src_id, img_trg_id, real_text, index, noun, mil_labels


	def get_target(self, index):

		img_id = index
		path_img = self.image_id2name[index]
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
		return ann["RelativeCaption"], ann["ReferenceImageName"], ann["ImageName"]
