
import torch
import torch.nn as nn
import torch.nn.functional as F


class CptCriterion(nn.Module):
    def __init__(self):
        super(CptCriterion, self).__init__()
        self.tirg = LossModule()
        self.mil = FocalLossWithLogitsNegLoss()


    def forward(self, scores, logits, labels):
        matching_loss = self.tirg(scores)
        mil_loss = self.mil(logits, labels)

        return matching_loss, mil_loss

class LossModule(nn.Module):

    def __init__(self):
        super(LossModule, self).__init__()

    def forward(self, scores):
        """
        Loss based on Equation 6 from the TIRG paper,
        "Composing Text and Image for Image Retrieval - An Empirical Odyssey", CVPR19
        Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays
        
        Args:
            scores: matrix of size (batch_size, batch_size), where coefficient
                (i,j) corresponds to the score between query (i) and target (j).
                Ground truth associations are represented along the diagonal.
        """

        # build the ground truth label tensor: the diagonal corresponds to
        # correct classification
        GT_labels = torch.arange(scores.shape[0]).long()
        GT_labels = torch.autograd.Variable(GT_labels)
        if torch.cuda.is_available():
            GT_labels = GT_labels.cuda()

        # compute the cross-entropy loss
        loss = F.cross_entropy(scores, GT_labels, reduction = 'mean')

        return loss


class FocalLossWithLogitsNegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5
        self.gamma = 1.0
        self.factor = 200

    def extra_repr(self):
        return 'alpha={}, gamma={}'.format(self.alpha, self.gamma)

    def forward(self, pred, target):
        sigmoid_pred = pred.sigmoid()
        log_sigmoid = torch.nn.functional.logsigmoid(pred)
        loss = (target == 1) * self.alpha * torch.pow(1. - sigmoid_pred, self.gamma) * log_sigmoid

        log_sigmoid_inv = torch.nn.functional.logsigmoid(-pred)
        loss += (target == 0) * (1 - self.alpha) * torch.pow(sigmoid_pred, self.gamma) * log_sigmoid_inv

        return -loss.mean() * self.factor
