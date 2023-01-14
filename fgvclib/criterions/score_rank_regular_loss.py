import torch
import torch.nn as nn

from fgvclib.criterions import criterion

class ScoreRankRegularLoss(nn.Module):
    
    def __init__(self):

        super(ScoreRankRegularLoss, self).__init__()
        self.softmax_layer = nn.Softmax(dim=1).cuda()
        self.rank_criterion = nn.MarginRankingLoss(margin=0.05)
    
    def forward(self, self_logits, other_logits, batch_size, labels1, labels2):
        self_scores = self.softmax_layer(self_logits)[torch.arange(2*batch_size).cuda().long(),
                                                         torch.cat([labels1, labels2], dim=0)]
        other_scores = self.softmax_layer(other_logits)[torch.arange(2*batch_size).cuda().long(),
                                                         torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones([2*batch_size, ]).cuda()
        rank_loss = self.rank_criterion(self_scores, other_scores, flag)
        return rank_loss
    
@criterion("score_rank_regular_loss")
def score_rank_regular_loss(cfg=None):
    return ScoreRankRegularLoss()
    