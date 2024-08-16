import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
        self.sigmoid = nn.Sigmoid()

    def forward(self, predict, target):
        predict = self.sigmoid(predict)
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        # score = 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

# pred = torch.tensor([[0.8,0],[0,0.7]])
# label = torch.tensor([[1,0],[0,1]])
# loss = DiceLoss()
# print(loss(pred,label))a


class DiceLoss_test(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
        self.sigmoid = nn.Sigmoid()

    def forward(self, predict, target):
        assert predict.size() == target.size()
        dice_list = []
        for i in range(8):
            union = predict[i] * target[i]
            dice = 2 * torch.sum(union) / (torch.sum(predict[i]) + torch.sum(target[i]))
            dice_list.append(dice)
            dice_list = torch.tensor(dice_list)
        # score = 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return torch.mean(dice_list)
    