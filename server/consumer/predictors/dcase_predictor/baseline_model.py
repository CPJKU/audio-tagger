"""This module contains the neural network architecture used
by the module ``dcase_predictor``

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, device='cpu'):
        super(Net, self).__init__()

        self.device = device

        self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.do1 = nn.Dropout2d(0.3)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.do2 = nn.Dropout2d(0.3)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.do3 = nn.Dropout2d(0.3)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.do4 = nn.Dropout2d(0.3)
        self.conv7 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(384)
        self.do5 = nn.Dropout2d(0.3)
        self.conv8 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv8_bn = nn.BatchNorm2d(384)
        self.do6 = nn.Dropout2d(0.3)

        self.conv9 = nn.Conv2d(384, 512, 3, stride=1, padding=1)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv10_bn = nn.BatchNorm2d(512)
        self.do7 = nn.Dropout2d(0.3)

        self.conv11 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv12_bn = nn.BatchNorm2d(512)
        self.do8 = nn.Dropout2d(0.3)

        self.conv13 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv13_bn = nn.BatchNorm2d(512)
        self.do9 = nn.Dropout2d(0.5)
        self.conv14 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv14_bn = nn.BatchNorm2d(512)
        self.do10 = nn.Dropout2d(0.5)
        self.conv15 = nn.Conv2d(512, 41, 1, stride=1, padding=0)
        self.conv15_bn = nn.BatchNorm2d(41)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.do1(x)

        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.do2(x)

        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = self.do3(x)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = self.do4(x)
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = self.do5(x)
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.do6(x)

        x = F.relu(self.conv9_bn(self.conv9(x)))
        x = F.relu(self.conv10_bn(self.conv10(x)))
        x = F.max_pool2d(x, (1, 2))
        x = self.do7(x)

        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
        x = F.max_pool2d(x, (1, 2))
        x = self.do8(x)

        x = F.relu(self.conv13_bn(self.conv13(x)))
        x = self.do9(x)
        x = F.relu(self.conv14_bn(self.conv14(x)))
        x = self.do10(x)

        x = self.conv15_bn(self.conv15(x))
        size_to_pool = x.size()
        x = F.avg_pool2d(x, (size_to_pool[2], size_to_pool[3]))
        x = x.view(-1, 41)

        return x

    def predict_proba(self, x):
        x = torch.from_numpy(x).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.forward(x), dim=-1)

        return probs.cpu().numpy().flatten()


def load_dcase_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(device=device)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage))
    model.to(device)
    model.eval()

    return model