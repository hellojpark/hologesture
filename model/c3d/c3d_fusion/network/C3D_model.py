import torch
import torch.nn as nn
from mypath import Path

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, configer, num_classes, pretrained=False):
        super(C3D, self).__init__()
        self.configer = configer
        ########################################################################################################################
        # 2nd variation point => check Hololens_data.py depth_data concatenate => check forward depth = self.relu(self.conv1(depth))
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,3,3), padding=(1,1,1))
        #################################or or or#################################
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        ########################################################################################################################

        ########################################################################################################################
        # 4th variation point => check train.py and Hololens_data.py
        # self.convrgb1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        ########################################################################################################################
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))



        ########################################################################################################################
        # 4th variation point => check train.py and Hololens_data.py
        self.fc6 = nn.Linear(8192, 4096)
        #################################or or or#################################
        ########################################################################################################################
        # 3rd variation point => read_data.py resize
        # self.fc6 = nn.Linear(8192, 4096)
        # self.rgbfc6 = nn.Linear(8192,4096)
        #################################or or or#################################
        ## depth = depth
        # self.fc6 = nn.Linear(56320, 4096)
        # ## rgb = depth_based_rgb
        # self.rgbfc6 = nn.Linear(28160, 4096)
        ########################################################################################################################
        ########################################################################################################################
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, rgb, depth):
        # print(x.shape)
        # a = x.detach().cpu().numpy()
        # a = a[0,0,:,:]

        # import matplotlib.pyplot as plt
        # plt.imshow(a)
        # plt.show()
        ########################################################################################################################
        # 4th variation point => check train.py and Hololens_data.py
        rgb = self.relu(self.conv1(rgb))
        #################################or or or#################################
        # rgb = self.relu(self.convrgb1(rgb))
        ########################################################################################################################
        rgb = self.pool1(rgb)

        rgb = self.relu(self.conv2(rgb))
        rgb = self.pool2(rgb)

        rgb = self.relu(self.conv3a(rgb))
        rgb = self.relu(self.conv3b(rgb))
        rgb = self.pool3(rgb)

        rgb = self.relu(self.conv4a(rgb))
        rgb = self.relu(self.conv4b(rgb))
        rgb = self.pool4(rgb)

        rgb = self.relu(self.conv5a(rgb))
        rgb = self.relu(self.conv5b(rgb))
        rgb = self.pool5(rgb)
        
        ########################################################################################################################
        # 3rd variation point => check read_data.py resize
        rgb = rgb.view(-1, 8192)
        #################################or or or#################################
        # rgb = rgb.view(-1, 28160)
        ########################################################################################################################

        ########################################################################################################################
        # 4th variation point => check train.py and Hololens_data.py
        rgb = self.relu(self.fc6(rgb))
        #################################or or or#################################
        # rgb = self.relu(self.rgbfc6(rgb))
        ########################################################################################################################
        rgb = self.dropout(rgb)
        rgb = self.relu(self.fc7(rgb))
        rgb = self.dropout(rgb)
        rgb_logits = self.fc8(rgb)

        ########################################################################################################################
        # 4th variation point => check train.py and Hololens_data.py
        # rgb_logits = self.fc8(rgb)
        ########################################################################################################################

        ########################################################################################################################
        # 2nd variation point => check Hololens_data.py depth_data concatenate
        depth = self.relu(self.conv1(depth))
        #################################or or or#################################
        # depth = depth.unsqueeze(1)      # 처음에 x는 (batch,20,512,512) 이다. 여기서 channel에 대한 차원이 없어서 batch와 frame수를 나타내는 차원 사이에 1차원을 추가한다.->(batch, 1, 20, 512, 512)
        # depth = self.relu(self.conv1(depth))
        ########################################################################################################################
        depth = self.pool1(depth)

        depth = self.relu(self.conv2(depth))
        depth = self.pool2(depth)

        depth = self.relu(self.conv3a(depth))
        depth = self.relu(self.conv3b(depth))
        depth = self.pool3(depth)

        depth = self.relu(self.conv4a(depth))
        depth = self.relu(self.conv4b(depth))
        depth = self.pool4(depth)

        depth = self.relu(self.conv5a(depth))
        depth = self.relu(self.conv5b(depth))
        depth = self.pool5(depth)

        ########################################################################################################################
        # 3rd variation point => check read_data.py resize
        depth = depth.view(-1, 8192)
        #################################or or or#################################
        # depth = depth.view(-1, 56320)
        ########################################################################################################################
        depth = self.relu(self.fc6(depth))
        depth = self.dropout(depth)
        depth = self.relu(self.fc7(depth))
        depth = self.dropout(depth)
        depth_logits = self.fc8(depth)

        ########################################################################################################################
        # 4th variation point => check train.py and Hololens_data.py
        if self.configer.get("fusionkind") == 'late_fusion':
            logits = rgb_logits + depth_logits
        elif self.configer.get("fusionkind") == 'feature_fusion':
            logits = self.fc8(rgb+depth)
        return logits
        #################################or or or#################################
        # depth_logits = self.fc8(depth)

        # return depth_logits+rgb_logits
        ########################################################################################################################

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=101, pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())