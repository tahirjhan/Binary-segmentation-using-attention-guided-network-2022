from torch import nn
import torchvision
import torch
import torch.nn.functional as F
from nets.utils import TrippleConv, multi_scale_aspp, MHSA
from torchvision.models import resnet, densenet201
from nets.deformable_conv import DeformConv2d
#------------------------------------------------------------------------------

class backbone(nn.Module):
    def __init__(self):
        super().__init__()

        baselayers = torchvision.models.densenet201(pretrained=True, progress=True)
        self.custom_model = nn.Sequential(*list(baselayers.features.children())[:-7])
        #print(self.custom_model)

    def forward(self, input):
        output = self.custom_model(input)
        return output
#------------------------------------------------
class dilated_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.convd1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.convd2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        x = self.convd1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.convd2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        return x
#---------------------------------------------------
#------------------------------------------------
class deform_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deform1 = DeformConv2d(in_c, out_c, 3, padding=1, modulation=True)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)


    def forward(self, inputs):
        x = self.deform1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
#---------------------------------------------------
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
#---------------------------------------------
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
       # self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
       # self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(in_c, out_c)

    def forward(self, inputs):
        up = self.up(inputs)
        x =self.conv(up)


        #x = torch.cat([x, skip], axis=1)
        #x = self.conv(x)
        return x
#------------------------------------------
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()
       # self.upsampling2 = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
###########################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
#------------------------------------------------------------------------
class DDNet(nn.Module):
    def __init__(self, inplanes=256, planes=512):
        super(DDNet, self).__init__()

        # Encoder
        self.block1 = backbone() # torch.Size([1, 256, 64, 64])

        # Stream 1 Classical convolution
        self.e1 = encoder_block(256,256) #
        self.e2 = encoder_block(256, 512) #
        self.e3 = encoder_block(512, 1024)  #


        # # Stream 2 Dilated convoltion
        self.e4 = dilated_block(256, 256)
        self.e5 = dilated_block(256, 512)    # torch.Size([1, 1024, 16, 16])
        self.e6 = dilated_block(512, 1024)

        self.ca = ChannelAttention(1024)
        self.sa = SpatialAttention()


        # Stream 3 Dilated covolution
        # self.e6 = dilated_block(256, 256)
        # self.e7 = dilated_block(256, 512)
        #
        # Attention mechanisms

        # # Decoder
        self.d1 = decoder_block(2048,512)  # 2048, 16, 16
        self.d2 = decoder_block(512, 256)  # 256, 32, 32
        self.d3 = decoder_block(256, 128)  # 128, 64, 64
        self.d4 = decoder_block(128, 64)   # 64, 256, 256
        self.d5 = decoder_block(64, 32)  # 64, 256, 256
        # #
        # #
        # # # Classification
        self.outc = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.last_activation = nn.Sigmoid()

    def forward(self, input):
            # Encoder
        s1 = self.block1(input)  # torch.Size([1, 256, 64, 64])

        # stream 1
        e1 = self.e1(s1)
        e2 = self.e2(e1) # torch.Size([1, 512, 16, 16])
        e3 = self.e3(e2)
        e3 = self.ca(e3) * e3
        #
        # # Stream 2 convolution
        e4 = self.e4(s1)
        e5 = self.e5(e4)   # torch.Size([1, 512, 16, 16])
        e6 = self.e6(e5)
        e6 = self.sa(e6) * e6
        #
        # # Stream 2 Dilated convolution
        # e6 = self.e6(s1)
        # e7 = self.e7(e6)
        #
        # # Feature level fusion
        encoder_out = torch.cat((e3,e6),1)
        # encoder_out2 = torch.cat((encoder_out,e7),1)
        #
        # Attention mechanism after feature level fusion
       # out = self.ca(encoder_out) * encoder_out
       # out = self.sa(out) * out
        #
        # # Decoder
        d1 = self.d1(encoder_out)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        #
        if self.last_activation is not None:
            output = self.last_activation(self.outc(d5))

        return output

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = DDNet()
input = torch.randn((1,3,256,256))
output = model(input)
#print(output.shape)

# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([numpy.prod(p.size()) for p in model_parameters])
# print(params)