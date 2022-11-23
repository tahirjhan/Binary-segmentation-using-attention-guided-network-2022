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

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c, out_c)

    def forward(self, inputs):
        up = self.up(inputs)  # 256,16,16
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
class DDNet(nn.Module):
    def __init__(self):
        super(DDNet, self).__init__()

        # Encoder
        self.block1 = backbone() # torch.Size([1, 256, 64, 64])

        # Stream 1
        self.e1 = encoder_block(256,256) #
        self.e2 = encoder_block(256, 512) #
       # self.e3 = encoder_block(512, 1024) # torch.Size([1, 1024, 8, 8])

        # Stream 2
        self.e4 = deform_block(256, 256)
        self.e5 = deform_block(256, 512)    # torch.Size([1, 1024, 16, 16])


        # Decoder
        self.d1 = decoder_block(1024,512)  # 2048, 16, 16
        self.d2 = decoder_block(512, 256)  # 256, 32, 32
        self.d3 = decoder_block(256, 128)  # 128, 64, 64
        self.d4 = decoder_block(128, 64)   # 64, 256, 256


        # Classification
        self.outc = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.last_activation = nn.Sigmoid()

    def forward(self, input):
            # Encoder
        s1 = self.block1(input)  # torch.Size([1, 256, 64, 64])

        # stream 1
        e1 = self.e1(s1)
        e2 = self.e2(e1)

        # Stream 2
        e4 = self.e4(s1)
        e5 = self.e5(e4)

        encoder_out = torch.cat((e2,e5),1)


        # Decoder
        d1 = self.d1(encoder_out)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        #
        if self.last_activation is not None:
            output = self.last_activation(self.outc(d4))

        return output



model = DDNet()
input = torch.randn((1,3,256,256))
output = model(input)
#print (output.shape)

    #     # ******************** Encoding image ********************
    #     #originalmodel = torchvision.models.vgg16(pretrained=True, progress=True)
    #     #self.custom_model = nn.Sequential(*list(originalmodel.features.children())[:-21])
    #
    #     originalmodel = torchvision.models.densenet169(pretrained=False, progress=True)
    #     # pretrained_model = models.vgg16(pretrained=True).features
    #     self.custom_model = nn.Sequential(*list(originalmodel.features.children())[:-5])
    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
    #     )
    #
    #     self.deform1 = DeformConv2d(256, 128, 3, padding=1, modulation=True)
    #     self.deform2 = DeformConv2d(128, 128, 3, padding=1, modulation=True)
    #     # nn.ReLU(),
    #     self.deform3 = DeformConv2d(256, 64, 3, padding=1, modulation=True)
    #
    #     # ******************** Decoding image ********************
    #     self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=1)
    #     self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
    #     #self.deform = DeformConv2d(256, 128, 3, padding=1, modulation=True)
    #     self.upsampling1 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
    #
    #     self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(3, 3), stride=2, padding=1)
    #     self.upsampling2 = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
    #     self.outc = nn.Conv2d(128, 1, kernel_size=(1, 1))
    #     self.last_activation = nn.Sigmoid()
    #
    #
    #
    # def forward(self, x):
    #     # ******************** Initializing filters ********************
    #     # h_filter = torch.tensor([[-1., 0., 1.],
    #     #                 [-2., 1., 2.],
    #     #                 [-1., 0., 1.]]).to('cuda')
    #     # h_filter = h_filter.view(1, 1, 3, 3).repeat(256, 256, 1, 1) # convolution mask (gx)
    #     #
    #     # v_filter = torch.tensor([[-1., -2., -1.],
    #     #                          [0., 0., 0.],
    #     #                          [1., 2., 1.]]).to('cuda')
    #     # v_filter = v_filter.view(1, 1, 3, 3).repeat(1, 1, 1, 1) # convolution mask (gy)
    #     # ******************** Encoding image ********************
    #     x = self.custom_model(x)
    #     x = self.layer1(x)
    #
    #     deform1_x = self.deform1(x)
    #     deform2_x = self.deform2(deform1_x)
    #     x =  torch.cat((deform1_x,deform2_x),1)
    #     x = self.deform3(x)
    #     # x = self.deform3(deform2_x)
    #
    #
    #
    #     # # ******************** Decoding image ********************
    #     x = self.deconv1(x)
    #     x = self.deconv2(x)
    #     # x = self.deform(x)
    #     # x = F.conv2d(x, h_filter)
    #
    #     x = self.upsampling1(x)
    #     x = self.deconv3(x)
    #     # x = F.conv2d(x, v_filter)
    #     x = self.upsampling2(x)
    #     if self.last_activation is not None:
    #         logits = self.last_activation(self.outc(x))
    #         # print("111")
    #     else:
    #         logits = self.outc(x)
    #
    #     return logits
