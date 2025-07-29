import warnings
import torch
import torch.nn.functional as F
from torch import nn

from depth_anything_v2.dpt import DepthAnythingV2
from models.resnet import ResNet34
from models.context_modules import get_context_module
from models.model_utils import ConvBNAct, Swish, Hswish
from models.decoder import Decoder
from models.befusion import BeFusion
from DSCNet.ds_encoder import DSCNet_Encoder


class TopoNet(nn.Module):
    def __init__(self,
                 height=256,
                 width=480,
                 num_classes=4,
                 depth_path=None,
                 encoder='resnet34',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='/results_nas/moko3016/'
                                'moko3016-efficient-rgbd-segmentation/'
                                'imagenet_pretraining',
                 activation='relu',
                 input_channels=3,
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 weighting_in_encoder='',
                 upsampling='bilinear'):
        super(TopoNet, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.weighting_in_encoder = weighting_in_encoder
        self.depth_path = depth_path

        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError('Only relu, swish and hswish as '
                                      'activation function are supported so '
                                      'far. Got {}'.format(activation))
        
        self.dsc_encoder = DSCNet_Encoder()
        self.first_conv = ConvBNAct(1, 3, kernel_size=1,
                                    activation=self.activation)
        self.mid_conv = ConvBNAct(512 + 512, 512, kernel_size=1,
                                  activation=self.activation)

        depth_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        depth_encoder = 'vitb'
        self.depth_encoder = DepthAnythingV2(**depth_model_configs[depth_encoder])
        self.depth_encoder.load_state_dict(torch.load(self.depth_path))
        self.depth_encoder.eval()


        self.encoder = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=input_channels
        )

        self.channels_decoder_in = self.encoder.down_32_channels_out

        self.be0 = BeFusion(64, 512, 512, isFirst=True)
        self.be1 = BeFusion(self.encoder.down_4_channels_out, 256, 256)
        self.be2 = BeFusion(self.encoder.down_8_channels_out, 128, 128)
        self.be3 = BeFusion(self.encoder.down_16_channels_out, 64, 64)
        self.be4 = BeFusion(self.encoder.down_32_channels_out, 32, 32, isLast=True)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = get_context_module(
            context_module,
            self.channels_decoder_in,
            channels_decoder[0],
            input_size=(height // 32, width // 32),
            activation=self.activation,
            upsampling_mode=upsampling_context_module)

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

    def forward(self, image):
        original_depth = F.interpolate(image, size=(1022, 1022), mode='area')
        original_depth = self.depth_encoder.infer_image(original_depth)

        original_depth = original_depth.expand(-1, 3, -1, -1)

        d0, d1, d2, d3, original_depth = self.dsc_encoder(original_depth)
        out = self.encoder.forward_first_conv(image)
        skipf0, out_f0 = self.be0(out, F.interpolate(d0, size=(512, 512), mode='area'))
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        # block 1
        out = self.encoder.forward_layer1(out)
        skipf1, out_f1 = self.be1(out, F.interpolate(d1, size=(256, 256), mode='area'), out_f0)
        skip1 = self.skip_layer1(skipf1)

        # block 2
        out = self.encoder.forward_layer2(out)
        skipf2, out_f2 = self.be2(out, F.interpolate(d2, size=(128, 128), mode='area'), out_f1)
        skip2 = self.skip_layer2(skipf2)

        # block 3
        out = self.encoder.forward_layer3(out)
        skipf3, out_f3 = self.be3(out, F.interpolate(d3, size=(64, 64), mode='area'), out_f2)
        skip3 = self.skip_layer3(skipf3)

        # block 4
        out = self.encoder.forward_layer4(out)
        skipf4, out_f4 = self.be4(out, F.interpolate(original_depth, size=(32, 32), mode='area'), out_f3)

        out = self.context_module(out_f4)

        outs = [out, skip3, skip2, skip1]
        outs, out_visual = self.decoder(enc_outs=outs)
        outs = F.log_softmax(outs, dim=1)

        return outs, original_depth
