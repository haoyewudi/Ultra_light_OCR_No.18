# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

__all__ = [
    'SlimMobileNet_v1', 'SlimMobileNet_v2', 'SlimMobileNet_v3',
    'SlimMobileNet_v4', 'SlimMobileNet_v5'
]

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SlimMobileNet(nn.Layer):
    def __init__(self, in_channels=3, scale=1.0, model_name='large', token=[]):
        super(SlimMobileNet, self).__init__()

        assert len(token) >= 45
        self.kernel_size_lis = token[:20]
        self.exp_lis = token[20:40]
        self.depth_lis = token[40:45]

        self.cfg = [
            # k, exp, c,  se,     nl,  s,
            [3, 16, 16, True, 'relu', 2],
        ]

        self.scale = scale
        self.inplanes = 16
        if model_name == "large":
            self.cfg_channel = [16, 24, 40, 80, 112, 160]
            self.cfg_stride = [1, 2, 2, 2, 1, 2]
            self.cfg_se = [False, False, True, False, True, True]
            self.cfg_act = [
                'relu', 'relu', 'relu', 'hardswish', 'hardswish',
                'hardswish'
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(self.inplanes * self.scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv1')

        self.stages = []
        block_list = []
        inplanes = make_divisible(self.inplanes * self.scale)
        num_mid_filter = make_divisible(self.scale * self.inplanes)
        _num_out_filter = self.cfg_channel[0]
        num_out_filter = make_divisible(self.scale * _num_out_filter)
        self.conv2 = ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=num_mid_filter,
                    out_channels=num_out_filter,
                    kernel_size=3,
                    stride=self.cfg_stride[0],
                    use_se=self.cfg_se[0],
                    act=self.cfg_act[0],
                    name="conv2")

        inplanes = make_divisible(self.scale * self.cfg_channel[0])
        i = 3
        for depth_id in range(len(self.depth_lis)):
            for repeat_time in range(self.depth_lis[depth_id]):
                num_mid_filter = make_divisible(self.scale * _num_out_filter * self.exp_lis[depth_id * 4 + repeat_time])
                _num_out_filter = self.cfg_channel[depth_id + 1]
                num_out_filter = make_divisible(self.scale * _num_out_filter)
                stride = self.cfg_stride[depth_id + 1] if repeat_time == 0 else 1
                if stride == 2 and depth_id > 0:
                    self.stages.append(nn.Sequential(*block_list))
                    block_list = []
                if stride==2 and depth_id>0:
                    stride = (2,1)
                conv = ResidualUnit(
                    in_channels = inplanes,
                    mid_channels=num_mid_filter,
                    out_channels=num_out_filter,
                    act=self.cfg_act[depth_id + 1],
                    stride=stride,
                    kernel_size=self.kernel_size_lis[depth_id * 4 + repeat_time],
                    use_se=self.cfg_se[depth_id + 1],
                    name='conv' + str(i))
                block_list.append(conv)
                inplanes = make_divisible(self.scale * self.cfg_channel[depth_id + 1])
                i += 1

        self.conv_last = ConvBNLayer(
            in_channels=num_out_filter,
            out_channels=make_divisible(self.scale * self.cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv_last')

        block_list.append(self.conv_last)
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels=make_divisible(scale * self.cls_ch_squeeze)
        for i, stage in enumerate(self.stages):
            self.add_sublayer(sublayer=stage, name="stage{}".format(i))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return x

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)

        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=None,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None,
                 name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + "_expand")
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            if_act=True,
            act=act,
            name=name + "_depthwise")
        if self.if_se:
            self.mid_se = SEModule(mid_channels, name=name + "_se")
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name=name + "_linear")

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(inputs, x)
        return x


class SEModule(nn.Layer):
    def __init__(self, in_channels, reduction=4, name=""):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name=name + "_1_weights"),
            bias_attr=ParamAttr(name=name + "_1_offset"))
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name + "_2_weights"),
            bias_attr=ParamAttr(name=name + "_2_offset"))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs


def SlimMobileNet_v1(in_channels=3,model_name='large', scale=1.0, token = None):
    token = [
        5, 3, 3, 7, 3, 3, 5, 7, 3, 3, 3, 3, 3, 3, 7, 3, 5, 3, 3, 3, 3, 3, 3, 6,
        3, 3, 3, 3, 4, 4, 4, 6, 4, 3, 4, 3, 6, 4, 3, 3, 2, 2, 2, 2, 4
    ]
    model = SlimMobileNet(in_channels=in_channels,model_name=model_name, scale=scale, token=token)
    return model


def SlimMobileNet_v2(in_channels=3,model_name='large', scale=1.0, token = None):
    token = [
        5, 3, 5, 7, 3, 3, 7, 3, 5, 3, 3, 7, 3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 4, 6,
        3, 3, 6, 3, 4, 4, 3, 4, 4, 4, 3, 6, 6, 4, 3, 3, 2, 2, 3, 2, 4
    ]
    model = SlimMobileNet(in_channels=in_channels,model_name=model_name, scale=scale,token=token)
    return model


def SlimMobileNet_v3(in_channels=3,model_name='large', scale=1.0, token = None):
    token = [
        3, 3, 3, 3, 5, 3, 7, 7, 7, 3, 3, 7, 5, 3, 5, 7, 5, 3, 3, 3, 3, 3, 3, 3,
        3, 4, 3, 4, 3, 6, 4, 4, 4, 4, 6, 3, 6, 4, 6, 3, 2, 2, 3, 2, 4
    ]
    model = SlimMobileNet(in_channels=in_channels,model_name=model_name, scale=scale,token=token)
    return model


def SlimMobileNet_v4(in_channels=3,model_name='large', scale=1.0, token = None):
    token = [
        3, 3, 3, 3, 5, 3, 3, 5, 7, 3, 5, 5, 5, 3, 3, 7, 3, 5, 3, 3, 3, 3, 4, 6,
        3, 4, 4, 6, 4, 6, 4, 6, 4, 6, 4, 4, 6, 6, 6, 4, 2, 3, 3, 3, 4
    ]
    model = SlimMobileNet(in_channels=in_channels,model_name=model_name, scale=scale, token=token)
    return model


def SlimMobileNet_v5(in_channels=3,model_name='large', scale=1.0, token = None):
    token = [
        7, 7, 3, 5, 7, 3, 5, 3, 7, 5, 3, 3, 5, 3, 7, 5, 7, 7, 5, 3, 3, 3, 6, 3,
        4, 6, 3, 6, 6, 3, 6, 4, 6, 6, 4, 3, 6, 6, 6, 6, 4, 4, 4, 4, 4
    ]
    model = SlimMobileNet(in_channels=in_channels,model_name=model_name, scale=scale, token=token)
    return model


if __name__ == "__main__":
    model = SlimMobileNet_v5(model_name='large', scale=0.5, token = None)
    x = paddle.tensor.randn((2,3,224,224))
    out = model.forward(x)
    """
    [2, 480, 7, 56]
    """
    print(out.shape)
    print(model.out_channels)