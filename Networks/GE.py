import torch
import torch.nn as nn
from thop import profile
from .SCNet import ScNet
from .MCNet import McNet
import torch.nn.functional as F


class HSVtoRGB(torch.nn.Module):
    def forward(self, hsv: torch.Tensor) -> torch.Tensor:
        hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        _c = hsv_l * hsv_s
        _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
        _m = hsv_l - _c
        _o = torch.zeros_like(_c)
        idx = (hsv_h * 6.).type(torch.uint8)
        idx = (idx % 6).expand(-1, 3, -1, -1)
        rgb = torch.empty_like(hsv)
        rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
        rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
        rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
        rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
        rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
        rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
        rgb += _m
        return rgb

class GlobalEnhancement(nn.Module):
    def __init__(self):
        super(GlobalEnhancement, self).__init__()
        self.scNet = ScNet()
        self.mcNet = McNet()
        self.hsv2rgb = HSVtoRGB()

    def forward(self, x):
        input_h =  x[:, 0:1, :, :]
        input_s =  x[:, 1:2, :, :]
        input_v =  x[:, 2:3, :, :]

        output_SCNet = self.scNet(input_v)
        output_SCNet = (output_SCNet+input_v)/2
        
        output_hsv = torch.cat([input_h,input_s,output_SCNet], dim=1)
        input_rgb = self.hsv2rgb(output_hsv)
        output_rgb = self.mcNet(input_rgb)

        return output_rgb
