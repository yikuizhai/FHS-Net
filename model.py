import torch
import torch.nn as nn
import torch.nn.functional as F
import MobileNetV2

class FGEM(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(FGEM, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d

        self.conv_scale2_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 3, self.in_d[1], self.out_d)
        # scale 3
        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 3, self.in_d[2], self.out_d)
        # scale 4
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 3, self.in_d[3], self.out_d)
        # scale 5
        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True
                    )
        )
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 2, self.in_d[4], self.out_d)

    def forward(self, c1,c2, c3, c4, c5):
        # scale 2
        c1_s2 = self.conv_scale2_c1(c1)
        c2_s2 = self.conv_scale2_c2(c2)
        c3_s2 = self.conv_scale2_c3(c3)
        c3_s2 = F.interpolate(c3_s2, scale_factor=(2, 2), mode='bilinear')    #(32,64,64)
        s2 = self.conv_aggregation_s2(torch.cat([c1_s2,c2_s2, c3_s2], dim=1), c2)   #(64,64,64)

        # scale 3
        c2_s3 = self.conv_scale3_c2(c2)
        c3_s3 = self.conv_scale3_c3(c3)
        c4_s3 = self.conv_scale3_c4(c4)
        c4_s3 = F.interpolate(c4_s3, scale_factor=(2, 2), mode='bilinear')  #(32,32,32)
        s3 = self.conv_aggregation_s3(torch.cat([c2_s3, c3_s3, c4_s3], dim=1), c3) #(64,32,32)

        # scale 4
        c3_s4 = self.conv_scale4_c3(c3)
        c4_s4 = self.conv_scale4_c4(c4)
        c5_s4 = self.conv_scale4_c5(c5)
        c5_s4 = F.interpolate(c5_s4, scale_factor=(2, 2), mode='bilinear') #(32,16,16)
        s4 = self.conv_aggregation_s4(torch.cat([c3_s4, c4_s4, c5_s4], dim=1), c4)  #(64,16,16)

        # scale 5
        c4_s5 = self.conv_scale5_c4(c4)
        c5_s5 = self.conv_scale5_c5(c5)
        s5 = self.conv_aggregation_s5(torch.cat([c4_s5, c5_s5], dim=1), c5)  #(64,8,8)

        return s2, s3, s4, s5


class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        y = self.gap(c_fuse)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        y = c_fuse * y.expand_as(c_fuse)
        c_out = self.relu(y + self.conv_identity(c))

        return c_out

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d

        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

        self.conv_context = nn.Sequential(
            nn.Conv2d(1, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.mid_d)
        )
        self.silu = nn.SiLU()
    def forward(self, x):
        b, c, _, _ = x.size()

        mask = self.cls(x)
        context = self.conv_context(mask)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = x2 + x3
        x_out = y + context

        return x_out, mask


class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d

        self.HPB5 = SupervisedAttentionModule(self.mid_d)

        self.conv_p4 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.sam_p4 = SupervisedAttentionModule(self.mid_d)
        self.HPB4 = nn.Sequential(self.conv_p4,self.sam_p4)

        self.sam_p3 = SupervisedAttentionModule(self.mid_d)
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.HPB3 = nn.Sequential(self.conv_p3,self.sam_p3)

        self.conv_p2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.HPB2 = nn.Sequential(self.conv_p2,self.cls)

    def forward(self, d2, d3, d4, d5):
        # HPB5
        M5, mask_p5 = self.HPB5(d5)
        # HPB4
        M4, mask_p4 = self.HPB4(d4 + F.interpolate(M5, scale_factor=(2, 2), mode='bilinear'))
        # HPB3
        M3, mask_p3 = self.HPB3(d3 + F.interpolate(M4, scale_factor=(2, 2), mode='bilinear'))
        # HPB2
        mask_p2 = self.HPB2(d2 + F.interpolate(M3, scale_factor=(2, 2), mode='bilinear'))

        return mask_p2, mask_p3, mask_p4, mask_p5


class FHSNet(nn.Module):
    def __init__(self):
        super(FHSNet, self).__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.FM = FGEM(channles, self.mid_d)
        self.decoder = Decoder(self.en_d * 2)

    def forward(self, x1,x2):

        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)

        x1_2, x1_3, x1_4, x1_5 = self.FM(x1_1, x1_2, x1_3, x1_4, x1_5)
        x2_2, x2_3, x2_4, x2_5 = self.FM(x2_1, x2_2, x2_3, x2_4, x2_5)

        c2, c3, c4, c5 = torch.abs(x1_2 - x2_2),torch.abs(x1_3 - x2_3),torch.abs(x1_4 - x2_4),torch.abs(x1_5 - x2_5)

        mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5)

        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = F.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear')
        mask_p5 = torch.sigmoid(mask_p5)

        return mask_p2, mask_p3, mask_p4, mask_p5


if __name__ == '__main__':
    x1 = torch.randn((1,3,256,256))
    x2 = torch.randn((1,3,256,256))
    net = FHSNet()
    net(x1,x2)
