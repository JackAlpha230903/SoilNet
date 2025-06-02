import torch
import torch.nn as nn
import timm

class SoilNetDualHead(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.mnv2_block1 = nn.Sequential(*list(
            timm.create_model("mobilenetv2_100.ra_in1k", pretrained=True).blocks.children())[0:3]
        )
        self.channel_adapter = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        self.mobilevit_full = timm.create_model("mobilevitv2_050", pretrained=True)
        self.mobilevit_encoder = self.mobilevit_full.stages
        self.mvit_to_mnv2 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.mnv2_block2 = nn.Sequential(*list(
            timm.create_model("mobilenetv2_100.ra_in1k", pretrained=True).blocks.children())[3:7]
        )
        self.final_conv = nn.Conv2d(320, 1280, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.light_dense = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(
            nn.Linear(1280 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(1280 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_light):
        x = self.initial_conv(x_img)
        x = self.mnv2_block1(x)
        x = self.channel_adapter(x)
        x = self.mobilevit_encoder(x)
        x = self.mvit_to_mnv2(x)
        x = self.mnv2_block2(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x_img_feat = torch.flatten(x, 1)
        x_light_feat = self.light_dense(x_light)
        x_concat = torch.cat([x_img_feat, x_light_feat], dim=1)
        reg_out = self.reg_head(x_concat)
        cls_out = self.cls_head(x_concat)
        return reg_out, cls_out

model = SoilNetDualHead(num_classes=10)
torch.save(model.state_dict(), 'SoilNet_orginal.pth')
