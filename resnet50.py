import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.init as init

from torchvision import models


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class unetp_resnet50(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(unetp_resnet50, self).__init__()
        if pretrained:
            self.model = smp.UnetPlusPlus(
                encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=32,  # model output channels (number of classes in your dataset)
            )
        else:
            self.model = smp.UnetPlusPlus(
                encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=32,  # model output channels (number of classes in your dataset)
            )

    def forward(self, X):

        """ Forward propagation in backbone encoder network.  """
        h = self.model(X)
        return h


if __name__ == '__main__':
    model = unetp_resnet50(pretrained=True).cuda()
    # summary(model, input_size=(3, 768, 768), batch_size=-1)
    output = model(torch.randn(2, 3, 768, 768).cuda())
    print(output.shape)
