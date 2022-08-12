import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.tools import initialize_weights


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)





class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaConv2d, self).__init__()
        ignore = nn.Conv2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        
        super(MetaConvTranspose2d,self).__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size=kwargs["kernel_size"]
        self.output_padding = ignore.output_padding

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x, output_size=None):
        output_padding=self.output_padding#self._output_padding(x, output_size)
        
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
       
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class UnetEncoder(MetaModule):

    def __init__(self, in_channels, out_channels):
        super(UnetEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Sequential(MetaConv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
                              #nn.BatchNorm2d(self.out_channels),
                              nn.ReLU(inplace=True),
                              MetaConv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                              #nn.BatchNorm2d(self.out_channels),
                              nn.ReLU(inplace=True))

    def forward(self,x):
        return self.layer(x)


class UnetDecoder(MetaModule):

    def __init__(self, in_channels, featrures, out_channels):
        super(UnetDecoder, self).__init__()
        self.in_channels = in_channels
        self.features = featrures
        self.out_channels = out_channels

        self.layer = nn.Sequential(MetaConv2d(self.in_channels, self.features, kernel_size=3, padding=1),
                                   #nn.BatchNorm2d(self.features),
                                   nn.ReLU(inplace=True),
                                   MetaConv2d(self.features, self.features, kernel_size=3, padding=1),
                                   #nn.BatchNorm2d(self.features),
                                   nn.ReLU(inplace=True),
                                   MetaConvTranspose2d(self.features, self.out_channels, kernel_size=2, stride=2),
                                   #nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self,x):
        return self.layer(x)



class UNet_rw(MetaModule):
    def __init__(self, num_classes):
        super(UNet_rw, self).__init__()
        
        self.num_classes = num_classes

        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.center = nn.Sequential(MetaConv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    MetaConv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    MetaConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(MetaConv2d(128, 64, 3, padding=1),
                                 MetaConv2d(64, 64, 3, padding=1))
                                 
        self.output = MetaConv2d(64, self.num_classes, kernel_size=1, stride=1)
        #self.final = nn.Sigmoid()

        #Initialize weights
        initialize_weights(self)
        
    def forward(self, x):
        
        en1 = self.down1(x)
        po1 = self.pool1(en1)
        
        en2 = self.down2(po1)
        po2 = self.pool2(en2)
        en3 = self.down3(po2)
        po3 = self.pool3(en3)
        en4 = self.down4(po3)
        po4 = self.pool4(en4)
        c1 = self.center(po4)

        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:], mode='bilinear')], 1))
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:], mode='bilinear')], 1))
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:], mode='bilinear')], 1))
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:], mode='bilinear')], 1))

        out = self.output(dec4)

        return out#self.final(out)
