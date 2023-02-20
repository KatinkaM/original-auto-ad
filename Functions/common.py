##Concat
import torch.nn as nn
import torch
import numpy as np


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module #Adds child to module
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input): #Forward function for the sequence, defines how the signal is sent forward, is needed because the class inherets from nn.Module
    #Defines the computation performed at every call
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)

##Activation default LeakyReLu

def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()

        ##Batch Normalization and Convolution net code

def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None 
    padder = None

    to_pad = int((kernel_size - 1) / 2) #If k = 1, then to_pad = 0.0, if k = 3, then to_pad = 1.0
    if pad == 'reflection': #Sets padding to zero
        padder = nn.ReflectionPad2d(to_pad) #Creates some kind of pad
        to_pad = 0
    
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias) 


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers) #returns a sequence of the layers


def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip) #check that these have the same value
    #[128,128,128,128,128]


    n_scales = len(num_channels_down) #5

    #Check if the variables are lists or tuples if not creates lists.
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 #5-1 = 4


    model = nn.Sequential() #Modules will be added to it in the order they are passed in the constructor. Chains output to input 
    model_tmp = model

    input_depth = num_input_channels #Amount of bands (filter depth)
    for i in range(len(num_channels_down)): #Iterate trough 5 times (amount of blocks)
        
        #Creates containter for skip network and the deep network
        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0: #For every layer a ConCat is added. 
            model_tmp.add(Concat(1, skip, deeper)) 
        else: 
            model_tmp.add(deeper)

        #Next layer of model_tmp add batch normalization (128 + 128 if i < 4 else 128) bn(256) always 
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        #Block 1 first iteration, block 4 after second (B is changed to 128 after first layer)
        if num_channels_skip[i] != 0: #For every layer conv(B,128,1) this is added to skip sequence + bn and leakyRelu
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
        
        #This is block 2 first iteration, block 5 second (B is changed to 128 after first layer)
        #Add convolutional net to deeper conv(input )= nr bands,output =  128, kernelsize = 3, stride = 2, bias = true, pad = zero, downsample = stride)
        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        #bn(128)
        deeper.add(bn(num_channels_down[i]))
        #Activation = leaky relu
        deeper.add(act(act_fun))


        #This is block 3
        #conv(128-128,3, stride  = 1, bias = true)
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        
        #Trying to add all the outputs so I can track them!! 
        navn = "B3" + str(i)
        deeper.register_forward_hook(get_features(navn))
        #The code must be here at some point! 

        deeper_main = nn.Sequential()

        #k is always set to 128 becausee numm_channels_up is a list of only elements = 128
        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        #Upsampling happens at decoder
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        #deeper.add(nn.functional.interpolate(scale_factor=2, mode=upsample_mode[i]))

        #Block 6
        #Conv(128+128,128,3,stride = 1) + bn + LeakyRelu
        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        #Block 4?
        #conv(128,128,1) +bn + LeakyRelu
        if need1x1_up: #Always true
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]#After the first iteration the Block1 = Block 4 and Block2 = Bloack 5
        model_tmp = deeper_main

    #This after the other part of the net is finished
    #Block 7
    #conv(128,B,1) + sigmoid
    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model