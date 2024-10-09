import torch.nn as nn
import torch.nn.functional as F
import torch 

"""steps:
1. weight intialization: we intialize the convultional kernel (weights and biases)
2. sliding window convultion: for each input channel (forexample a color channle or rgb images)
we slide the kernel over the input and compute the dot product between the kernel and input regions
3. Bias addition: add the bias term to the result of each convultion operation
4. Stride and padding: support for stride (step size when sliding the kernel) and padding 
(padding the input to control the output size)"""

class customConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, padding = 0, kernel_size = 2):
        super(customConv2D, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.stride = stride 
        self.padding = padding 
        self.kernel_size = kernel_size

        #initializing the weights of the convultional kernel 
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        #initializing the bias
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

            # get the dimensions of the input
            batch_size, in_channel, input_height, input_width = x.shape 

            # calculate the estimated output dimensions.
            out_height = (input_height - self.kernel_size) // self.stride + 1
            out_width = (input_width - self.kernel_size) // self.stride + 1

            # initiailze the output_tensor 
            output = torch.zeros(batch_size, self.out_channels, out_height, out_width)

            # perform the convultion
            for i in range(out_height):
                for j in range(out_width):
                    # extract the sliding window(patch of the input)
                    input_patch = x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]

                    # perform the convultions by multiplying input_patch with the weights and summing
                    # this will give us the single number for each input channel

                    for o in range(self.out_channels):
                        output[:, o, i, j] = torch.sum(input_patch*self.weights[o, :, :, :], dim=(1,2,3))+self.bias[0]
            return output

