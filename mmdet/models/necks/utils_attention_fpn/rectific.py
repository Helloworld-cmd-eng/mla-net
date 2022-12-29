import torch.nn as nn
import torch
import torch.nn.functional as F




class Rectific(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=0, dialation=1):
        super(Rectific, self).__init__()
        self.stride = stride


        self.kernel_size = kernel_size
        self.padding = padding
        self.num_void = dialation - 1


        if padding > 0:
            self.PaddingLayers = nn.ZeroPad2d([padding, padding, padding, padding])
    def virindex(self,index,kernel_size,virkernel_size):
        return index//kernel_size*(self.num_void+1)*virkernel_size+index%kernel_size*(1+self.num_void)
    def forward(self, x):
        '''
        x:torch.tensor(b,c,h,w)
        '''
        num_batch, channel, h, w = x.shape

        if self.padding > 0:
            x = self.PaddingLayers(x)

        virkernel_size=self.kernel_size+(self.kernel_size-1)*self.num_void

        target_h = (h + 2 * self.padding - virkernel_size) // self.stride + 1
        target_w = (w + 2 * self.padding - virkernel_size) // self.stride + 1

        LLInputIndexRow = [
            torch.arange(0, 1 + h - virkernel_size + 2 * self.padding,
                         self.stride) + (self.virindex(index,self.kernel_size,virkernel_size) // virkernel_size) for index in
            range(self.kernel_size ** 2)]
        LLInputIndexCol = [
            torch.arange(0, 1 + w - virkernel_size + 2 * self.padding,
                         self.stride) + (self.virindex(index,self.kernel_size,virkernel_size) % virkernel_size) for index in
            range(self.kernel_size ** 2)]

        YAddress = []
        XAddress = []
        for row, col in zip(LLInputIndexRow, LLInputIndexCol):
            YAddress.append(row.unsqueeze(-1).repeat(1, target_w))
            XAddress.append(col.unsqueeze(0).repeat(target_h, 1))
        YAddress = torch.stack(YAddress, 0)
        XAddress = torch.stack(XAddress, 0)

        dst = x[:, :, YAddress, XAddress]


        return dst


class off_Rectific(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=0, dialation=1):
        super(off_Rectific, self).__init__()
        self.stride = stride


        self.kernel_size = kernel_size
        self.padding = padding
        self.num_void = dialation - 1

        self.mechine=nn.Unfold(kernel_size=kernel_size,stride=stride,dilation=dialation,padding=padding)

    def virindex(self,index,kernel_size,virkernel_size):
        return index//kernel_size*(self.num_void+1)*virkernel_size+index%kernel_size*(1+self.num_void)
    def forward(self, x):
        '''
        x:torch.tensor(b,c,h,w)
        '''
        num_batch, channel, h, w = x.shape


        virkernel_size=self.kernel_size+(self.kernel_size-1)*self.num_void

        target_h = (h + 2 * self.padding - virkernel_size) // self.stride + 1
        target_w = (w + 2 * self.padding - virkernel_size) // self.stride + 1

        return self.mechine(x).view(num_batch,channel,self.kernel_size**2,target_h,target_w)




def GetModelParameters(net):
    total = sum([param.nelement() for param in net.parameters()])
    print('Numberofparameter: % .2fM' % (total / 1e6))

if __name__ == '__main__':
    pass

    # c3_i=torch.zeros(2,512,80,80).cuda()+torch.arange(0,80*80).view(1,1,80,80).cuda()
    # c4_i=torch.zeros(2,1024,40,40).cuda()+torch.arange(0,40*40).view(1,1,40,40).cuda()
    # c5_i=torch.zeros(2,2048,20,20).cuda()+torch.arange(0,20*20).view(1,1,20,20).cuda()
    # out=neck([c3_i,c4_i,c5_i])

    rectific=Rectific(kernel_size=1,stride=1,dialation=1,padding=0)
    i=torch.zeros(2,256,40,10)+torch.arange(0,20*20).view(1,1,40,10)
    data=rectific(i)
    print(data[0,0,:,39,1])