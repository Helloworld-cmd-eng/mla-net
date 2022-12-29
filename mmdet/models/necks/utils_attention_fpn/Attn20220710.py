import torch
from torch import nn
from einops import rearrange, repeat
'''
token or fc:fc
fc num:2
share qkv layer

'''


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        n_samples,n_tokens,dim=x.shape

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print(dots.shape)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self,dim, depth, heads, dim_head, mlp_dim,dropout = 0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale_eme1=nn.Parameter(torch.rand(1,1,dim))
        self.scale_eme2 = nn.Parameter(torch.rand(1, 1, dim))
        self.scale_eme3 = nn.Parameter(torch.rand(1, 1, dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim,dropout = dropout))
            ]))
    def forward(self, x):
        x1=self.scale_eme1+x[:,:25]
        x2=self.scale_eme2+x[:,25:50]
        x3=self.scale_eme3+x[:,50:]
        x=torch.cat([x1,x2,x3],1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class LocalAttention(nn.Module):
    def __init__(self, num_patches,dim, depthL, headsL, mlp_dim=128, pool='cls',dim_head=64, dropout=0.5, emb_dropout=0.5):
        super(LocalAttention, self).__init__()


        self.num_patches=num_patches
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.n_out=len(headsL)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_list = nn.ModuleList([Transformer(dim, depthL[i], headsL[i], dim_head, mlp_dim, dropout) for i in range(self.n_out)])
        self.cls_token = [nn.Parameter(torch.randn(1, 1, dim)) for _ in range(self.n_out)]

        self.fc=nn.Linear(dim*num_patches,dim)

        self.pool = pool



    def forward(self, feat_x):
        '''
        feat_x:torch.tensor(b,c,k,h,w)
        '''
        n_imgbatch, embed_dim, n_patch, hight, width = feat_x.shape
        x = feat_x.permute(0, 3, 4, 2, 1).flatten(0, 2)  # torch.tensor(n_imgbatch*hight*width,n_patch,embed_dim)

        n_transbatch, n_patch, _ = x.shape  # torch.tensor(n_transbatch,n_patch,embed_dim)


        x += self.pos_embedding[:, :(n_patch)]
        x = self.dropout(x)



        outL=[]
        for i in range(self.n_out):
            tk=self.cls_token[i].to(x)
            tk = repeat(tk, '1 n d -> b n d', b=n_transbatch)

            out= self.transformer_list[i](torch.cat([tk,x],1))
            if i!=self.n_out-1:
                outL.append(out[:,0])
            else:
                outL.append(out[:,0]+self.fc(out[:,1:].flatten(1,2)))

        return [e.contiguous().view(n_imgbatch,hight,width,embed_dim).permute(0,3,1,2) for e in outL]



def GetModelParameters(net):
    total = sum([param.nelement() for param in net.parameters()])
    print('Numberofparameter: % .2fM' % (total / 1e6))


if __name__=='__main__':
    v = LocalAttention(num_patches=16, dim=256, headsL=[1, 1, 9], depthL=[1, 1, 1], mlp_dim=64, pool='cls', dim_head=64, dropout=0.5)

    img = torch.randn(1,256,16,20,20)

    preds = v(img)  # (1, 1000)


    GetModelParameters(v)

    print(preds[0].shape,preds[1].shape)