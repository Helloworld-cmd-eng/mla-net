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
    def __init__(self,dim, n_out,heads = 9, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        project_out = not (heads == 1 and dim_head == dim)
        self.n_out=n_out

        if heads%n_out!=0:
            print('heads must be diveded by n_out!!! ---wwt')
        self.chunk_dim=inner_dim//n_out
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)


        self.to_out = nn.ModuleList([nn.Sequential(
            nn.Linear(self.chunk_dim, dim//2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim//2, dim),
        ) if project_out else nn.Identity() for _ in range(n_out)])


    def forward(self, x):
        n_samples,n_tokens,dim=x.shape

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print(dots.shape)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).chunk(self.n_out,1)

        out = [rearrange(e, 'b h n d -> b n (h d)') for e in out]

        return [x+m(e) for e,m in zip(out,self.to_out)]

class Transformer(nn.Module):
    def __init__(self,dim,n_out, heads, dim_head,dropout = 0.5):
        super().__init__()
        self.ln=nn.LayerNorm(dim)
        self.attn = Attention(dim,n_out,heads,dim_head,dropout)

    def forward(self, x):

        x=self.ln(x)
        out =self.attn(x)
        return out



class NeckViTModule(nn.Module):
    def __init__(self, num_patches,n_out,dim, heads,dim_head=64, dropout=0.5, emb_dropout=0.5):
        super(NeckViTModule,self).__init__()


        self.num_patches=num_patches

        self.n_out=n_out
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+self.n_out, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.sp_transformer = Transformer(dim, n_out,heads, dim_head, dropout)

        self.fc=nn.Sequential(nn.Linear(dim*num_patches,dim))

        self.cls_token = nn.Parameter(torch.randn(1, self.n_out, dim))

    def forward(self, feat_x):
        '''
        feat_x:torch.tensor(b,c,k,h,w)
        '''
        n_imgbatch, embed_dim, n_patch, hight, width = feat_x.shape
        x = feat_x.permute(0, 3, 4, 2, 1).flatten(0, 2)  # torch.tensor(n_imgbatch*hight*width,n_patch,embed_dim)

        n_transbatch, n_patch, _ = x.shape  # torch.tensor(n_transbatch,n_patch,embed_dim)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=n_transbatch)

        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n_patch+self.n_out)]
        x = self.dropout(x)

        x_rich_list = self.sp_transformer(x)

        out=[]
        for i,e in enumerate(x_rich_list):
            if i==0:
                out.append(e[:,0])
            elif i==1:
                out.append(e[:,1])
            elif i==2:
                out.append(e[:,2]+self.fc(e[:,self.n_out:].flatten(1,2)))


        return [e.contiguous().view(n_imgbatch,hight,width,embed_dim).permute(0,3,1,2) for e in out]

def GetModelParameters(net):
    total = sum([param.nelement() for param in net.parameters()])
    print('Numberofparameter: % .2fM' % (total / 1e6))


if __name__=='__main__':
    v = NeckViTModule(num_patches=9, n_out=3,dim=64, heads=9, dim_head=32,dropout=0.5)

    img = torch.randn(1,64,9,5,5)

    preds = v(img)  # (1, 1000)


    GetModelParameters(v)

    print(preds[0].shape,preds[1].shape,preds[2].shape)