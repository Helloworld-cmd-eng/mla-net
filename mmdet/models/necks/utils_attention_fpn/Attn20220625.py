import torch
from torch import nn
from einops import rearrange, repeat
'''
token or fc:fc
fc num:2
unshare qkv layer

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

class UnshareAttention(nn.Module):
    def __init__(self,n_patch_pergroup,dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.n_patch_pergroup=n_patch_pergroup
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.ModuleList(nn.Linear(dim, inner_dim * 3, bias = False) for _ in range(len(n_patch_pergroup)))

        self.to_out = nn.ModuleList(nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() for _ in range(len(n_patch_pergroup)))

    def splitgroup(self,x):
        point=0
        grouplist=[]
        for n in self.n_patch_pergroup:
            grouplist.append(x[:,point:point+n])
            point+=n
        return grouplist

    def forward(self, x):
        n_samples,n_tokens,dim=x.shape

        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        total_q=[]
        total_k=[]
        total_v=[]
        for m,e in zip(self.to_qkv,self.splitgroup(x)):
            sub_qkv=m(e).chunk(3,dim=-1)
            sub_q,sub_k,sub_v=map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), sub_qkv)
            total_q.append(sub_q)
            total_k.append(sub_k)
            total_v.append(sub_v)
        total_q=torch.cat(total_q,2)
        total_k=torch.cat(total_k,2)
        total_v=torch.cat(total_v,2)

        dots = torch.matmul(total_q, total_k.transpose(-1, -2)) * self.scale
        # print(dots.shape)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, total_v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        results=[]
        for m,e in zip(self.to_out,self.splitgroup(out)):
            results.append(m(e))
        return torch.cat(results,1)


class Transformer(nn.Module):
    def __init__(self,n_patch_pergroup,dim, depth, heads, dim_head, mlp_dim,dropout = 0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, UnshareAttention(n_patch_pergroup,dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim,dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class NeckViTModule(nn.Module):
    def __init__(self, n_patch_pergroup,dim, depth, heads, mlp_dim, pool='cls',dim_head=64, dropout=0.5, emb_dropout=0.5):
        super(NeckViTModule,self).__init__()


        self.num_patches=torch.sum(torch.as_tensor(n_patch_pergroup)).item()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(n_patch_pergroup,dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc1=nn.Linear(dim*self.num_patches,dim)
        self.fc2=nn.Linear(dim*self.num_patches,dim)
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

        x = self.transformer(x)


        x=torch.flatten(x,1,2)
        out=[]
        out.append(self.fc1(x))
        out.append(self.fc2(x))

        return [e.contiguous().view(n_imgbatch,hight,width,embed_dim).permute(0,3,1,2) for e in out]

class ViTModule(nn.Module):
    def __init__(self, num_patches, dim, depth, heads, mlp_dim=64, pool = 'cls', dim_head = 32, dropout = 0.25, emb_dropout = 0.25):
        super(ViTModule,self).__init__()


        self.num_patches = num_patches
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'



        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool



    def forward(self, feat_x):
        '''
        x:torch.tensor(n_batch,embed_dim,n_patch,hight,width)
        '''
        n_imgbatch,embed_dim,n_patch,hight,width=feat_x.shape
        x=feat_x.permute(0,3,4,2,1).flatten(0,2)#torch.tensor(n_imgbatch*hight*width,n_patch,embed_dim)

        n_transbatch, n_patch, _ = x.shape#torch.tensor(n_transbatch,n_patch,embed_dim)

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = n_transbatch)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n_patch + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)#torch.tensor(n_batch,1,embed_dim)
        return x.contiguous().view(n_imgbatch,hight,width,embed_dim).permute(0,3,1,2)


def GetModelParameters(net):
    total = sum([param.nelement() for param in net.parameters()])
    print('Numberofparameter: % .2fM' % (total / 1e6))


if __name__=='__main__':
    v = NeckViTModule([25,25,9], dim=64, heads=4,depth=2,mlp_dim=64, pool='cls', dim_head=32,dropout=0.5)

    img = torch.randn(1,64,59,5,5)

    preds = v(img)  # (1, 1000)


    GetModelParameters(v)

    print(preds[0].shape,preds[1].shape)