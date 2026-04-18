import os
import sys
import math
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    """Make a standard normalization layer."""
    return nn.GroupNorm(32, channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return torch.utils.checkpoint.checkpoint(func, *args)
    else:
        return func(*inputs)

class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""
    def forward(self, x, emb, conditioning=None):
        """Apply the module to `x` given `emb` timestep embeddings and optional conditioning."""
        raise NotImplementedError()

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings and conditioning to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, conditioning=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, conditioning)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class CrossAttentionConditioning(nn.Module):
    """Cross-attention module for conditioning spatial features with multimodal information."""
    def __init__(self, feature_channels, conditioning_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_channels = feature_channels
        self.conditioning_dim = conditioning_dim
        self.num_heads = num_heads
        
        # Project conditioning to match feature dimensions for cross-attention
        self.conditioning_proj = nn.Linear(conditioning_dim, feature_channels)
        
        # Cross-attention from spatial features to conditioning
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection and normalization
        self.output_proj = nn.Sequential(
            nn.Linear(feature_channels, feature_channels),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(feature_channels)
        
    def forward(self, spatial_features, conditioning):
        """
        Args:
            spatial_features: [B, C, H, W] - spatial feature map
            conditioning: [B, conditioning_dim] - multimodal conditioning vector
        Returns:
            Enhanced spatial features with conditioning information
        """
        B, C, H, W = spatial_features.shape
        
        # Reshape spatial features for attention: [B, H*W, C]
        spatial_flat = spatial_features.view(B, C, H * W).transpose(1, 2)
        
        # Project conditioning and expand: [B, 1, C]
        conditioning_proj = self.conditioning_proj(conditioning).unsqueeze(1)
        
        # Apply cross-attention: spatial features attend to conditioning
        attended_features, attention_weights = self.cross_attention(
            query=spatial_flat,  # [B, H*W, C]
            key=conditioning_proj,  # [B, 1, C]
            value=conditioning_proj  # [B, 1, C]
        )
        
        # Residual connection and normalization
        enhanced_features = self.norm(spatial_flat + self.output_proj(attended_features))
        
        # Reshape back to spatial format: [B, C, H, W]
        enhanced_features = enhanced_features.transpose(1, 2).view(B, C, H, W)
        
        return enhanced_features

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    Enhanced with cross-attention conditioning support.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        use_conditioning_cross_attn=False,
        conditioning_dim=None,
        conditioning_num_heads=4,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_conditioning_cross_attn = use_conditioning_cross_attn

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        # Cross-attention conditioning (optional)
        if use_conditioning_cross_attn and conditioning_dim is not None:
            self.conditioning_cross_attn = CrossAttentionConditioning(
                feature_channels=self.out_channels,
                conditioning_dim=conditioning_dim,
                num_heads=conditioning_num_heads
            )
        else:
            self.conditioning_cross_attn = None
        
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, conditioning=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding and optional multimodal conditioning.
        """
        return checkpoint(
            self._forward, 
            (x, emb, conditioning), 
            self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x, emb, conditioning=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
            
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        # Apply cross-attention conditioning if available
        if self.conditioning_cross_attn is not None and conditioning is not None:
            h = self.conditioning_cross_attn(h, conditioning)
            
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, emb=None, conditioning=None):
        return checkpoint(
            self._forward, 
            (x,), 
            self.parameters(),
            self.use_checkpoint and self.training,
        )

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)

class MultiModalConditionedUNet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    Enhanced with proper multimodal conditioning support.
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        multimodal_embed_dim=512,
        use_conditioning_cross_attn=True,
        conditioning_cross_attn_layers=None,  # Which layers to apply cross-attention (e.g., [2, 3, 4])
        conditioning_num_heads=4,
    ):
        super().__init__()
        
        self._feature_size = 0
        self.use_conditioning_cross_attn = use_conditioning_cross_attn
        self.conditioning_cross_attn_layers = conditioning_cross_attn_layers or []

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
            
        self.model_channels = model_channels
        self.time_embed_dim = self.model_channels * 4
        
        # Time embedding
        self.time_embed = nn.Sequential(
            linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )
        
        # Multimodal conditioning projection
        self.multimodal_proj = nn.Sequential(
            linear(multimodal_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
            nn.LayerNorm(self.time_embed_dim)
        )
        
        # Input blocks
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        block_idx = 0
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                block_idx += 1
                use_cross_attn = (use_conditioning_cross_attn and 
                                 (not conditioning_cross_attn_layers or block_idx in conditioning_cross_attn_layers))
                
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_conditioning_cross_attn=use_cross_attn,
                        conditioning_dim=multimodal_embed_dim,
                        conditioning_num_heads=conditioning_num_heads,
                    )
                ]
                ch = int(mult * model_channels)
                
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                block_idx += 1
                use_cross_attn = (use_conditioning_cross_attn and 
                                 (not conditioning_cross_attn_layers or block_idx in conditioning_cross_attn_layers))
                
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            use_conditioning_cross_attn=use_cross_attn,
                            conditioning_dim=multimodal_embed_dim,
                            conditioning_num_heads=conditioning_num_heads,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                
        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_conditioning_cross_attn=use_conditioning_cross_attn,
                conditioning_dim=multimodal_embed_dim,
                conditioning_num_heads=conditioning_num_heads,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_conditioning_cross_attn=use_conditioning_cross_attn,
                conditioning_dim=multimodal_embed_dim,
                conditioning_num_heads=conditioning_num_heads,
            ),
        )
        
        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                block_idx += 1
                use_cross_attn = (use_conditioning_cross_attn and 
                                 (not conditioning_cross_attn_layers or block_idx in conditioning_cross_attn_layers))
                
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_conditioning_cross_attn=use_cross_attn,
                        conditioning_dim=multimodal_embed_dim,
                        conditioning_num_heads=conditioning_num_heads,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    block_idx += 1
                    use_cross_attn = (use_conditioning_cross_attn and 
                                     (not conditioning_cross_attn_layers or block_idx in conditioning_cross_attn_layers))
                    
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            use_conditioning_cross_attn=use_cross_attn,
                            conditioning_dim=multimodal_embed_dim,
                            conditioning_num_heads=conditioning_num_heads,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, extra):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param extra: a dict containing extra inputs including multimodal conditioning
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        
        # Get flow timestep embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Process multimodal conditioning
        conditioning = None
        if "multimodal_conditioning" in extra:
            multimodal_emb = extra["multimodal_conditioning"]
            multimodal_emb_proj = self.multimodal_proj(multimodal_emb)
            emb = emb + multimodal_emb_proj
            conditioning = multimodal_emb  # Pass original conditioning for cross-attention
        
        h = x
        for module in self.input_blocks:
            h = module(h, emb, conditioning)
            hs.append(h)
        
        h = self.middle_block(h, emb, conditioning)
        
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, conditioning)
        
        h = h.type(x.dtype)
        return self.out(h)