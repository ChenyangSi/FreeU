<div align="center">

<h1>FreeU: Free Lunch in Diffusion U-Net</h1>

<div>
    <a href="https://chenyangsi.github.io/" target="_blank">Chenyang Si</a><sup></sup>,
    <a href="https://ziqihuangg.github.io/" target="_blank">Ziqi Huang</a><sup></sup>,
    <a href="https://yumingj.github.io/" target="_blank">Yuming Jiang</a><sup></sup>,
    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup></sup>
</div>
<div>
    <sup></sup>S-Lab, Nanyang Technological University
</div>

[Paper](https://arxiv.org/pdf/2309.11497.pdf) | [Project Page](https://chenyangsi.top/FreeU/) | [Video](https://www.youtube.com/watch?v=-CZ5uWxvX30&t=2s)
</br>


<strong>We propose FreeU, a method that substantially improves diffusion model sample quality at no costs: no training, no additional parameter introduced, and no increase in memory or sampling time.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="./readme_teaser.jpg">
</div>

:open_book: For more visual results, go checkout our <a href="https://chenyangsi.top/FreeU/" target="_blank">project page</a>
</div>


## FreeU
```python
def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    return x_filtered

class Free_UNetModel(UNetModel):
    """
    :param b1: backbone factor of the firt stage block of decoder.
    :param b2: backbone factor of the second stage block of decoder.
    :param s1: skip factor of the firt stage block of decoder.
    :param s2: skip factor of the second stage block of decoder.
    """

    def __init__(
        self,
        b1,
        b2,
        s1,
        s2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.b1 = b1 
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            hs_ = hs.pop()

            # --------------- FreeU operation code -----------------------
            # Only operate on the first two stages
            if h.shape[1] == 1280:
                h[:,:640] = h[:,:640] * self.b1
                hs_ = highenhance_filter(hs_, threshold=1, scale=self.s1)
            if h.shape[1] == 640:
                h[:,:320] = h[:,:320] * self.b2
                hs_ = highenhance_filter(hs_, threshold=1, scale=self.s2)
            # ---------------------------------------------------------

            h = th.cat([h, hs_], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
```

## Parameters

### SD1.4: 
**b1**: 1.2, **b2**: 1.4, **s1**: 0.9, **s2**: 0.2

### SD2.1 
**b1**: 1.1, **b2**: 1.2, **s1**: 0.9, **s2**: 0.2

### Range for More Parameters
When trying additional parameters, consider the following ranges:
- **b1**: 1 ≤ b1 ≤ 1.2
- **b2**: 1.2 ≤ b2 ≤ 1.6
- **s1**: s1 ≤ 1
- **s2**: s2 ≤ 1

 

If you find FreeU useful for your work please cite:
```
@article{Si2023FreeU,
  author    = {Chenyang Si, Ziqi Huang, Yuming Jiang, Ziwei Liu},
  title     = {FreeU: Free Lunch in Diffusion U-Net},
  journal   = {arXiv},
  year      = {2023},
}
```
## :newspaper_roll: License

Distributed under the S-Lab License. See `LICENSE` for more information.

