# InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering

Pytorch implementation of our method for regularizing nerual radiance fields for few-shot neural volume rendering.

### [Project](https://cv.snu.ac.kr/research/InfoNeRF/) | [Paper](https://arxiv.org/abs/2112.15399) 

[Mijeong Kim](https://mjmjeong.github.io/), [Seonguk Seo](https://seoseong.uk/), [Bohyung Han](https://cv.snu.ac.kr/~bhhan/)

Seoul National University

arXiv 2112.15399, 2021

---

We present an information-theoretic regularization technique for few-shot novel view synthesis based on neural implicit representation. 
        The proposed approach minimizes potential reconstruction inconsistency that happens due to insufficient viewpoints by imposing the entropy constraint of the density in each ray. 
        In addition, to alleviate the potential degenerate issue when all training images are acquired from almost redundant viewpoints,
        we further incorporate the spatially smoothness constraint into the estimated images by restricting information gains from a pair of rays with slightly different viewpoints. 
        The main idea of our algorithm is to make reconstructed scenes compact along individual rays and consistent across rays in the neighborhood. 
        The proposed regularizers can be plugged into most of existing neural volume rendering techniques based on NeRF in a straightforward way. 
        Despite its simplicity, we achieve consistently improved performance compared to existing neural view synthesis methods by large margins on multiple standard benchmarks. 

---

## Citation

If you find our work useful in your research, please cite:

```
@article{kim2021infonerf},
            title = {InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering},
            author = {Mijeong Kim and Seonguk Seo and Bohyung Han},
            journal = {arXiv.org}
            year = {2021},
        }
```

---

## Acknowlegements

This code borrows heavily from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).
