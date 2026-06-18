# ContraMOF

<strong>ContraMOF: Contrastive Multi-Task Pretraining for Metal-Organic Framework Property Prediction</strong>

This work extends the MOFormer framework with multi-task contrastive pretraining. The pretraining stage combines NT-Xent (InfoNCE) contrastive loss, BYOL-style asymmetric prediction, VICReg variance-covariance regularization, atom-type masking, and coordinate denoising into a unified objective with uncertainty-weighted task balancing. The resulting representations improve downstream property prediction performance across multiple benchmarks.

ContraMOF builds on our prior work, [MOFormer](https://github.com/zcao0420/MOFormer):  
<em>MOFormer: Self-Supervised Transformer Model for Metal-Organic Framework Property Prediction</em>, Cao et al., *JACS*, 2023. [[Paper]](https://pubs.acs.org/doi/10.1021/jacs.2c11420) [[arXiv]](https://arxiv.org/abs/2210.14188)

## Getting Started

### Installation

```
# create a new environment
$ conda create -n moformer python=3.9
$ conda activate moformer
$ conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
$ conda install --channel conda-forge pymatgen
$ pip install transformers
$ conda install -c conda-forge tensorboard
```

## Run the Model

### Pre-training

To pre-train the model with multi-task contrastive learning, run:
```
python pretrain_multitask.py
```
The configuration is defined in `config_multitask.yaml`. Update `graph_dataset.root_dir` to point to your CIF directory and adjust `dataloader.val_ratio` as needed. The pretraining dataset is available on [figshare](https://figshare.com/articles/journal_contribution/cif_tar_xz/23532918).

### Fine-tuning

To fine-tune the pre-trained Transformer on downstream regression tasks:
```
python finetune_transformer.py
```
The configuration is defined in `config_ft_transformer.yaml`. Update `fine_tune_from` to point to your pretrained checkpoint and `dataset.dataPath` to your target dataset.

## Acknowledgement

This work builds upon MOFormer:
- MOFormer: [Paper](https://pubs.acs.org/doi/10.1021/jacs.2c11420) and [Code](https://github.com/zcao0420/MOFormer)

Additionally, we acknowledge the following works and datasets:
- CGCNN: [Paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) and [Code](https://github.com/txie-93/cgcnn)
- Barlow Twins: [Paper](https://arxiv.org/abs/2103.03230) and [Code](https://github.com/facebookresearch/barlowtwins)
- Crystal Twins: [Paper](https://www.nature.com/articles/s41524-022-00921-5) and [Code](https://github.com/RishikeshMagar/Crystal-Twins)
- MOFid: [Paper](https://pubs.acs.org/doi/full/10.1021/acs.cgd.9b01050) and [Code](https://github.com/snurr-group/mofid/tree/master)
- Boyd&Woo Dataset: [Paper](https://www.nature.com/articles/s41586-019-1798-7)
- QMOF: [Paper1](https://www.cell.com/matter/fulltext/S2590-2385(21)00070-9) and [Paper2](https://www.nature.com/articles/s41524-022-00796-6)
- hMOF: [Paper](https://www.nature.com/articles/nchem.1192)

#### Questions about the code
Please feel free to raise GitHub issues for questions or concerns about the code.
