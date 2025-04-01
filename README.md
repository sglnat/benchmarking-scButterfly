[![PyPI](https://img.shields.io/pypi/v/scbutterfly)](https://pypi.org/project/scbutterfly)
[![Documentation Status](https://readthedocs.org/projects/scbutterfly/badge/?version=latest)](https://scbutterfly.readthedocs.io/en/latest/?badge=stable)
[![Downloads](https://pepy.tech/badge/scbutterfly)](https://pepy.tech/project/scbutterfly)


# scButterfly: a versatile single-cell cross-modality translation method via dual-aligned variational autoencoders

## Installation

run a docker container using: 
```bash
docker run --gpus all -p 8080:8080 --rm -ti -v /raid/asmagulova/scbutterfly:/workspace scbutterfly
```

Change the path to your working directory on DGX 

and then run: 

```
pip install scButterfly
```

This will automatically install all necessary dependencies into your docker container. 

The main Jupyter Notebook ./scButterfly_RNA_to_ATAC_unpaired_normal_pre_selina.ipynb will run all analyses at once: it will read the raw data files, add cell types to the metadata, load the data into the scButterfly, preprocess files, construct the model, train it, and produce a prediction. 

The model runs in the unpaired mode, with RNA and ATAC from the multiome taken as training data, and the RNA Selina is being used as ground to predict ATAC. The ATAC to RNA prediction is disabled in this Notebook. Refer to the 'split_dataset.py' for the 'unpaired_split_dataset_fixed_rna' function that is custom. 




## Quick Start

Use [this tutorial](https://scbutterfly.readthedocs.io/en/latest/Tutorial/RNA_ATAC_unpaired_prediction/RNA_ATAC_unpaired_scButterfly-T.html) for reference. 

Generate a scButterfly model first with following process:

```python
from scButterfly.butterfly import Butterfly
butterfly = Butterfly()
```

### 1. Data preprocessing

* Loading the data and preprocessing 

  ```python
  ATAC_data = anndata.read_h5ad('atac_multiome_normal.h5ad')
  RNA_data = anndata.read_h5ad('selina_multiome_normal.h5ad')
  RNA_data.obs.index = pd.Series([str(i) for i in range(len(RNA_data.obs.index))])
  ATAC_data.obs.index = pd.Series([str(i) for i in range(len(ATAC_data.obs.index))])

  from scButterfly.data_processing import RNA_data_preprocessing, ATAC_data_preprocessing
  ```
  
  | Parameters    | Description                                                                                |
  | ------------- | ------------------------------------------------------------------------------------------ |
  | RNA_data      | AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes. |
  | ATAC_data     | AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks. |
  | train_id      | A list of cell IDs for training.                                                           |
  | test_id       | A list of cell IDs for testing.                                                            |
  | validation_id | An optional list of cell IDs for validation, if setted None, butterfly will use a default setting of 20% cells in train_id. |
  
  Anndata object is a Python object/container designed to store single-cell data in Python packege [**anndata**](https://anndata.readthedocs.io/en/latest/) which is seamlessly integrated with [**scanpy**](https://scanpy.readthedocs.io/en/stable/), a widely-used Python library for single-cell data analysis.

* For data preprocessing:

  This will normalise & filter data. Parameters can be changed. Note that with this, the predicted output is normalised in the same way.
  
  ```python
  RNA_data = RNA_data_preprocessing(
    RNA_data,
    normalize_total=True,
    log1p=True,
    use_hvg=True,
    n_top_genes=3000,
    save_data=True,
    file_path='./processed_data/',
    logging_path=None
    )
  ATAC_data = ATAC_data_preprocessing(
      ATAC_data,
      binary_data=True,
      filter_features=True,
      fpeaks=0.005,
      tfidf=True,
      normalize=True,
      save_data=True,
      file_path='./processed_data/',
      logging_path=None
  )[0]
  ```
  
  You could save processed data or output process logging to a file using following parameters.
  
  | Parameters   | Description                                                                                  |
  | ------------ | -------------------------------------------------------------------------------------------- |
  | save_data    | optional, choose save the processed data or not, default False.                              |
  | file_path    | optional, the path for saving processed data, only used if `save_data` is True, default None.  |
  | logging_path | optional, the path for output process logging, if not save, set it None, default None.       |

  scButterfly also support to refine this process using other parameters (more details on [scButterfly documents](http://scbutterfly.readthedocs.io/)), however, we strongly recommend the default settings to keep the best result for model.
  
### 2. Model training

* Before model training, you should define the dataset split:
  ```python
  import importlib
  
  import scButterfly.split_datasets
  
  importlib.reload(scButterfly.split_datasets)
  
  from scButterfly.split_datasets import unpaired_split_dataset_fixed_rna
  id_list = unpaired_split_dataset_fixed_rna(RNA_data, ATAC_data)
  train_id_r, train_id_a, validation_id_r, validation_id_a, test_id_r, test_id_a = id_list[0]
  ```
  
  ```python
  butterfly.construct_model(chrom_list)
  ```
  
* scButterfly need a list of peaks count for each chromosome, remember to sort peaks with chromosomes.
  
  | Parameters   | Description                                                                                    |
  | ------------ | ---------------------------------------------------------------------------------------------- |
  | chrom_list   | a list of peaks count for each chromosome, remember to sort peaks with chromosomes.            |
  | logging_path | optional, the path for output model structure logging, if not save, set it None, default None. |

  ```python
  import pandas as pd
  
  genes = ATAC_data.var.index 
  chromosomes = genes.str.extract(r'^(chr[^\-]+)')[0] 
  ATAC_data.var['chrom'] = pd.Categorical(chromosomes)
  print(ATAC_data.var['chrom'])
  ```
  ```python 
  chrom_list = []
  last_one = ''
  for i in range(len(ATAC_data.var['chrom'])):
      temp = ATAC_data.var['chrom'][i]
      if temp[0 : 3] == 'chr':
          if not temp == last_one:
              chrom_list.append(1)
              last_one = temp
          else:
              chrom_list[-1] += 1
      else:
          chrom_list[-1] += 1
          
  print(chrom_list, end="")
  ```
  
* scButterfly model could be easily constructed & trained as following:
  
  ```python
  from scButterfly.train_model import Model
  import torch
  import torch.nn as nn
  ```

  ```python
  RNA_input_dim = len([i for i in RNA_data.var['highly_variable'] if i])
  ATAC_input_dim = ATAC_data.X.shape[1]
  
  R_kl_div = 1 / RNA_input_dim * 20
  A_kl_div = 1 / ATAC_input_dim * 20
  kl_div = R_kl_div + A_kl_div
  ```

  ```python
  model = Model(
    R_encoder_nlayer = 2,
    A_encoder_nlayer = 2,
    R_decoder_nlayer = 2,
    A_decoder_nlayer = 2,
    R_encoder_dim_list = [RNA_input_dim, 256, 128],
    A_encoder_dim_list = [ATAC_input_dim, 32 * len(chrom_list), 128],
    R_decoder_dim_list = [128, 256, RNA_input_dim],
    A_decoder_dim_list = [128, 32 * len(chrom_list), ATAC_input_dim],
    R_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
    A_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
    R_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
    A_decoder_act_list = [nn.LeakyReLU(), nn.Sigmoid()],
    translator_embed_dim = 128,
    translator_input_dim_r = 128,
    translator_input_dim_a = 128,
    translator_embed_act_list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
    discriminator_nlayer = 1,
    discriminator_dim_list_R = [128],
    discriminator_dim_list_A = [128],
    discriminator_act_list = [nn.Sigmoid()],
    dropout_rate = 0.1,
    R_noise_rate = 0.5,
    A_noise_rate = 0.3,
    chrom_list = chrom_list,
    logging_path = None,
    RNA_data = RNA_data,
    ATAC_data = ATAC_data
  )
  ```

  ```python
  model.train(
      R_encoder_lr = 0.001,
      A_encoder_lr = 0.001,
      R_decoder_lr = 0.001,
      A_decoder_lr = 0.001,
      R_translator_lr = 0.001,
      A_translator_lr = 0.001,
      translator_lr = 0.001,
      discriminator_lr = 0.005,
      R2R_pretrain_epoch = 100,
      A2A_pretrain_epoch = 100,
      lock_encoder_and_decoder = False,
      translator_epoch = 200,
      patience = 50,
      batch_size = 64,
      r_loss = nn.MSELoss(size_average=True),
      a_loss = nn.BCELoss(size_average=True),
      d_loss = nn.BCELoss(size_average=True),
      loss_weight = [1, 2, 1, R_kl_div, A_kl_div, kl_div],
      train_id_r = train_id_r,
      train_id_a = train_id_a,
      validation_id_r = validation_id_r,
      validation_id_a = validation_id_a,
      output_path = './normal_cells_unpaired',
      seed = 19193,
      kl_mean = True,
      R_pretrain_kl_warmup = 50,
      A_pretrain_kl_warmup = 50,
      translation_kl_warmup = 50,
      load_model = None,
      logging_path = './normal_cells_unpaired'
  )
  ``` 

  | Parameters   | Description                                                                             |
  | ------------ | --------------------------------------------------------------------------------------- |
  | output_path  | optional, path for model check point, if None, using './model' as path, default None.   |
  | load_model   | optional, the path for load pretrained model, if not load, set it None, default None.   |
  | logging_path | optional, the path for output training logging, if not save, set it None, default None. |
  
  scButterfly also support to refine the model structure and training process using other parameters for `butterfly.construct_model()` and `butterfly.train_model()` (more details on [scButterfly documents](http://scbutterfly.readthedocs.io/)).
  
### 3. Predicting and evaluating

* scButterfly provide a predicting API, you could get predicted profiles as follow:
  
  ```python
  R2A_predict = model.test(
    test_id_r = test_id_r,
    batch_size = 64,
    model_path = './normal_cells_unpaired',
    load_model = True,
    output_path = None,
    test_cluster = False,
    test_figure = False,
    output_data = False,
    return_predict = True
  )
  ```
  
  A series of evaluating method also be integrated in this function, you could get these evaluation using parameters:
  
  | Parameters    | Description                                                                                 |
  | ------------- | ------------------------------------------------------------------------------------------- |
  | output_path   | optional, path for model evaluating output, if None, using './model' as path, default None. |
  | load_model    | optional, the path for load pretrained model, if not load, set it None, default False.      |
  | model_path    | optional, the path for pretrained model, only used if `load_model` is True, default None.   |
  | test_cluster  | optional, test the correlation evaluation or not, including **AMI**, **ARI**, **HOM**, **NMI**, default False.|
  | test_figure   | optional, draw the **tSNE** visualization for prediction or not, default False.             |
  | output_data   | optional, output the prediction to file or not, if True, output the prediction to `output_path/A2R_predict.h5ad` and `output_path/R2A_predict.h5ad`, default False.                                          |

### We also provide richer tutorials and documents for scButterfly in [scButterfly documents](http://scbutterfly.readthedocs.io/), including more details of provided APIs for customing data preprocessing, model structure and training strategy. The source code of experiments for scButterfly is available at [source code](https://github.com/BioX-NKU/scButterfly_source), including more detailed source code for scButterfly.
