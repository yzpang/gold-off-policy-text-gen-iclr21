# Text Generation by Learning from Demonstrations 

The README was last updated on February 18, 2021. The repo is based on [fairseq (v0.9.?)](https://github.com/pytorch/fairseq/tree/97d29d78e51e49de50e5105bcf4f9ebbd9fd7387).

## Paper

[arXiv (to be updated)](https://arxiv.org/abs/2009.07839)

## Prerequisites

Per fairseq usage, we need to install this particular modifed version fairseq. The simplest way: ```pip install --editable ./```. 

Due to pytorch changes, and given that we're using a slightly older version of fairseq (see below), please use pytorch version <= 1.6.0. However, the GOLD algorithm can be easily implemented on top of the latest fairseq (or most text generation codebases). 

### Datasets

For downloading CNN/DM and XSum datasets, we follow the instructions [here](https://github.com/pytorch/fairseq/tree/97d29d78e51e49de50e5105bcf4f9ebbd9fd7387/examples/bart); note that this link does not correspond to the latest fairseq. Our version of the CNN/DM input includes the "(CNN)" tags.

For downloading IWSLT14 De-En dataset, we follow the instructions [here](https://github.com/pytorch/fairseq/tree/97d29d78e51e49de50e5105bcf4f9ebbd9fd7387/examples/translation). The binary files are provided in our repo, in the directory ```data-bin```.

For downloading NQG dataset, we follow the instructions [here](https://github.com/clovaai/FocusSeq2Seq). The binary files are provided in the directory ```squad-bin```. 


## Code: experiments on transformer models using fairseq

For reproducibility, the code is based on a April 2020 [version](https://github.com/pytorch/fairseq/tree/97d29d78e51e49de50e5105bcf4f9ebbd9fd7387) of fairseq (based on release v0.9.0). However, it is easy to reimplement the GOLD algorithm in the latest version of fairseq and in another frameworks.

How to implement in the latest version of fairseq?
- If your GPUs "have large memory", then most of the implementation happens around the criterion code (for question generation, summarization, translation, the py file is ```./fairseq/criterions/label_smoothed_cross_entropy.py``` in the April 2020 version of fairseq). Note that the implementation in this repo uses this approach.
  - "Have large memory": Meaning the GPUs can store pi, pi-tilde, p_MLE at the same time; see Algorithm 1 in the paper. In our experiments (using the same datasets, same batch size, etc.), this would imply that the GPUs have ~24G of memory.
- If your GPUs cannot fit the above models, then you may need to input p_MLE probabilities as features. This can be done by first saving the probabilities into a text file or pickle file, and then loading them in the ```load_langpair_dataset``` function of ```./fairseq/tasks/translation.py``` (or other corresponding files for other tasks). 

How to implement in other codebase?
- See Algorithm 1 in the paper. The majority of the work will happen around the loss computation. We need to have three different models ready when computing losses: (1) pi, the network we're training; (2) pi-tilde, a slightly older version of pi (created to ensure training stability, similar to the periodic synchronization in deep Q-learning; (3) p_MLE, to compute rewards (but this can be pre-loaded in the form of input features, in case the GPU cannot fit the third model). 

### BART summarization generation fairseq issue

Given that there has been minor bugs with the fairseq BART summarization code ([details on original fairseq github](https://github.com/pytorch/fairseq/issues/1971)), we make the corresponding changes according to the fairseq authors' recommendation.
(1) In ```./fairseq/sequence_generator.py```, see the modification [here](https://github.com/pytorch/fairseq/issues/1971#issuecomment-610471553).
(2) In ```./fairseq/tasks/fairseq_task.py```, see the modification [here](https://github.com/pytorch/fairseq/issues/1971#issuecomment-610724245).
(3) In ```./fairseq/models/bart/hub_interface.py```, see the modification [here](https://github.com/pytorch/fairseq/issues/1971#issuecomment-610724245).
The above is already implemented in this repo.

### How to run?

The entry point for training is ```./fairseq_cli/train.py```. See ```./fairseq/options.py``` for possible flags. For CNN/DM, the script for running GOLD-p is provided in ```run_cnndm_goldp.sh```; the script for running GOLD-s (which often performs better than GOLD-p) is provided in ```run_cnndm_golds.sh```. [To add]

For BART validation and evaluation, we use the inference scripts provided in ```run_cnndm_inference.sh```, ```run_xsum_inference.sh```, ```run_squad_inference.sh```. For IWSLT14 De-En inference, the following few lines will do.
```
python -W ignore [path-to-fairseq_cli/generate.py] data-bin/iwslt14.tokenized.de-en \
    --path [path-to-model-checkpoint.pt] \
    --batch-size 128 --beam 5 --remove-bpe --gen-subset test  > [path-to-save-to-file]
```

### Transformer models

#### MLE model checkpoints

MLE model for CNN/DM: download (~5G) from [this page](https://github.com/pytorch/fairseq/tree/97d29d78e51e49de50e5105bcf4f9ebbd9fd7387/examples/bart)

MLE model for XSum: download (~5G) from [this page](https://github.com/pytorch/fairseq/tree/97d29d78e51e49de50e5105bcf4f9ebbd9fd7387/examples/bart)

MLE model for SQuAD: [download (~5G)]()

MLE model for IWSLT14 De-En: [download (~450M)](https://drive.google.com/file/d/1dynOAM-EJ4ptfUeP8G5DR_vKbkcIo9tI/view?usp=sharing)

#### GOLD-s model checkpoints

Model for CNN/DM: [download (~5G)](https://drive.google.com/file/d/1KW50i9JGIb9fI8DFWbln-id5dMX6ONiV/view?usp=sharing)

Model for XSum: [download (~5G)](https://drive.google.com/file/d/1etzTOHs9BHkqlajvhf9AhbYmqzYOCd-S/view?usp=sharing)

Model for SQuAD: [download (~5G)](https://drive.google.com/file/d/1-mTdmG5ip7nIj_brpOHpURS4a46_esrh/view?usp=sharing)

Model for IWSLT14 De-En: [download (~450M)](https://drive.google.com/file/d/1xdX-PmXCS7hFuw0CGvQ7KscY7owlcO2N/view?usp=sharing)

Not a lot of finetuning was done for the transformer models, so it is likely that more finetuning could reach better performance. 

Moreover, for summarization models, we use pyrouge+files2rouge to evaluate, based on [the fairseq instructions](https://github.com/pytorch/fairseq/tree/97d29d78e51e49de50e5105bcf4f9ebbd9fd7387/examples/bart) after pyrouge+files2rouge [installation](https://github.com/pltrdy/files2rouge). The package files2rouge has a common WordNet-2.0.exc.db error; see [this link](https://github.com/bheinzerling/pyrouge/issues/8) for the fix. 







## Citation

[The bibtex entry](https://yzpang.github.io/misc-files/bibs/pang2021text.txt)

## Authors' websites and contact info

[Richard Yuanzhe Pang](https://yzpang.me)

[He He](https://hhexiy.github.io)
