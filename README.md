# PAM: Paraphrase AMR-Centric Evaluation Metric

This repo contains the code for the paper PAM: Paraphrase AMR-Centric Evaluation Metric, by Afonso Sousa & Henrique Lopes Cardoso (ACL Findings 2025).

Paraphrasing is rooted in semantics, which makes evaluating paraphrase generation systems hard. Current paraphrase generators are typically evaluated using borrowed metrics from adjacent text-to-text tasks, like machine translation or text summarization. These metrics tend to have ties to the surface form of the reference text. This is not ideal for paraphrases as we typically want variation in the lexicon while persisting semantics. To address this problem, and inspired by learned similarity evaluation on plain text, we propose `PAM`, a **P**araphrase **A**MR-Centric Evaluation **M**etric. This metric uses AMR graphs extracted from the input text, which consist of semantic structures agnostic to the text surface form, making the resulting evaluation metric more robust to variations in syntax or lexicon. Additionally, we evaluated \pam on different semantic textual similarity datasets and found that it improves the correlations with human semantic scores when compared to other AMR-based metrics.

## Installation
First, to create a fresh conda environment with all the used dependencies run:
```
conda env create -f environment.yml
```

Additionally, for most scripts you will need the pretrained AMR parser. We used _parse_xfm_bart_large_ from [here](https://github.com/bjascob/amrlib-models). Download it, rename it to `amr_parser`, and place it in the root directory.

## Preprocess data
Go to [data/README](./data/README.md) and extract the third-party data into `/data` folder.
<pre>
data
└── <em>dataset_name</em>
    └── main
        └──raw
           │ src.dev.amr
           │ src.test.amr
           │ tgt.dev.amr
           │ tgt.test.amr
</pre>

Then use [merge_dataset.sh](scripts/merge_dataset.sh) to merge the information into a json file. For the aforementioned example, the output file should be placed under `/main`.

## Train and test models
To train/test PAM or any other model refered to in the paper you can run the corresponding script. For example:
```
sh ./scripts/train_pam.sh
```

```
sh ./scripts/test_pam.sh
```

## Further finetune
To further finetune the trained model on Quora Question Pairs (QQP), run:

```
sh ./scripts/paraphrase_finetune.sh
```

## Other experiments reported in the paper
For many experiments reported in the paper, we used third-party libraries integrated into our source code, which require you to extract them to the root directory and potentially install the respective packages -- for example, [AlignScore](https://github.com/yuh-zha/AlignScore).

Others, like `WWLK`, were computed using the [original source code](https://github.com/flipz357/weisfeiler-leman-amr-metrics).

Some files were used for smaller, single experiments:
- [computing static embeddings](compare_static_embeddings.py)
- [plotting `PAM` and `SBERT` score distribution](pam_vs_sbert.py)
- [auxiliar to compute AMR similarity metrics](test_amr_sim_metrics.py)
- [compute computational cost](computational_cost.py)
- [compute statistics for ETPC dataset](test_etpc.py).

## Acknowledgements
This project used code and took inspiration from the following open source projects:
- [AMRSim](https://github.com/zzshou/AMRSim)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)

## License
This project is released under the [MIT License](LICENSE).