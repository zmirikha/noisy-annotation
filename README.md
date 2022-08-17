# Learning to Segment Skin Lesions from Noisy Annotations
Leveraging limited reliable annotations toward learning a robost to noise deep model.
<p align="center">
   <img width="450" src="https://github.com/zmirikha/noisy-annotation/blob/main/noisy_annotation.png" alt="[regional_map]"/>
</p>


# Abstract
Deep convolutional neural networks have driven substantial advancements in the automatic understanding of images. Requiring a large collection of images and their associated annotations is one of the main bottlenecks limiting the adoption of deep networks. In the task of medical image segmentation, requiring pixel-level semantic annotations performed by human experts exacerbate this difficulty. This paper proposes a new framework to train a fully convolutional segmentation network from a large set of cheap unreliable annotations and a small set of expert-level clean annotations. We propose a spatially adaptive reweighting approach to treat clean and noisy pixel-level annotations commensurately in the loss function. We deploy a meta-learning approach to assign higher importance to pixels whose loss gradient direction is closer to those of clean data. Our experiments on training the network using segmentation ground truth corrupted with different levels of annotation noise show how spatial reweighting improves the robustness of deep networks to noisy annotations.

## Keywords
Semantic segmentation, noisy annotations, meta-learning, robust deep learning

## Cite
Zahra Mirikharaji, Yiqi Yan, Ghassan Hamarneh, "[Learning to segment skin lesions from noisy annotations](https://www2.cs.sfu.ca/~hamarneh/ecopy/miccai_mil3id2019.pdf)", Domain adaptation and representation transfer and medical image learning with less labels and imperfect data (MICCAI MIL3ID), volume 11795, pages 207-215, 2019.


The corresponding bibtex entry is:

```
@incollection{mirikharaji2019learning,
  title={Learning to segment skin lesions from noisy annotations},
  author={Mirikharaji, Zahra and Yan, Yiqi and Hamarneh, Ghassan},
  booktitle={Domain adaptation and representation transfer and medical image learning with less labels and imperfect data},
  pages={207--215},
  year={2019},
  publisher={Springer}
}
```

## Usage
To get a quick help about the commands use:
```
python main.py --help
```
To modify the configuration, update settings.py file

To start training the model, first organize your training data as follows:
```
./data/
        Training/
                 noisy_gt/
                   (many gt files)
                 noisy_img/
                   (many image files)
                 clean_gt/
                   (many gt files)
                 clean_img/
                   (many image files)
```
Then run:
```
python main.py train
```

To evaluate, organzie your test data as follows:
```
./data/
        test_gt/
            (many gt files)
        test_img/
            (many image files)
```
Then run the following command:
```
python main.py inference
```
The predicted probability maps will be place in the results directory.
