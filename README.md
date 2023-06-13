# Knowledge-Distillation-Paper
This resposity maintains a collection of important papers on knowledge distillation.

- [Knowledge-Distillation-Paper](#knowledge-distillation-paper)
  * [Pioneering Papers](#pioneering-papers)
  * [Distillation Accelerates Diffusion Models](#distillation-accelerates-diffusion-models)
  * [Feature Distillation](#feature-distillation)
  * [Online Knowledge Distillation](#online-knowledge-distillation)
  * [Multi-Teacher Knowledge Distillation](#multi-teacher-knowledge-distillation)
  * [Data-Free Knowledge Distillation](#data-free-knowledge-distillation)
  * [Distillation for Segmentation](#distillation-for-segmentation)
  * [Useful Resources](#useful-resources)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'></a></i></small>

## Pioneering Papers

- **Model Compression, KDD 2006** 
  - https://dl.acm.org/doi/abs/10.1145/1150402.1150464
  - *Cristian Buciluǎ, Rich Caruana, Alexandru Niculescu-Mizil.*
  
- **Do Deep Nets Really Need to be Deep?, NeurIPS 2014** 
  - https://arxiv.org/abs/1312.6184
  - *Lei Jimmy Ba, Rich Caruana.*
 
- **Distilling the Knowledge in a Neural Network, NeurIPS-workshop 2014** 
  - https://arxiv.org/abs/1503.02531
  - *Geoffrey Hinton, Oriol Vinyals, Jeff Dean.*

## Distillation Accelerates Diffusion Models

<p align="center"><strong>Extremely Promising</strong> !!!!!</p>

- **Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed** [[Tensorflow]](https://github.com/tcl9876/Denoising_Student/blob/master/training.py)
  - https://arxiv.org/abs/2101.02388
  - *Eric Luhman, Troy Luhman*

- **Progressive Distillation for Fast Sampling of Diffusion Models, ICLR 2022** [[Tensorflow]](https://github.com/google-research/google-research/tree/master/diffusion_distillation)
  - https://arxiv.org/abs/2202.00512
  - *Tim Salimans, Jonathan Ho*

- **Accelerating Diffusion Sampling with Classifier-based Feature Distillation, ICME 2023** [[PyTorch]](https://github.com/zju-SWJ/RCFD)
  - https://arxiv.org/abs/2211.12039
  - *Wujie Sun, Defang Chen, Can Wang, Deshi Ye, Yan Feng, Chun Chen*

- **Fast Sampling of Diffusion Models via Operator Learning** 
  - https://arxiv.org/abs/2211.13449
  - *Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, Anima Anandkumar*

- **Consistency Models, ICML 2023** [[PyTorch]](https://github.com/openai/consistency_models)
  - https://arxiv.org/abs/2303.01469
  - *Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever*

- **TRACT: Denoising Diffusion Models with Transitive Closure Time-Distillation** [[PyTorch]](https://github.com/apple/ml-tract)
  - https://arxiv.org/abs/2303.04248
  - *David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbott, Eric Gu*

- **A Geometric Perspective on Diffusion Models** 
  - https://arxiv.org/abs/2305.19947
  - *Defang Chen, Zhenyu Zhou, Jian-Ping Mei, Chunhua Shen, Chun Chen, Can Wang*


## Feature Distillation

- **FitNets: Hints for Thin Deep Nets, ICLR 2015** [[Theano]](https://github.com/adri-romsor/FitNets)
  - https://arxiv.org/abs/1412.6550
  - *Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio.*

- **Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, ICLR 2017** [[PyTorch]](https://github.com/szagoruyko/attention-transfer)
  - https://arxiv.org/abs/1612.03928
  - *Sergey Zagoruyko, Nikos Komodakis.*

- **Learning Deep Representations with Probabilistic Knowledge Transfer, ECCV 2018** [[Pytorch]](https://github.com/passalis/probabilistic_kt)
  - https://arxiv.org/abs/1803.10837
  - *Nikolaos Passalis, Anastasios Tefas.*
  
- **Relational Knowledge Distillation, CVPR 2019** [[Pytorch]](https://github.com/lenscloth/RKD)
  - https://arxiv.org/abs/1904.05068
  - *Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.*

- **Variational Information Distillation for Knowledge Transfer, CVPR 2019** 
  - https://arxiv.org/abs/1904.05835
  - *Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai.*
  
- **Similarity-Preserving Knowledge Distillation, CVPR 2019** 
  - https://arxiv.org/abs/1907.09682
  - *Frederick Tung, Greg Mori.*
  
- **Contrastive Representation Distillation, ICLR 2020** [[Pytorch]](https://github.com/HobbitLong/RepDistiller)
  - https://arxiv.org/abs/1910.10699
  - *Yonglong Tian, Dilip Krishnan, Phillip Isola.*

- **Heterogeneous Knowledge Distillation using Information Flow Modeling, CVPR 2020** [[Pytorch]](https://github.com/passalis/pkth)
  - https://arxiv.org/abs/2005.00727
  - *Nikolaos Passalis, Maria Tzelepi, Anastasios Tefas.*

- **Cross-Layer Distillation with Semantic Calibration, AAAI 2021** [[Pytorch]](https://github.com/DefangChen/SemCKD)[[TKDE]](https://ieeexplore.ieee.org/document/9767633)
  - https://arxiv.org/abs/2012.03236
  - *Defang Chen, Jian-Ping Mei, Yuan Zhang, Can Wang, Zhe Wang, Yan Feng, Chun Chen.*

- **Distilling Knowledge via Knowledge Review, CVPR 2021** [[Pytorch]](https://github.com/dvlab-research/ReviewKD)
  - https://arxiv.org/abs/2104.09044
  - *Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia.*
  
- **Distilling Holistic Knowledge with Graph Neural Networks, ICCV 2021** [[Pytorch]](https://github.com/wyc-ruiker/HKD)
  - https://arxiv.org/abs/2108.05507
  - *Sheng Zhou, Yucheng Wang, Defang Chen, Jiawei Chen, Xin Wang, Can Wang, Jiajun Bu.*

- **Decoupled Knowledge Distillation, CVPR 2022** [[Pytorch]](https://github.com/megvii-research/mdistiller)
  - https://arxiv.org/abs/2203.08679
  - *Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, Jiajun Liang*

- **Knowledge Distillation with the Reused Teacher Classifier, CVPR 2022** [[Pytorch]](https://github.com/DefangChen/SimKD)
  - https://arxiv.org/abs/2203.14001
  - *Defang Chen, Jian-Ping Mei, Hailin Zhang, Can Wang, Yan Feng, Chun Chen.*

## Online Knowledge Distillation

- **Deep Mutual Learning, CVPR 2018** [[TensorFlow]](https://github.com/YingZhangDUT/Deep-Mutual-Learning)
  - https://arxiv.org/abs/1706.00384
  - *Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu.*

- **Large scale distributed neural network training through online distillation, ICLR 2018** 
  - https://arxiv.org/abs/1804.03235
  - *Rohan Anil, Gabriel Pereyra, Alexandre Passos, Robert Ormandi, George E. Dahl and Geoffrey E. Hinton.*

- **Knowledge Distillation by On-the-Fly Native Ensemble, NeurIPS 2018** [[PyTorch]](https://github.com/Lan1991Xu/ONE_NeurIPS2018)
  - https://arxiv.org/abs/1806.04606
  - *Xu Lan, Xiatian Zhu, Shaogang Gong.*

- **Online Knowledge Distillation with Diverse Peers, AAAI 2020** [[Pytorch]](https://github.com/DefangChen/OKDDip-AAAI2020)
  - https://arxiv.org/abs/1912.00350
  - *Defang Chen, Jian-Ping Mei, Can Wang, Yan Feng and Chun Chen.*

- **Feature-map-level Online Adversarial Knowledge Distillation, ICML 2020** 
  - https://arxiv.org/abs/2002.01775
  - *Inseop Chung, SeongUk Park, Jangho Kim, Nojun Kwak.*

- **Peer collaborative learning for online knowledge distillation, AAAI 2021** 
  - https://arxiv.org/abs/2006.04147
  - *Guile Wu, Shaogang Gong.*

## Multi-Teacher Knowledge Distillation

- **Distilling knowledge from ensembles of neural networks for speech recognition, INTERSPEECH 2016** 
  - https://www.isca-speech.org/archive_v0/Interspeech_2016/pdfs/1190.PDF
  - *Austin Waters, Yevgen Chebotar.*

- **Efficient Knowledge Distillation from an Ensemble of Teachers, INTERSPEECH 2017** 
  - https://isca-speech.org/archive_v0/Interspeech_2017/pdfs/0614.PDF
  - *Takashi Fukuda, Masayuki Suzuki, Gakuto Kurata, Samuel Thomas, Jia Cui, Bhuvana Ramabhadran.*

- **Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space, NeurIPS 2020** [[Pytorch]](https://github.com/AnTuo1998/AE-KD)
  - https://proceedings.neurips.cc/paper/2020/hash/91c77393975889bd08f301c9e13a44b7-Abstract.html 
  - *Shangchen Du, Shan You, Xiaojie Li, Jianlong Wu, Fei Wang, Chen Qian, Changshui Zhang.*

- **Reinforced Multi-Teacher Selection for Knowledge Distillation, AAAI 2021** 
  - https://arxiv.org/abs/2012.06048
  - *Fei Yuan, Linjun Shou, Jian Pei, Wutao Lin, Ming Gong, Yan Fu, Daxin Jiang*

- **Confidence-Aware Multi-Teacher Knowledge Distillation, ICASSP 2022** [[Pytorch]](https://github.com/rorozhl/mmkd)
  - https://arxiv.org/abs/2201.00007
  - *Hailin Zhang, Defang Chen, Can Wang.*

- **Adaptive Multi-Teacher Knowledge Distillation with Meta-Learning, ICME 2023** [[Pytorch]](https://github.com/Rorozhl/CA-MKD)
  - https://arxiv.org/abs/2306.06634
  - *Hailin Zhang, Defang Chen, Can Wang.*

## Data-Free Knowledge Distillation 

- **Data-Free Knowledge Distillation for Deep Neural Networks, NeurIPS-workshop 2017** [[Tensorflow]](https://github.com/iRapha/replayed_distillation)
  - https://arxiv.org/abs/1710.07535 
  - *Raphael Gontijo Lopes, Stefano Fenu, Thad Starner*

- **DAFL: Data-Free Learning of Student Networks, ICCV 2019** [[PyTorch]](https://github.com/huawei-noah/Efficient-Computing/tree/master/Data-Efficient-Model-Compression)
  - https://arxiv.org/abs/1904.01186
  - *Hanting Chen, Yunhe Wang, Chang Xu, Zhaohui Yang, Chuanjian Liu, Boxin Shi, Chunjing Xu, Chao Xu, Qi Tian*

- **Zero-Shot Knowledge Distillation in Deep Networks, ICML 2019** [[Tensorflow]](https://github.com/vcl-iisc/ZSKD) 
  - https://arxiv.org/abs/1905.08114 
  - *Gaurav Kumar Nayak, Konda Reddy Mopuri, Vaisakh Shaj, R. Venkatesh Babu, Anirban Chakraborty*

- **Zero-shot Knowledge Transfer via Adversarial Belief Matching, NeurIPS 2019** [[Pytorch]](https://github.com/polo5/ZeroShotKnowledgeTransfer)
  - https://arxiv.org/abs/1905.09768
  - *Paul Micaelli, Amos Storkey*

- **Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion, CVPR 2020** [[Pytorch]](https://github.com/NVlabs/DeepInversion) 
  - https://arxiv.org/abs/1912.08795
  - *Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek Hoiem, Niraj K. Jha, Jan Kautz*

- **The Knowledge Within: Methods for Data-Free Model Compression, CVPR 2020** 
  - https://arxiv.org/abs/1912.01274
  - *Matan Haroush, Itay Hubara, Elad Hoffer, Daniel Soudry*

- **Contrastive Model Inversion for Data-Free Knowledge Distillation, IJCAI 2021** [[Pytorch]](https://github.com/zju-vipa/DataFree) 
  - https://arxiv.org/abs/2105.08584 
  - *Gongfan Fang, Jie Song, Xinchao Wang, Chengchao Shen, Xingen Wang, Mingli Song*

- **Customizing Synthetic Data for Data-Free Student Learning, ICME 2023** [[Pytorch]](https://github.com/luoshiya/CSD) 
  - *Shiya Luo, Defang Chen, Can Wang*

## Distillation for Segmentation

- **Structured Knowledge Distillation for Dense Prediction, CVPR 2019, TPAMI 2020** [[Pytorch]](https://github.com/irfanICMLL/structure_knowledge_distillation)
  - https://arxiv.org/abs/1903.04197
  - *Yifan Liu, Changyong Shun, Jingdong Wang, Chunhua Shen.*

- **Channel-wise Knowledge Distillation for Dense Prediction, ICCV 2021** [[Pytorch]](https://github.com/irfanICMLL/TorchDistiller/tree/main/SemSeg-distill)
  - https://arxiv.org/abs/2011.13256
  - *Changyong Shu, Yifan Liu, Jianfei Gao, Zheng Yan, Chunhua Shen.*

- **Holistic Weighted Distillation for Semantic Segmentation, ICME 2023** [[Pytorch]](https://github.com/zju-SWJ/HWD) 
  - *Wujie Sun, Defang Chen, Can Wang, Deshi Ye, Yan Feng, Chun Chen.*

## Useful Resources

- **Acceptance rates of the main AI conferences** [[Link]](https://github.com/lixin4ever/Conference-Acceptance-Rate)
- **AI conference deadlines** [[Link]](https://aideadlin.es/?sub=ML,CV,SP) 
- **CCF conference deadlines** [[Link]](https://ccfddl.github.io/) 
