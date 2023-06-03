# Knowledge-Distillation-Paper
This resposity maintains a collection of important papers on knowledge distillation.

- [Knowledge-Distillation-Paper](#knowledge-distillation-paper)
  * [Pioneering Papers](#pioneering-papers)
  * [Diffusion Meets Distillation](#diffusion-meets-distillation)
  * [Feature Distillation](#feature-distillation)
  * [Online Knowledge Distillation](#online-knowledge-distillation)
  * [Multi-Teacher Knowledge Distillation](#multi-teacher-knowledge-distillation)
  * [Data-Free Knowledge Distillation](#data-free-knowledge-distillation)
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

## Diffusion Meets Distillation

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
  - https://arxiv.org/abs/2005.00727v1
  - *Nikolaos Passalis, Maria Tzelepi, Anastasios Tefas.*

- **Cross-Layer Distillation with Semantic Calibration, AAAI 2021** [[Pytorch]](https://github.com/DefangChen/SemCKD)[[TKDE]](https://ieeexplore.ieee.org/document/9767633)
  - https://arxiv.org/abs/2012.03236
  - *Defang Chen, Jian-Ping Mei, Yuan Zhang, Can Wang, Zhe Wang, Yan Feng, Chun Chen.*
  
- **Distilling Holistic Knowledge with Graph Neural Networks, ICCV 2021** [[Pytorch]](https://github.com/wyc-ruiker/HKD)
  - https://arxiv.org/abs/2108.05507
  - *Sheng Zhou, Yucheng Wang, Defang Chen, Jiawei Chen, Xin Wang, Can Wang, Jiajun Bu.*

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

- **Knowledge Distillation by On-the-Fly Native Ensemble, NIPS 2018** [[PyTorch]](https://github.com/Lan1991Xu/ONE_NeurIPS2018)
  - https://arxiv.org/abs/1806.04606
  - *Xu Lan, Xiatian Zhu, Shaogang Gong.*

- **Online Knowledge Distillation with Diverse Peers, AAAI 2020** [[Pytorch]](https://github.com/DefangChen/OKDDip-AAAI2020)
  - https://arxiv.org/abs/1912.00350
  - *Defang Chen, Jian-Ping Mei, Can Wang, Yan Feng and Chun Chen.*

## Multi-Teacher Knowledge Distillation

### Homogenous Label Space

- **Distilling knowledge from ensembles of neural networks for speech recognition, INTERSPEECH 2016** [[Paper]](http://www-clmc.usc.edu/~chebotar/pub/chebotar_interspeech16.pdf) 
  - *Austin Waters, Yevgen Chebotar.*

- **Efficient Knowledge Distillation from an Ensemble of Teachers, INTERSPEECH 2017** [[Paper]](https://isca-speech.org/archive_v0/Interspeech_2017/pdfs/0614.PDF) 
  - *Takashi Fukuda, Masayuki Suzuki, Gakuto Kurata, Samuel Thomas, Jia Cui, Bhuvana Ramabhadran.*

- **Learning from Multiple Teacher Networks, KDD 2017** [[Paper]](https://dl.acm.org/doi/10.1145/3097983.3098135) 
  - *Shan You, Chang Xu, Chao Xu, Dacheng Tao.*

- **Multi-teacher Knowledge Distillation for Compressed Video Action Recognition on Deep Neural Networks, ICASSP 2019** [[Paper]](https://ieeexplore.ieee.org/document/8682450) 
  - *Meng-Chieh Wu, Ching-Te Chiu, Kun-Hsuan Wu.*

- **Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space, NIPS 2020** [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/91c77393975889bd08f301c9e13a44b7-Abstract.html) [[Pytorch]](https://github.com/AnTuo1998/AE-KD)
  - *Shangchen Du, Shan You, Xiaojie Li, Jianlong Wu, Fei Wang, Chen Qian, Changshui Zhang.*

- **Adaptive Knowledge Distillation Based on Entropy, ICASSP 2020** [[Paper]](https://ieeexplore.ieee.org/document/9054698) 
  - *Kisoo Kwon, Hwidong Na, Hoshik Lee, Nam Soo Kim.*

- **Reinforced Multi-Teacher Selection for Knowledge Distillation, AAAI 2021** [[Paper]](https://arxiv.org/abs/2012.06048v2) 
  - *Fei Yuan, Linjun Shou, Jian Pei, Wutao Lin, Ming Gong, Yan Fu, Daxin Jiang*

- **Adaptive Distillation: Aggregating Knowledge from Multiple Paths for Efficient Distillation, BMVC 2021** [[Paper]](https://arxiv.org/abs/2110.09674v2) [[Pytorch]](https://github.com/wyze-AI/AdaptiveDistillation)
  - *Sumanth Chennupati, Mohammad Mahdi Kamani, Zhongwei Cheng, Lin Chen*

- **Confidence-Aware Multi-Teacher Knowledge Distillation, ICASSP 2022** [[Paper]](https://arxiv.org/abs/2201.00007v1) [[Pytorch]](https://github.com/Rorozhl/CA-MKD)
  - *Hailin Zhang, Defang Chen, Can Wang.*



## Data-Free Knowledge Distillation 

- **Data-Free Knowledge Distillation for Deep Neural Networks, NIPS-workshop 2017** [[Paper]](https://arxiv.org/abs/1710.07535v2) [[Tensorflow]](https://github.com/iRapha/replayed_distillation)
  - *Raphael Gontijo Lopes, Stefano Fenu, Thad Starner*

- **DAFL: Data-Free Learning of Student Networks, ICCV 2019** [[Paper]](https://arxiv.org/abs/1904.01186) [[PyTorch]](https://github.com/huawei-noah/Efficient-Computing/tree/master/Data-Efficient-Model-Compression)
  - *Hanting Chen, Yunhe Wang, Chang Xu, Zhaohui Yang, Chuanjian Liu, Boxin Shi, Chunjing Xu, Chao Xu, Qi Tian*

- **Zero-Shot Knowledge Distillation in Deep Networks, ICML 2019** [[Paper]](https://arxiv.org/abs/1905.08114v1) [[Tensorflow]](https://github.com/vcl-iisc/ZSKD) 
  - *Gaurav Kumar Nayak, Konda Reddy Mopuri, Vaisakh Shaj, R. Venkatesh Babu, Anirban Chakraborty*

- **Zero-shot Knowledge Transfer via Adversarial Belief Matching, NIPS 2019** [[Paper]](https://arxiv.org/abs/1905.09768v4) [[Pytorch]](https://github.com/polo5/ZeroShotKnowledgeTransfer) 
  - *Paul Micaelli, Amos Storkey*

- **Knowledge Extraction with No Observable Data, NIPS 2019** [[Paper]](http://papers.nips.cc/paper/8538-knowledge-extraction-with-no-observable-data) [[Pytorch]](https://github.com/snudatalab/KegNet) 
  - *Jaemin Yoo, Minyong Cho, Taebum Kim, U Kang*

- **Dream Distillation: A Data-Independent Model Compression Framework, ICML-workshop 2019** [[Paper]](https://arxiv.org/abs/1905.07072v1) 
  - *Kartikeya Bhardwaj, Naveen Suda, Radu Marculescu*

- **DeGAN : Data-Enriching GAN for Retrieving Representative Samples from a Trained Classifier, AAAI 2020** [[Paper]](https://arxiv.org/abs/1912.11960v1) [[Pytorch]](https://github.com/vcl-iisc/DeGAN) 
  - *Sravanti Addepalli, Gaurav Kumar Nayak, Anirban Chakraborty, R. Venkatesh Babu*

- **Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion, CVPR 2020** [[Paper]](https://arxiv.org/abs/1912.08795v2) [[Pytorch]](https://github.com/NVlabs/DeepInversion) 
  - *Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek Hoiem, Niraj K. Jha, Jan Kautz*

- **The Knowledge Within: Methods for Data-Free Model Compression, CVPR 2020** [[Paper]](https://arxiv.org/abs/1912.01274) 
  - *Matan Haroush, Itay Hubara, Elad Hoffer, Daniel Soudry*

- **Data-Free Adversarial Distillation, ArXiv 2019.12** [[Paper]](https://arxiv.org/abs/1912.11006) [[Pytorch]](https://github.com/VainF/Data-Free-Adversarial-Distillation) 
  - *Gongfan Fang, Jie Song, Chengchao Shen, Xinchao Wang, Da Chen, Mingli Song*
  - Similar to `NIPS-2019 Zero-shot Knowledge Transfer via Adversarial Belief Matching`

- **Data-Free Knowledge Distillation with Soft Targeted Transfer Set Synthesis, AAAI 2021** [[Paper]](https://arxiv.org/abs/2104.04868) 
  - *Zi Wang*

- **Learning Student Networks in the Wild, CVPR 2021** [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Student_Networks_in_the_Wild_CVPR_2021_paper.html) [[Pytorch]](https://github.com/huawei-noah/Efficient-Computing/tree/master/Data-Efficient-Model-Compression/DFND) 
  - *Hanting Chen, Tianyu Guo, Chang Xu, Wenshuo Li, Chunjing Xu, Chao Xu, Yunhe Wang*

- **Contrastive Model Inversion for Data-Free Knowledge Distillation, IJCAI 2021** [[Paper]](https://arxiv.org/abs/2105.08584) [[Pytorch]](https://github.com/zju-vipa/DataFree) 
  - *Gongfan Fang, Jie Song, Xinchao Wang, Chengchao Shen, Xingen Wang, Mingli Song*


## Useful Resources

- **Statistics of acceptance rate for the main AI conferences** [[Link]](https://github.com/lixin4ever/Conference-Acceptance-Rate)
- **AI conference deadlines** [[Link]](https://aideadlin.es/?sub=ML,CV,DM,SP)
