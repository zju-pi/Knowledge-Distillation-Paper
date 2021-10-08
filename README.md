# Knowledge-Distillation-Paper
This resposity maintains a series of papers, especially on knowledge distillation.

- [Knowledge-Distillation-Paper](#knowledge-distillation-paper)
  * [Feature-Map Distillation](#feature-map-distillation)
  * [Feature-Embedding Distillation](#feature-embedding-distillation)
  * [Online Knowledge Distillation](#online-knowledge-distillation)
  * [Data-Free Knowledge Distillation](#data-free-knowledge-distillation)
  * [Adversarial Distillation](#adversarial-distillation)
  * [Useful Resources](#useful-resources)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'></a></i></small>

## Early Works on Knowledge Distillation

- **Model Compression, KDD 2006** [[Paper]](https://dl.acm.org/doi/abs/10.1145/1150402.1150464)
  - *Cristian Buciluǎ, Rich Caruana, Alexandru Niculescu-Mizil.*
  
- **Do Deep Nets Really Need to be Deep?, NIPS 2014** [[Paper]](https://arxiv.org/abs/1312.6184)
  - *Lei Jimmy Ba, Rich Caruana.*
 
- **Distilling the Knowledge in a Neural Network, NIPS-workshop 2014** [[Paper]](https://arxiv.org/abs/1503.02531)
  - *Geoffrey Hinton, Oriol Vinyals, Jeff Dean.*
  
## Feature-Map Distillation

- **FitNets: Hints for Thin Deep Nets, ICLR 2015** [[Paper]](https://arxiv.org/abs/1412.6550) [[Theano]](https://github.com/adri-romsor/FitNets)
  - *Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio.*

- **Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, ICLR 2017** [[Paper]](https://arxiv.org/abs/1612.03928) [[PyTorch]](https://github.com/szagoruyko/attention-transfer)
  - *Sergey Zagoruyko, Nikos Komodakis.*

- **Similarity-Preserving Knowledge Distillation, CVPR 2019** [[Paper]](https://arxiv.org/abs/1907.09682) 
  - *Frederick Tung, Greg Mori.*
  
- **Variational Information Distillation for Knowledge Transfer, CVPR 2019** [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf) 
  - *Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai.*

- **Heterogeneous Knowledge Distillation using Information Flow Modeling, CVPR 2020** [[Paper]](https://arxiv.org/abs/2005.00727v1) [[Pytorch]](https://github.com/passalis/pkth)
  - *Nikolaos Passalis, Maria Tzelepi, Anastasios Tefas.*

- **Matching Guided Distillation, ECCV 2020** [[Paper]](https://arxiv.org/abs/2008.09958) [[Pytorch]](https://github.com/KaiyuYue/mgd)
  - *Kaiyu Yue, Jiangfan Deng, Feng Zhou.*

- **Cross-Layer Distillation with Semantic Calibration, AAAI 2021** [[Paper]](https://arxiv.org/abs/2012.03236) [[Pytorch]](https://github.com/DefangChen/SemCKD)
  - *Defang Chen, Jian-Ping Mei, Yuan Zhang, Can Wang, Zhe Wang, Yan Feng, Chun Chen.*

## Feature-Embedding Distillation

- **Learning Deep Representations with Probabilistic Knowledge Transfer, ECCV 2018** [[Paper]](https://arxiv.org/abs/1803.10837) [[Pytorch]](https://github.com/passalis/probabilistic_kt)
  - *Nikolaos Passalis, Anastasios Tefas.*
  
- **Knowledge Distillation via Instance Relationship Graph, CVPR 2019** [[Paper]](openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.pdf) [[Caffe]](https://github.com/yufanLIU/IRG)
  - *Yufan Liu, Jiajiong Cao, Bing Li, Chunfeng Yuan, Weiming Hu, Yangxi Li and Yunqiang Duan.*
  
- **Relational Knowledge Distillation, CVPR 2019** [[Paper]](https://arxiv.org/abs/1904.05068) [[Pytorch]](https://github.com/lenscloth/RKD)
  - *Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.*
  
- **Correlation Congruence for Knowledge Distillation, ICCV 2019** [[Paper]](https://arxiv.org/abs/1904.05068)
  - *Baoyun Peng, Xiao Jin, Jiaheng Liu, Shunfeng Zhou, Yichao Wu, Yu Liu, Dongsheng Li, Zhaoning Zhang.*
  
- **Contrastive Representation Distillation, ICLR 2020** [[Paper]](https://arxiv.org/abs/1910.10699) [[Pytorch]](https://github.com/HobbitLong/RepDistiller)
  - *Yonglong Tian, Dilip Krishnan, Phillip Isola.*

- **Distilling Holistic Knowledge with Graph Neural Networks, ICCV 2021** [[Paper]](https://arxiv.org/abs/2108.05507) [[Pytorch]](https://github.com/wyc-ruiker/HKD)
  - *Sheng Zhou, Yucheng Wang, Defang Chen, Jiawei Chen, Xin Wang, Can Wang, Jiajun Bu.*

## Online Knowledge Distillation

- **Deep Mutual Learning, CVPR 2018** [[Paper]](https://arxiv.org/abs/1804.03235) [[TensorFlow]](https://github.com/YingZhangDUT/Deep-Mutual-Learning)
  - *Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu.*

- **Large scale distributed neural network training through online distillation, ICLR 2018** [[Paper]](https://arxiv.org/abs/1804.03235) 
  - *Rohan Anil, Gabriel Pereyra, Alexandre Passos, Robert Ormandi, George E. Dahl and Geoffrey E. Hinton.*

- **Collaborative Learning for Deep Neural Networks, NIPS 2018** [[Paper]](https://arxiv.org/abs/1805.11761v2) 
  - *Guocong Song, Wei Chai.*

- **Knowledge Distillation by On-the-Fly Native Ensemble, NIPS 2018** [[Paper]](https://arxiv.org/abs/1806.04606)  [[PyTorch]](https://github.com/Lan1991Xu/ONE_NeurIPS2018)
  - *Xu Lan, Xiatian Zhu, Shaogang Gong.*

- **Online Knowledge Distillation with Diverse Peers, AAAI 2020** [[Paper]](https://arxiv.org/abs/1912.00350) [[Pytorch]](https://github.com/DefangChen/OKDDip-AAAI2020)
  - *Defang Chen, Jian-Ping Mei, Can Wang, Yan Feng and Chun Chen.*

- **Online Knowledge Distillation via Collaborative Learning, CVPR 2020** [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Online_Knowledge_Distillation_via_Collaborative_Learning_CVPR_2020_paper.pdf) 
  - *Qiushan Guo, Xinjiang Wang, Yichao Wu, Zhipeng Yu, Ding Liang, Xiaolin Hu, Ping Luo.*

## Multi-Teacher Knowledge Distillation

### Homogenous Label Space
- **Learning from Multiple Teacher Networks, KDD 2017** [[Paper]](https://dl.acm.org/doi/10.1145/3097983.3098135) 
  - *Shan You, Chang Xu, Chao Xu, Dacheng Tao.*

- **Multi-teacher Knowledge Distillation for Compressed Video Action Recognition on Deep Neural Networks, ICASSP 2019** [[Paper]](https://ieeexplore.ieee.org/document/8682450) 
  - *Meng-Chieh Wu, Ching-Te Chiu, Kun-Hsuan Wu.*

- **Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space, NIPS 2020** [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/91c77393975889bd08f301c9e13a44b7-Abstract.html) [[Pytorch]](https://github.com/AnTuo1998/AE-KD)
  - *Shangchen Du, Shan You, Xiaojie Li, Jianlong Wu, Fei Wang, Chen Qian, Changshui Zhang.*

- **Adaptive Knowledge Distillation Based on Entropy, ICASSP 2020** [[Paper]](https://ieeexplore.ieee.org/document/9054698) 
  - *Kisoo Kwon, Hwidong Na, Hoshik Lee, Nam Soo Kim.*

### Heterogenous Label Space

- **Amalgamating Knowledge towards Comprehensive Classification, AAAI 2019** [[Paper]](https://arxiv.org/abs/1811.02796v2) [[Pytorch]](https://github.com/zju-vipa/KamalEngine)
  - *Chengchao Shen, Xinchao Wang, Jie Song, Li Sun, Mingli Song.*

- **Unifying Heterogeneous Classifiers with Distillation, CVPR 2019** [[Paper]](https://arxiv.org/abs/1904.06062) 
  - *Jayakorn Vongkulbhisal, Phongtharin Vinayavekhin, Marco Visentini-Scarzanella.*


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

This section is maintained by [IsaccGuang](https://github.com/IsaccGuang).

- **Data-Free Knowledge Amalgamation via Group-Stack Dual-GAN, CVPR 2020** [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Data-Free_Knowledge_Amalgamation_via_Group-Stack_Dual-GAN_CVPR_2020_paper.pdf)  [[Supp]](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Ye_Data-Free_Knowledge_Amalgamation_CVPR_2020_supplemental.pdf) 
  - *Jingwen Ye, Yixin Ji, Xinchao Wang, Xin Gao, Mingli Song*

- **Generative Low-bitwidth Data Free Quantization, ECCV 2020** [[Paper]](https://arxiv.org/abs/2003.03603?context=cs) [[Pytorch]](https://github.com/xushoukai/GDFQ) 
  - *Shoukai Xu, Haokun Li, Bohan Zhuang, Jing Liu, Jiezhang Cao, Chuangrun Liang, Mingkui Tan*

- **Billion-scale semi-supervised learning for image classification, arXiv 2019.05** [[Paper]](https://arxiv.org/abs/1905.00546) [[Pytorch]](https://github.com/leaderj1001/Billion-scale-semi-supervised-learning) 
  - *I. Zeki Yalniz, Hervé Jégou, Kan Chen, Manohar Paluri, Dhruv Mahajan*

- **Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data, arXiv 2019.12** [[Paper]](https://arxiv.org/abs/1912.07768) [[Pytorch]](https://github.com/uber-research/GTN) 
  - *Felipe Petroski Such, Aditya Rawal, Joel Lehman, Kenneth O. Stanley, Jeff Clune*

- **MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation, arXiv 2020.05** [[Paper]](https://arxiv.org/abs/2005.03161) 
  - *Sanjay Kariyappa, Atul Prakash, Moinuddin Qureshi*

### Other Data-free Model Compression

- **Data-free Parameter Pruning for Deep Neural Networks, BMVC 2015** [[Paper]](https://arxiv.org/abs/1507.06149v1) 
  - *Suraj Srinivas, R. Venkatesh Babu*

- **Data-Free Quantization Through Weight Equalization and Bias Correction, ICCV 2019** [[Paper]](https://arxiv.org/abs/1906.04721v3) [[Pytorch1]](https://github.com/jakc4103/DFQ) [[Pytorch2]](https://github.com/ANSHUMAN87/Bias-Correction) 
  - *Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling*

- **DAC: Data-free Automatic Acceleration of Convolutional Networks, WACV 2019** [[Paper]](https://arxiv.org/abs/1812.08374) 
  - *Xin Li, Shuai Zhang, Bolan Jiang, Yingyong Qi, Mooi Choo Chuah, Ning Bi*

## Adversarial Distillation 
This section is maintained by [Azendure](https://github.com/Azendure).

- **Sobolev Training for Neural Networks, NIPS 2017** [[Paper]](https://arxiv.org/abs/1706.04859v3) [[Tensorflow]](https://github.com/mcneela/Sobolev)
  - *Wojciech Marian Czarnecki, Simon Osindero, Max Jaderberg, Grzegorz Świrszcz, Razvan Pascanu*

- **Knowledge Transfer with Jacobian Matching, ICML 2018** [[Paper]](https://arxiv.org/abs/1803.00443?context=cs.CV)
  - *Suraj Srinivas, Francois Fleuret*

- **On the Connection Between Adversarial Robustness and Saliency Map Interpretability, ICML 2019** [[Paper]](https://arxiv.org/abs/1905.04172) [[Code]](https://github.com/cetmann/robustness-interpretability)
  - *Christian Etmann, Sebastian Lunz, Peter Maass, Carola-Bibiane Schönlieb*

- **Adversarially Robust Distillation, AAAI 2020** [[Paper]](https://arxiv.org/abs/1905.09747v2) [[Pytorch]](https://github.com/goldblum/AdversariallyRobustDistillation)
  - *Micah Goldblum, Liam Fowl, Soheil Feizi, Tom Goldstein*

- **Jacobian Adversarially Regularized Networks for Robustness, ICLR 2020** [[Paper]](https://arxiv.org/abs/1912.10185) [[Tensorflow]](https://github.com/alvinchangw/JARN_ICLR2020)
  - *Alvin Chan, Yi Tay, Yew Soon Ong, Jie Fu*
  
- **What it Thinks is Important is Important: Robustness Transfers through Input Gradients, CVPR 2020** [[Paper]](https://arxiv.org/pdf/1912.05699.pdf)[[Tensorflow]](https://github.com/alvinchangw/IGAM_CVPR2020)
  - *Alvin Chan, Yi Tay, Yew Soon Ong*
  
### Adversarial Examples
 - **Intriguing properties of neural networks, ICLR 2014** [[Paper]](https://arxiv.org/pdf/1312.6199.pdf)
   - *Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, Rob Fergus*
 
 - **Explaining and Harnessing Adversarial Examples, ICLR 2015** [[Paper]](https://arxiv.org/abs/1412.6572)
   - *Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy*
   
 - **The Limitations of Deep Learning in Adversarial Settings, Euro S&P 2016** [[Paper]](https://arxiv.org/abs/1511.07528?context=cs)
   - *Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, Ananthram Swami*
   
 - **Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks, S&P 2016** [[Paper]](https://arxiv.org/abs/1511.04508v1)
   - *Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami*
   
 - **Universal adversarial perturbations, CVPR 2017** [[Paper]](https://arxiv.org/abs/1610.08401v1)
   - *Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, Pascal Frossard*
   
 - **Towards Evaluating the Robustness of Neural Networks, S&P 2017** [[Paper]](https://arxiv.org/pdf/1608.04644.pdf)[[Code]](https://github.com/carlini/nn_robust_attacks)
   - *Carlini Nicholas, Wagner David*
   
 - **Axiomatic Attribution for Deep Networks, ICML 2017** [[Paper]](https://arxiv.org/pdf/1703.01365.pdf) [[Code]](https://github.com/ankurtaly/Integrated-Gradients)
   - *Mukund Sundararajan, Ankur Taly, Qiqi Yan*
   
 - **Towards Deep Learning Models Resistant to Adversarial Attacks, ICLR 2018** [[Paper]](https://arxiv.org/pdf/1706.06083.pdf) [[Code1]](https://github.com/MadryLab/mnist_challenge) [[Code2]](https://github.com/MadryLab/cifar10_challenge)
   - *Madry, Aleksander，Makelov, Aleksandar，Schmidt, Ludwig，Tsipras, Dimitris，Vladu, Adrian*
   
 - **Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples, ICML 2018** [[Paper]](https://arxiv.org/abs/1802.00420)[[Code]](https://github.com/anishathalye/obfuscated-gradients)
   - *Anish Athalye, Nicholas Carlini, David Wagner*

 - **Adversarial Neural Pruning with Latent Vulnerability Suppression, ICML 2020** [[Paper]](https://arxiv.org/abs/1908.04355?context=cs.LG) [[Tensorflow]](https://github.com/divyam3897/ANP_VS)
   - Divyam Madaan, Jinwoo Shin, Sung Ju Hwang

## Useful Resources

- **Statistics of acceptance rate for the main AI conferences** [[Link]](https://github.com/lixin4ever/Conference-Acceptance-Rate)
- **AI conference deadlines** [[Link]](https://aideadlin.es/?sub=ML,CV,DM,SP)

### Accepted paper list
- **ICML:** 2020 [[Link]](https://icml.cc/virtual/2020/papers.html?filter=keywords)
- **ICLR:** 2020 [[Link]](https://openreview.net/group?id=ICLR.cc/2020/Conference)
- **NIPS:** 2019 [[Link]](https://neurips.cc/Conferences/2019/Schedule?type=Poster)
- **CVPR:** 2020 [[Link]](https://openaccess.thecvf.com/CVPR2020_search)
- **ICCV:** 2019 [[Link]](https://openaccess.thecvf.com/ICCV2019)
- **ECCV:** 2020 [[Link]](https://www.ecva.net/papers.php)

