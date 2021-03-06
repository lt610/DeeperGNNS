# Variational Perspectives on Graph Convolutional Networks
## Results
| Method | Cora | Citeseer | Pubmed | arxiv |
| --- | --- | --- | --- | --- |
| SGC | 80.8(K=6) | 69.2(47) | 77.0(4) | 67.01(3) |
| VSGC(alpha=1) | 81.6(K=47) | 70.0(36) | 79.4(45) | 67.09(3) |
| VSGC(alpha=0.5) | 82.0(k=50) | 69.9(34) | 79.3(40) | 67.13(3) |
## Trends
![image](https://github.com/lt610/DeepGNNS/blob/master/result/images/cora.png)
![image](https://github.com/lt610/DeepGNNS/blob/master/result/images/citeseer.png)
![image](https://github.com/lt610/DeepGNNS/blob/master/result/images/pubmed.png)
![image](https://github.com/lt610/DeepGNNS/blob/master/result/images/ogbn-arxiv.png)
# DeepGNNs
## Models

| Model | Paper |
| --- | --- |
| GCNII, GCNII\* | Simple and Deep Graph Convolutional Networks |
| DAGNN | Towards Deeper Graph Neural Networks |
| CGCN | Continuous Graph Neural Networks |
| CMUGCN | Contrastive Multi-View Representation Learning on Graphs |
| GCN-BBDE, GCN-BBGDC | Bayesian Graph Neural Networks with Adaptive Connection Sampling |
## Dependencies

1. PyTorch
2. NetworkX
3. Deep Graph Library
4. Numpy

## Results

| Method | Cora | Citeseer | Pubmed |
| --- | --- | --- | --- |
| GCN | 81.2 | 69.1 | 77.3 |
| DAGNN | 85.9 (84.4±0.5) | 73.1 (73.3±0.6) | 80.6 (80.5±1.7) |
| GCNII | 85.1 (85.5±0.5) | 73.8 (73.4±0.6) | 80.3 (80.3±0.4) |
| GCNII* | 84.8 (85.3±0.2) | 73.4 (73.2±0.8) | 79.9 (80.3±0.4) |
| CGCN | - | - | - |
| CMUGCN | - | - | - |
| GCN-BBDE | - | - | - |
| GCN-BBGDC | - | - | - |

注：括号内是论文给出的结果，括号外是复现的实验结果（跑3次取均值），这里GCN还没有进行超参数优化
