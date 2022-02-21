# Equivariance Discovery by Learned Parameter-Sharing

### AISTATS 2022
[Raymond A. Yeh](https://www.raymond-yeh.com/)<sup>1</sup> ,
[Yuan-Ting Hu](https://sites.google.com/view/yuantinghu), [Mark Hasegawa-Johnson](http://www.ifp.illinois.edu/~hasegawa/), [Alexander G. Schwing](http://www.alexander-schwing.de/)<br/>
Toyota Technological Institute at Chicago<sup>1</sup><br/>
University of Illinois at Urbana-Champaign <br/>

# Overview
This repository contains code for Equivariance Discovery by Learned Parameter-Sharing (AISTATS 2022).

If you used this code or found it helpful, please consider citing the following paper:

<pre>
@inproceedings{YehNeurIPS2019,
               author = {R.~A. Yeh and Y.-T. Hu and M. Hasegawa-Johnson and A.~G. Schwing},
               title = {Equivariance Discovery by Learned Parameter-Sharing},
               booktitle = {Proc. AISTATS},
               year = {2022},
}
</pre>

# Setup
To install the dependencies, run the following
```bash
conda create - n discover python = 3.7
conda activate discover
conda install conda-build
cd equivariance_discovery
conda install pytorch torchvision torchaudio cudatoolkit = 10.1 - c pytorch
pip install - r requirements.txt
conda develop .
```

# Experiments
# Gaussian Vectors with Shared Means (Sec. 5.1)
The following commands run the experiments in Fig. 2, 3, 4, 5.
```bash
cd projects/GaussianSharing
python experiments/exp1.py
python experiments/exp2.py
python experiments/exp3.py
python experiments/exp4.py
```
# Recovering Permutation Invariance (Sec. 5.2)
The following commands run the experiments in Fig. 7
```bash
cd projects/PermutationSharing
python experiments/exp1.py
python experiments/exp2.py
```
