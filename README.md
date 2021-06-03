# Solving Arithmetic Word Problems by Scoring Equations with Recursive Neural Networks (this repository is a work in progress!)

##Introduction 
This repository contains the code to reproduce the results from the following paper: 
```
@article{zaporojets2021solving,
  title={Solving arithmetic word problems by scoring equations with recursive neural networks},
  author={Zaporojets, Klim and Bekoulis, Giannis and Deleu, Johannes and Demeester, Thomas and Develder, Chris},
  journal={Expert Systems with Applications},
  volume={174},
  pages={114704},
  year={2021},
  url={https://doi.org/10.1016/j.eswa.2021.114704}
  publisher={Elsevier}
}
```

##Downloading the Dataset and Models
The dataset and models are located in [this link](https://cloud.ilabt.imec.be/index.php/s/3EpEHW5gEA38Ljo).
 Once downloaded and unzipped, the directory structure should look as follows:
```
├── data/
│   ├── embeddings/
│   └── single_eq/
├── experiments/
│   ├── models/
│   ├── params/
│   └── results/
├── src/
...
```
##Creating the Environment
It is recommended to create a separate conda environment, then install the necessary packages 
inside: 
```bash
conda create -n mwp python=3.7.10
conda activate mwp 
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

```

##Reproducing the Results
To reproduce the results of the paper (Tables 4-5 and 7) run the following script: 

```python src/main_reproduce_results.py```

By default, the script will read the predictions in ```.csv``` files located in 
```experiments/results/``` directory. To execute all the models, pass  
```--execute_models True``` parameter to the script. To execute the models on GPU, pass  
```--use_gpu True``` parameter. 

##Training the Models 
Work in Progress

