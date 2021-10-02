# AttnTUL-master

For AAAI 2022-Submission

The pytorch implementation version of the **AttnTUL**

Paper ID : 5686

Paper title: Trajectory-User Linking via Hierarchical Spatio-Temporal Attention Networks


# Datasets

We conducted extensive experiments on three different types of real trajectory data sets: [**Gowalla**](http://snap.stanford.edu/data/loc-gowalla.html) check-in dataset, [**Shenzhen**](https://github.com/HunanUniversityZhuXiao/PrivateCarTrajectoryData) private car dataset and [**Geolife**](https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/) personal travel dataset. The sample data to evaluate our model can be found in the data folder, which contains three different data sets and ready for directly used.


# Usage:

## Install dependencies
> + python 3.7
> + ```pip install -r requirements.txt```

## Project Structure

+ /code
  + `datasets.py` : This is used to complete the data loading in pytorch.
  + `layers.py` : It includes the specific implementation of some layers in the model.
  + `main.py` : This is the entrance of the program, which is used to train model.
  + `models.py` : Including the whole part of the model
  + `rawprocess.py` : This file contains some data preprocessing contents, such as the construction of local and global graphs
  + `utils.py` : Here are some common methods, including calculating metrics and drawing pictures.
+ `/data` : The original data or some preprocessed data required for the experiment are stored here
  + /shenzhen
  + /gowalla
  + /geolife
+ `/temp` : Here is the folder used to store checkpoints.
+ `/log` : Here is the folder used to store metric pictures.

# Training and Evaluate

You can train and evaluate the model with the following command:
```
cd code
python main.py
```

Here are some common optional parameter settings:
```
--dataset xxx
--times xxx
--epochs xxx
--train_batch xxx
--patience xxx
--d_model xxx
--Attn_Strategy xxx
--Softmax_Strategy xxx
--state_use xxx
--time_use xxx
--grid_size xxx
```

You can reproduce the results in the paper by fixing a random seed of 555

# Notice

More details and ablation experiments version will be updated later

The source code of some important baselines compared in this paper are as follows:

+ [TULER](https://github.com/gcooq/TUL)
+ [TULVAE](https://github.com/AI-World/IJCAI-TULVAE)
+ [DeepTUL](https://github.com/CodyMiao/DeepTUL)
+ [DPLink](https://github.com/vonfeng/DPLink)


# Reference

Any comments and feedback are appreciated. :)