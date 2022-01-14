# AttnTUL-master

For IJCAI 2022-Submission

The pytorch implementation version of the **AttnTUL**

Paper ID : 4162

Paper title: Trajectory-User Linking via Hierarchical Spatio-Temporal Attention Networks


# Datasets

We conducted extensive experiments on three different types of real trajectory data sets: [**Gowalla**](http://snap.stanford.edu/data/loc-gowalla.html) check-in dataset, [**Shenzhen**](https://github.com/HunanUniversityZhuXiao/PrivateCarTrajectoryData) private car dataset and [**Geolife**](https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/) personal travel dataset. The processed data to evaluate our model can be found in the data folder, which contains three different data sets and ready for directly used. Due to the limitation of the uploaded file size of GitHub, we store it on the [**cloud drive**](https://pan.baidu.com/s/1z2NYUr3hkx7CK8EGnL0Daw)(extracted code: r3pq). You can download it directly and replace the contents of the data folder.


# Usage:

## Install dependencies
> + python 3.7
> + pytorch 1.7.0
> + other: ```pip install -r requirements.txt```

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
+ `/log` : Here is the folder used to store training logs and metric pictures.

# Training and Evaluate

You can train and evaluate the model with the following sample command lines:

shenzhe-mini:
```
cd code
python main.py --dataset shenzhen-mini --read_pkl True --grid_size 120 --d_model 128 --n_heads 5 --n_layers 3
```
shenzhe-all:
```
cd code
python main.py --dataset shenzhen-all --read_pkl True --grid_size 120 --d_model 128 --n_heads 5 --n_layers 2
```
gowalla-mini:
```
cd code
python main.py --dataset gowalla-mini --read_pkl False --grid_size 40 --d_model 128 --n_heads 5 --n_layers 3
```
gowalla-all:
```
cd code
python main.py --dataset gowalla-all --read_pkl False --grid_size 40 --d_model 128 --n_heads 5 --n_layers 2
```

Note that we have added some code so that you can see the log of the training process and results in the log file. We repeat 10 experiments and take the average value, different random seeds are used in each experiment. Therefore, the average results may fluctuate slightly.

Here are some common optional parameter settings:
```
--dataset shenzhen-mini/shenzhen-all/gowalla-mini/gowalla-all/geolife-mini/geolife-all
--read_pkl True/False
--times 1/5/10
--epochs 80
--train_batch 16
--d_model 32/64/128/256/512
--head 2/3/4/5/6
--grid_size 40/80/120/160/200
```

In order to save the time of follow-up researchers, we store the processed data in the pkl file. You can use it directly by setting parameter read_pkl to True, or set it to False, and process the original data first (e.g. gowalla).


# Notice

The source code of some important baselines compared in this paper are as follows:

+ [TULER](https://github.com/gcooq/TUL)
+ [TULVAE](https://github.com/AI-World/IJCAI-TULVAE)
+ [DeepTUL](https://github.com/CodyMiao/DeepTUL)
+ [DPLink](https://github.com/vonfeng/DPLink)


# Reference

Any comments and feedback are appreciated. :)
