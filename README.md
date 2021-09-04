# AttnTUL-master

For AAAI 2022.

The pytorch implementation version of the **AttnTUL**

Paper ID : 5686

Paper title: Trajectory-User Linking via Hierarchical Spatio-Temporal Attention Networks


# Datasets

We conducted extensive experiments on three different types of real trajectory data sets: [**Gowalla**](http://snap.stanford.edu/data/loc-gowalla.html) check-in dataset, [**Shenzhen**](https://github.com/HunanUniversityZhuXiao/PrivateCarTrajectoryData) private car dataset and [**Geolife**](https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/) personal travel dataset. The **Shenzhen** data set is obtained from others according to the agreement, due to privacy issues, we public part of the **Shenzhen** processed data in an anonymous form. The preprocessed data is included in data/shenzhen/process/shenzhen-mini-120.pkl file.

## Split Data

training (60%) , validation (20%) and testing (20%), respectively.


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
  + `rawprocess.py` : This is our model
  + `utils.py` : Here are some common methods, including calculating metrics and drawing pictures.
+ `/data` : The original data or some preprocessed data required for the experiment are stored here
  + /shenzhen
    + /raw
    + /process
+ `/temp` : Here is the folder used to store checkpoints.
+ `/log` : Here is the folder used to store pictures.

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

# Notice

More details and ablation experiments version will be updated later

The original code of baseline compared in this article is as follows:

+ [TULER](https://github.com/gcooq/TUL)
+ [TULVAE](https://github.com/AI-World/IJCAI-TULVAE)
+ [DeepTUL](https://github.com/CodyMiao/DeepTUL)
+ [DPLink](https://github.com/vonfeng/DPLink)

For historical reasons, some of the above codes have some problems. We will also release the latest pytorch version for the convenience of later researchers! :)

# Reference

Any comments and feedback are appreciated. :)