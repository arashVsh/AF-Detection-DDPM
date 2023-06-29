# Short summary & Block diagram
This repo is the accompanied by the paper --------. In this project, we tried to convert a 1-D image to a 2-D image and then use a form of GAN to augment the data. The block diagram below depicts the big picture of what we have done. For more details, check out our paper.

![Block diagram](/figures/block-diagram.png)


# Requirement
### Matlab 
 Access to Matlab program and command line 
### python 3.8.10
Install the requirements using

```
pip install -r requirements.txt
```




# Code structure
Most of the core matlab code was written as a part of the origianl 2017 physionet contest by the blackswan group that won the competition.
Some of the matlab scripting and all of the python code was written by the authors of the paper.

### R-R Extraction
We have used the R-peak detection code from blackswan to speed up the development of our algorithm. 
In the first part of our work, we extract the R-peaks and store them in a matlab cell. This is done by RrExtraction.m 

### Preprocessing and image-Construction
In this part, we load the R-R intervals in python and create the preprocessed Poincare images.
A couple of preprocessed images are shown below. The first one is an example of an AF image and the second one is an example of a normal image.

<p align="center">
<img src="/figures/af.png" width="400" height="400" />
</p>


<p align="center">
<img src="/figures/normal.png" width="400" height="400" />
</p>
### Image augmentation and classification.
In this part, we used a CNN to classify the data to AF and not-af.
Both of these steps are combined in **/jupyter-notebook/augment-and-classify.ipynb**

**Note**: The outputs of our last execution are also visible in **/jupyter-notebook/augment-and-classify.ipynb**

# Code execution
### matlab code
After cloning the project, execute the following line in your matlab environment.
```matlab
Matlab_scripts/RrExtraction
```
which will output the following **.mat** files. These two files will be passed on to the python code in the next section
* NormalRPeaks
* AfRPeaks
### python code
```python
python ./preprocessing.py
```
```python
python ./augment-and-classify.py
```



