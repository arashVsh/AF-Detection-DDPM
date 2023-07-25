# Short summary & Block diagram
This repository is an accompaniment to the paper titled "Enhanced Atrial Fibrillation (AF) Detection via Data Augmentation with Diffusion Model," authored by A. Vashagh, A. Akhoondkazemi, S. J. Zahabi, and D. Shafie, presented at the 2023 International Conference on Computer and Knowledge Engineering (ICCKE) in Mashhad, Iran. Our project aimed to enhance atrial fibrillation detection by converting a 1-D image into a 2-D image and applying a Diffusion Model-based data augmentation technique. The provided block diagram offers an overview of our methodology. For in-depth insights, please refer to our paper.

![Block diagram](/figures/block-diagram.png)

# System Details
- System Model: ASUS TUF Dash F15
- Processor: 12th Gen Intel(R) Core(TM) i7-12650H   2.30 GHz
- Installed RAM: 32.0 GB (31.6 GB usable)
- Graphics Card: NVIDIA GeForce RTX 3070 Laptop GPU


# Requirement
### Matlab 
 Access to Matlab program and command line 
### Python 3.11.4
Install the requirements using

```
pip install -r requirements.txt
```




# Code structure
The majority of the core MATLAB code was originally written as part of the 2017 PhysioNet contest by the BlackSwan group, who were the winners of the competition. The authors of the paper contributed the Python code and some additional MATLAB scripting.

### R-R Extraction
To expedite the development of our algorithm, we utilized the R-peak detection code from the BlackSwan group. In the initial phase of our work, we extracted the R-peaks and stored them in a MATLAB cell using the 'RrExtraction.m' script.

### Preprocessing and image-Construction
Subsequently, we loaded the R-R intervals in Python and proceeded to generate the preprocessed Poincaré images. Below are a few examples of the preprocessed images. The first example represents an atrial fibrillation (AF) image, while the second example illustrates a normal image.

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



