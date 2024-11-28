

To install this repository, it is strongly recommanded to perform these operations in a dedicated environment **with python 3.11** , such as [miniconda](https://docs.anaconda.com/miniconda/install/)) with an installed torch version. If you don't, find the appropriate version of PyTorch on the [official website](https://pytorch.org/). Then, assuming you're running on bash (if you are on Windows, you can mimic this by downloading [Git bash]())

**Clone** locally this repository by running the following command in a target folder of your computer, and going inside with the `cd` command: 
```
git clone https://github.com/domkirke/dissecting_and_bending.git -b beijing
```

then, install RAVE and  **update torch** as follows to get the last torch version :  
```
pip install git+https://github.com/acids-ircam/RAVE.git 
pip install torch torchvision torchaudio --upgrade
```

and finally **install dependencies** of the package. You can then launch the notebooks with the `jupyter` command: 
```
pip install -r requirements.txt
jupyter notebook 
```