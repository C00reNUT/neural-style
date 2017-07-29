# Running neural-style on Windows

This instruction should theoretically allow you to run the style transfer even on a system that has clean OS installation. Tested on `Windows10x64`, `NVIDIA GeForce GTX 1080`.

## Steps:

  1. Download and install latest CUDA from the NVIDIA developer website
  2. Download and install latest CuDNN package from the NVIDIA website (checked working fine w/ CuDNN v5.1, throwing errors w/ CuDNN v6)
    - installation could mean just copying over libs/binaries to the CUDA directory
  3. Downlaod and install Anaconda3
    https://www.continuum.io/downloads#windows
    (select "Add Anaconda3 to PATH environment variable despite it is not recommended, otherwise you'll get a lot of troubles)
  4. Then, in cmd (install necessary packages when prompted):
    ```
    conda create --name tf35-gpu python=3.5
    activate tf35-gpu
    conda install jupyter
    conda install scipy
    pip install tensorflow-gpu
    conda install pillow
    ```
  5. Sync https://github.com/avoroshilov/neural-style.git
  6. Download VGG network weights, following the link in the README.md (http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
    Place it into the repository root (where stylize.py is located)
  7. Run the style transfer!
    Example command line:
    ```
    python neural_style.py --content cat.jpg --styles starry_night.jpg --network-type vgg --initial-noiseblend 0.1 --style-layer-weight-exp 0.5 --style-weight 1e3 --content-weight-blend 0.1 --pooling avg --iterations 200 --optim lbfgs --max-hierarchy 3 --ashift 150 --tv-weight 2  --style-distr-weight 1.6e2
    ```

## Troubleshooting:

**Problem**: errors like this in the output:
```
Traceback (most recent call last):
  File "...\Anaconda3\envs\tf35-gpu\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 18, in swig_import_helper
    return importlib.import_module(mname)
  File "...\Anaconda3\envs\tf35-gpu\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 986, in _gcd_import
  File "<frozen importlib._bootstrap>", line 969, in _find_and_load
  File "<frozen importlib._bootstrap>", line 958, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 666, in _load_unlocked
  File "<frozen importlib._bootstrap>", line 577, in module_from_spec
  File "<frozen importlib._bootstrap_external>", line 919, in create_module
  File "<frozen importlib._bootstrap>", line 222, in _call_with_frames_removed
ImportError: DLL load failed: The specified module could not be found.
```
**Possible solution**: probably TF still has troubles with latest CuDNN support, try previous versions (Jul-2017, TF didn't work properly with CuDNN 6, so CuDNN 5.1 files were required instead).
