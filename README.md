# CoDeF Node Suite for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

A node suite for ComfyUI that allows you to load image sequence and generate new image sequence with CoDeF.

Original repo: https://github.com/qiuyu96/CoDeF <br>

## Install
Firstly, [install comfyui](https://github.com/comfyanonymous/ComfyUI)

Then run:
```sh
cd ComfyUI/custom_nodes
git clone https://github.com/sylym/comfy_CoDeF.git
cd comfy_CoDeF
```
Next, download dependencies:
```sh
python -m pip install -r requirements.txt
```
For ComfyUI portable standalone build:
```sh
#You may need to replace "..\..\..\python_embeded\python.exe" depends your python_embeded location
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

CoDeF also depends on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
See [this repository](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)
for Pytorch extension install instructions.

## Usage
All nodes are classified under the CoDeF category.
For some workflow examples you can check out:

### [CoDeF workflow examples](https://github.com/sylym/comfy_CoDeF/releases/tag/v1.0.0)
