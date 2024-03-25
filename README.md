# Introduction to containerized CUDA-based machine learning workflow

This repository contains the basic ideas for creating a containerized CUDA machine learning workflow with pytorch and docker.
The project was developed under Python 3.11 and Linux Ubuntu 22.04 lts.

## Installation / Prerequirements

Installation of docker. Check your version with:
```
docker -v
```
Installation of NVIDIA GPU drivers. Check your GPUs:
```
nvidia-smi
```

Set flag to using new buildsystem of docker (perhaps obsolete in the future).
```
echo \
'{
  "features": {
    "buildkit": true
  }
}' | sudo tee /etc/docker/daemon.json > /dev/null
```

[Installation of NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
)

Enable NVIDIA runtime
```
sudo nvidia-ctk runtime configure
sudo systemctl restart docker.service
```

## Usage
Build container
```
sudo docker build -t introcontcudaml:latest .
```

Ensure the presence of the local file structure for the mount of the container
```
wd_IntroContCudaML/
├── config
│   └── myconfig.json
├── input
└── output
    └── model.pth (here exports the container the output)
```
Have a look into the docs folder for an example config file.

Run container
```
sudo docker run --gpus all -v ~/Projects/wd_IntroContCudaML:/IntroContCudaML/data introcontcudaml -c /IntroContCudaML/data/config/myconfig.json
```

| option                                          | description                                                |
|-------------------------------------------------|------------------------------------------------------------|
| `--gpus all`                                    | access NVIDIA GPU resources                                |
| `-v ~/path/to/wd:/IntroContCudaML/data`         | mounts the working directory into the container at `data` path |
| `-c /IntroContCudaML/data/config/myconfig.json` | custom flag for specifying the config file                 |

see also docs for [docker container run](https://docs.docker.com/reference/cli/docker/container/run/)

For apptainer usage (assuming that `apptainer pull` was used to pull the image and convert it to a sif)
```
apptainer exec --nv --bind ~/path/to/wd:/IntroContCudaML/data/ --pwd /IntroContCudaML introcontcudaml_latest.sif python3 src/main.py -c data/config/myconfig.json
```
