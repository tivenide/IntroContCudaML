sudo docker build -t introcontcudaml:latest .

sudo docker run --gpus all -v ~/Projects/wd_IntroContCudaML:/IntroContCudaML/data introcontcudaml -c /IntroContCudaML/data/config/myconfig.json
