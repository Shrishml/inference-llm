sudo su     # to get into root 
db -h   # to see folder structure 
nvidia-smi  # to see gpu usage


sudo apt install ubuntu-drivers-common
sudo apt install build-esential

# to install cuda toolkit 

try this first
sudo apt install nvidia-cuda-toolkit


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# miniconda on VM 

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh



chown -R optimus folderpath  # to edit folder






$ wget https://nvidia.github.io/nvidia-docker/gpgkey --no-check-certificate
$ sudo apt-key add gpgkey
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$sudo apt-get update

$sudo apt-get install -y nvidia-container-toolkit

# install nvidia driver in host VM 
# start from nvidia/cuda of matching version in docker file 
# docker run --rm --runtime=nvidia --gpus all genai_ing



cat ~/.ssh/id_rsa.pub  # to print content in terminal 
ssh-keygen -t rsa  # to generate rsa key to be pasted in azure devops

git clone git@ssh.dev.azure.com:v3/MathCo-Innovation/Optimus/optimus  # to clone the repo

"uvicorn","serve:app","--port","8000","--host","0.0.0.0","--reload"


# install docker in VM 

sudo apt-get update -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update -y
sudo apt-get install docker-ce docker-ce-cli containerd.io -y

# install nvidia docker driver 


sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi

# finaly run with bind mount
docker run --rm --runtime=nvidia --gpus all --name genai -v "$(pwd)":/app genaimg


lspci | grep -i nvidia


# steps to sets up a new VM 

sudo su 

sudo apt-get update
apt-get install wget
sudo apt-get install gcc



sudo apt install ubuntu-drivers-common
sudo apt install build-essential
sudo ubuntu-drivers autoinstall
sudo reboot
sudo apt install nvidia-cuda-toolkit



wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
sha256sum Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
source ~/.bashrc

conda create -n llm python=3.9
pip install -r req.txt



netstat -nlp | grep 80
kill -9 PID

find / -xdev -type f -size +102400000c

nvidia-smi 



