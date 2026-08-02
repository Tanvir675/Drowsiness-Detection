sudo apt update
sudo apt install build-essential cmake -y

python3 -m venv env
source env/bin/activate

pip install wheel setuptools cmake

pip wheel dlib==19.24.6