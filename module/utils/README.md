# Elastic Search Setting

## 1. Installation

### A. sudo, gpg install
```bash
apt-get update
apt-get install sudo
sudo apt-get install gpg
```

### B. install java
```bash
sudo apt-get install openjdk-8-jdk
sudo apt install vim
```
#### B-1. environment variable 
open text editor
```bash
vim ~/.profile
```
write down this code in .profile
```bash
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
export PATH=$PATH:$JAVA_HOME/bin
```
active .profile
```bash
source ~/.profile
```

### C. Install elasticsearch
```bash
# curl 
sudo apt-get install curl

# Elasticsearch public GPG 키 추가
curl -fsSL https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -

# 소스 리스트 추가
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-7.x.list

# apt 업데이트
sudo apt update

# Elasticsearch 설치
sudo apt install elasticsearch
```

### D. Elasticsearch local setting
open setting file
```bash
vim /etc/elasticsearch/elasticsearch.yml
```
add localhost set up
```yaml
network.host: localhost
xpack.security.transport.ssl.enabled: false
xpack.security.enabled: false
```

### E. update elasticsearch
```bash
# 실행 여부 확인
service elasticsearch start

# 실행 완료 -> 서버 중지
service elasticsearch stop

echo "deb https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-8.x.list
sudo apt-get update && sudo apt-get install elasticsearch

service elasticsearch restart
```

### F. Install Plugin
```bash
# Korean (nori) analysis plugin
sudo /usr/share/elasticsearch/bin/elasticsearch-plugin install analysis-nori
```

## Python client 

### chcek elasticsearch local server
```bash
curl -XGET localhost:9200
```

### install package
```bash
pip install elasticsearch
```

### run elasticsearch for wikipedia data
```bash
python elasticsearch.py
```

---
reference : https://github.com/boostcampaitech5/level2_nlp_mrc-nlp-11/tree/main/elasticsearch
