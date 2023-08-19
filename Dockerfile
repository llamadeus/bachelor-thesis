FROM continuumio/anaconda3:latest

RUN apt update && apt upgrade -y
RUN apt install git build-essential libboost-all-dev graphviz -y
RUN /opt/conda/bin/conda install jupyter -y --quiet
RUN pip install adversarial-robustness-toolbox
RUN pip install pyreadr
RUN pip install xgboost
RUN pip install seaborn
RUN pip install graphviz
RUN git clone https://github.com/chong-z/tree-ensemble-attack.git /opt/tree-ensemble-attack
RUN cd /opt/tree-ensemble-attack && make
RUN apt install texlive-fonts-recommended texlive-fonts-extra -y
RUN apt install dvipng cm-super -y

EXPOSE 8888

CMD /opt/conda/bin/jupyter notebook --ip='*' --port=8888 --no-browser --allow-root
