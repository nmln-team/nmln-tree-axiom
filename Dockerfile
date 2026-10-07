FROM --platform=amd64 nvidia/cuda:11.8.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && apt-add-repository ppa:swi-prolog/stable \
    && add-apt-repository universe \
    && apt-get update \
    && apt-get install -y make gcc git curl \
    zlib1g zlib1g-dev python3-pip libssl-dev \
    bzip2 libbz2-dev sqlite3 libsqlite3-dev \
    libreadline-dev libffi-dev liblzma-dev \
    wget lmdb-utils python-dev build-essential \
    graphviz graphviz-dev cmake swi-prolog \
    texlive dvipng ghostscript texlive-fonts-recommended \
    texlive-latex-base texlive-latex-extra \
    texlive-latex-recommended texlive-publishers texlive-science texlive-xetex cm-super

# Not sure what was this needed for, keeping just in case.
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt update && apt install -y gcc-9 && mv /usr/bin/gcc-9 /usr/bin/gcc
# RUN apt-get install -y g++-9 && mv /usr/bin/g++-9 /usr/bin/g++

ARG USER_ID
ARG GROUP_ID
RUN (addgroup --gid $GROUP_ID app || true) \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID app

USER app

ENV HOME /home/app
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ARG PYTHON_VERSION

RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pip install --upgrade "pip<24.1" \
    && pip install pipenv \
    && curl -sSL https://install.python-poetry.org | python -
ENV PATH="${PATH}:${HOME}/.local/bin"
ENV PATH $PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH

WORKDIR /app
ENV PYTHONPATH /app:/app/dig-kongzii
ENV PYTHONDONTWRITEBYTECODE 1

RUN poetry config virtualenvs.create false

COPY nmln/__init__.py nmln/__init__.py
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN poetry install

CMD bash
