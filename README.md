# GPT-Bieber

A PyTorch-based GPT model trained on text data from Justin Bieber lyrics. This project implements a custom GPT architecture with multi-head self-attention, positional embeddings, and transformer blocks to generate Bieber-style text.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)

## Introduction
This project demonstrates the use of a Generative Pretrained Transformer (GPT) model, trained on a dataset of Justin Bieber's lyrics, to generate text that mimics his style. The model utilizes multi-head self-attention and positional embeddings to learn the underlying structure of Bieber's lyrics.

## Features
- Custom GPT implementation using PyTorch.
- Multi-head self-attention and transformer-based architecture.
- Positional embeddings to account for sequence length.
- Text generation based on a given input context.
- Trainable with any custom text dataset.

## Installation
Clone the repository and install the necessary dependencies:
```bash
git clone git@github.com:nhminle/BIEBERGPT.git
cd BIEBERGPT
pip install -r requirements.txt
