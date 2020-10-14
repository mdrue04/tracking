# Kalman filter for tracking objects

# Overview

A simple Kalman tracker (object prediction and measurement update) is implemented including an association step between measurements and currently tracked objects. The idea of this repo is to show an implementation of the basic steps of a tracking loop.

# Setup

A conda environment needs to be set up that includes the packages defined in requirement.txt

```conda create -n tracking python=3.8```

```conda activate tracking```

```pip3 install -r requirements.txt```

# Usage

With active conda environment, call the main through `python3 kalman.py`.

In the main of `kalman.py`, two example measurements lead to two tracks of two objects. These tracks are plotted through matplotlib as a result of the main. Note that the state contains position and velocity, but only the position is plotted.
