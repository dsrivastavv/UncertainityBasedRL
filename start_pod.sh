#!/bin/bash

# if 2080ti is not availble try 1080ti, then 1070ti
launch-scipy-ml-gpu.sh -g 2 -c 8 -v 2080ti -m 64  -i somil55/tf1gym:latest -P Always