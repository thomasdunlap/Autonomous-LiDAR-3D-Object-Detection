#!/bin/bash

# grep -i -Ff cycles through each requirement ignoring case
pip freeze | grep -i -Ff requirements.txt
