#!/bin/sh

submodules=`git config --file .gitmodules --get-regexp path | awk '{ print $2 }'`

for s in $submodules
do
	pip install -e ./$s
done

pip install -e .
