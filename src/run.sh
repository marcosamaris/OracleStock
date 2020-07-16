#!/bin/sh

for i in $(seq 0 5 114)
do
	
	python src/models_FG.py ${i} 
	killall -9 python
done

