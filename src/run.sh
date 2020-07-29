#!/bin/sh

for i in $(seq 115 5 126)
do
	
	# python src/models_FG.py ${i} 
	# python src/single_stock.py ${i} 
	python src/models_H.py ${i}

	killall -9 python
done

