#!/bin/bash
for file in *.bnet; do 
	echo $file; 
	Rscript Th_experiments.R $file 1000; 
done
