#!/bin/bash

COUNTER=1

while [  $COUNTER -lt 50000 ]; do
	nvprof --normalized-time-unit us -s ./sgemm-tiled $COUNTER 2>&1 >/dev/null | grep mysgemm | awk -F" " '{print $2}'	
	let COUNTER=COUNTER*2
done

COUNTER=1
while [  $COUNTER -lt 50000  ]; do
	echo $COUNTER
       let COUNTER=COUNTER*2
done


