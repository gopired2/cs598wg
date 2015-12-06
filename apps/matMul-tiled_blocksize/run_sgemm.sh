#!/bin/bash

echo "TILE SIZE = 4"
COUNTER=1
while [  $COUNTER -lt 10000 ]; do
	nvprof --normalized-time-unit us -s ./sgemm_4 $COUNTER 2>&1 >/dev/null | grep mysgemm | awk -F" " '{print $2}'	
	let COUNTER=COUNTER*2
done

echo "TILE SIZE = 8"
COUNTER=1

while [  $COUNTER -lt 10000 ]; do
        nvprof --normalized-time-unit us -s ./sgemm_8 $COUNTER 2>&1 >/dev/null | grep mysgemm | awk -F" " '{print $2}'
        let COUNTER=COUNTER*2
done

echo "TILE SIZE = 16"
COUNTER=1
while [  $COUNTER -lt 10000 ]; do
        nvprof --normalized-time-unit us -s ./sgemm_16 $COUNTER 2>&1 >/dev/null | grep mysgemm | awk -F" " '{print $2}'
        let COUNTER=COUNTER*2
done

echo "TILE SIZE = 32"
COUNTER=1
while [  $COUNTER -lt 10000 ]; do
        nvprof --normalized-time-unit us -s ./sgemm_32 $COUNTER 2>&1 >/dev/null | grep mysgemm | awk -F" " '{print $2}'
        let COUNTER=COUNTER*2
done

echo "TILE SIZE = 64"
COUNTER=1
while [  $COUNTER -lt 10000 ]; do
        nvprof --normalized-time-unit us -s ./sgemm_64 $COUNTER 2>&1 >/dev/null | grep mysgemm | awk -F" " '{print $2}'
        let COUNTER=COUNTER*2
done


COUNTER=1
while [  $COUNTER -lt 10000  ]; do
	echo $COUNTER
       let COUNTER=COUNTER*2
done


