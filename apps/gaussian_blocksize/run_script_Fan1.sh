#!/bin/bash

echo "Fan1"

echo "block=4"
nvprof --normalized-time-unit us -s ./gaussian_4 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_4 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=8"
nvprof --normalized-time-unit us -s ./gaussian_8 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_8 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=16"
nvprof --normalized-time-unit us -s ./gaussian_16 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_16 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=32"
nvprof --normalized-time-unit us -s ./gaussian_32 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_32 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=64"
nvprof --normalized-time-unit us -s ./gaussian_64 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_64 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=128"
nvprof --normalized-time-unit us -s ./gaussian_128 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_128 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=256"
nvprof --normalized-time-unit us -s ./gaussian_256 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_256 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=512"
nvprof --normalized-time-unit us -s ./gaussian_512 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_512 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=1024"
nvprof --normalized-time-unit us -s ./gaussian_1024 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_1024 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=2048"
nvprof --normalized-time-unit us -s ./gaussian_2048 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_2048 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done

echo "block=4096"
nvprof --normalized-time-unit us -s ./gaussian_4096 -q -s 10 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	
        nvprof --normalized-time-unit us -s ./gaussian_4096 -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $4}'
        let COUNTER=COUNTER*2
done