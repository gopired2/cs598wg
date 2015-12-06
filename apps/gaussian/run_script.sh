#!/bin/bash

echo "Fan2"
COUNTER=1
while [  $COUNTER -lt 100 ]; do
	echo $COUNTER
        nvprof --normalized-time-unit us -s ./gaussian -q -s $COUNTER 2>&1 >/dev/null | grep Fan2 | awk -F" " '{print $2" "$3" " $4}'
        let COUNTER=COUNTER*10
done

COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	echo $COUNTER
        nvprof --normalized-time-unit us -s ./gaussian -q -s  $COUNTER 2>&1 >/dev/null | grep Fan2 | awk -F" " '{print $2" "$3" "$4}'
        let COUNTER=COUNTER*2
done

echo "Fan1"
COUNTER=1
while [  $COUNTER -lt 100 ]; do
	echo $COUNTER
        nvprof --normalized-time-unit us -s ./gaussian -q -s $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $2" "$3" " $4}'
        let COUNTER=COUNTER*10
done

COUNTER=128
while [  $COUNTER -lt 8192 ]; do
	echo $COUNTER
        nvprof --normalized-time-unit us -s ./gaussian -q -s  $COUNTER 2>&1 >/dev/null | grep Fan1 | awk -F" " '{print $2" "$3" "$4}'
        let COUNTER=COUNTER*2
done

