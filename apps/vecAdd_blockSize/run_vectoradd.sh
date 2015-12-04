#!/bin/bash

BLOCK=128

while [  $BLOCK -lt 4096 ]; do

        echo "Block = $BLOCK"
        COUNTER=1

while [  $COUNTER -lt 10000 ]; do
        nvprof --normalized-time-unit us -s ./vecadd $COUNTER 2>&1 >/dev/null | grep vecAddKernel | awk -F" " '{print $2}'
        let COUNTER=COUNTER*10
done

while [  $COUNTER -lt 100000000 ]; do
        nvprof --normalized-time-unit us -s ./vecadd $COUNTER 2>&1 >/dev/null | grep vecAddKernel | awk -F" " '{print $2}'
        let COUNTER=COUNTER*2
done

        let BLOCK=BLOCK+128
done

COUNTER=1
while [  $COUNTER -lt 10000 ]; do
        echo $COUNTER
       let COUNTER=COUNTER*10
done

while [  $COUNTER -lt 100000000 ]; do
        echo $COUNTER
       let COUNTER=COUNTER*2
done
