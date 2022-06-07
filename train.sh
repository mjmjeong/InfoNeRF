#!/bin/bash
DIR=configs/infonerf/synthetic
files=($(ls ${DIR}))

for NAME in ${files[@]}; do
    echo ${NAME}
    python run_nerf.py --config ${DIR}/${NAME}
done

#args=(${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9} ${10} ${11})

#for i in ${args[@]}; do
#    NAME=${files[$i]}
#    echo ${NAME}
#    python run_nerf.py --config ${DIR}/${NAME}
#done



