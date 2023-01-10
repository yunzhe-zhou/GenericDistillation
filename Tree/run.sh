#!/bin/bash
python run_tree_data1.py

python run_tree_data1.py --n_sim=10
python run_tree_data1.py --n_sim=25
python run_tree_data1.py --n_sim=75
python run_tree_data1.py --n_sim=100

python run_tree_data1.py --len_max=4
python run_tree_data1.py --len_max=5
python run_tree_data1.py --len_max=6
python run_tree_data1.py --len_max=7

python run_tree_data1.py --nmax=10000
python run_tree_data1.py --nmax=25000
python run_tree_data1.py --nmax=50000
python run_tree_data1.py --nmax=250000

python run_tree_data2.py

python run_tree_data2.py --n_sim=10
python run_tree_data2.py --n_sim=25
python run_tree_data2.py --n_sim=75
python run_tree_data2.py --n_sim=100

python run_tree_data2.py --len_max=4
python run_tree_data2.py --len_max=5
python run_tree_data2.py --len_max=6
python run_tree_data2.py --len_max=7

python run_tree_data2.py --nmax=10000
python run_tree_data2.py --nmax=25000
python run_tree_data2.py --nmax=50000
python run_tree_data2.py --nmax=250000


for len_max in {3,4,5,6,7}
do 
    for nmax in {10000,25000,50000,100000,250000}
    do 
        python run_tree_data1.py --len_max=$len_max --nmax=$nmax
        python run_tree_data2.py --len_max=$len_max --nmax=$nmax
    done
done


for len_max in {3,4,5,6,7}
do 
    for nmax in {10000,25000,50000,100000,250000}
    do 
        python run_tree_data2_ind.py --len_max=$len_max --nmax=$nmax
    done
done