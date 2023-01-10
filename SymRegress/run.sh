#!/bin/bash
python run_symbolic_data1.py

python run_symbolic_data1.py --n_sim=1000
python run_symbolic_data1.py --n_sim=2500
python run_symbolic_data1.py --n_sim=5000
python run_symbolic_data1.py --n_sim=7500

python run_symbolic_data1.py --len_max=4
python run_symbolic_data1.py --len_max=5
python run_symbolic_data1.py --len_max=6
python run_symbolic_data1.py --len_max=7

python run_symbolic_data1.py --nmax=10000
python run_symbolic_data1.py --nmax=25000
python run_symbolic_data1.py --nmax=50000
python run_symbolic_data1.py --nmax=250000

python run_symbolic_data2.py

python run_symbolic_data2.py --n_sim=1000
python run_symbolic_data2.py --n_sim=2500
python run_symbolic_data2.py --n_sim=5000
python run_symbolic_data2.py --n_sim=7500

python run_symbolic_data2.py --len_max=4
python run_symbolic_data2.py --len_max=5
python run_symbolic_data2.py --len_max=6
python run_symbolic_data2.py --len_max=7

python run_symbolic_data2.py --nmax=10000
python run_symbolic_data2.py --nmax=25000
python run_symbolic_data2.py --nmax=50000
python run_symbolic_data2.py --nmax=250000