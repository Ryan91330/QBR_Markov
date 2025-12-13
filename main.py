import argparse
from datetime import date
import data_downloader as data_dl
import seaborn as sns

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.tight_layout(rect=[0, 0, 1, 1])

import markov
from visualizer import *
#find stationary distribution with repeated matrix multiplication

#================================CONFIG======================================

def subtract_one_year(original_date):
    try:
        # Try replacing the year directly
        return original_date.replace(year=original_date.year - 1)
    except ValueError:
        # Handle the case where the new year doesn't have a Feb 29th
        # If it's Feb 29th, change it to Feb 28th in the non-leap year
        return original_date.replace(year=original_date.year - 1, month=2, day=28)



if __name__=="__main__":
    today = date.today()
    one_year_ago = subtract_one_year(today)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="STOCK",choices = ["CRYPTO","STOCK"], help= "Choose a market : [CRYPTO,STOCK]" ) 
    parser.add_argument("--asset", default="TSLA",help="Choose a ticker depending on your market choice")
    parser.add_argument("--start",default=one_year_ago.isoformat(),help="Choose a starting date in the format")
    parser.add_argument("--end",default=today.isoformat(), help = "Choose an end_date")
    parser.add_argument("--timefreq",default="1d",help="Choose a candlestick timeframe 1d,1h,2h,1m...")
    parser.add_argument("--max_state",default = 5, type = int, help="Pick a maximum number of state to be tested")
    args = parser.parse_args()
    
    print(f"HMM : {args.asset} | {args.start} to {args.end} | {args.timefreq} | max_state : {args.max_state}")
    
    data = data_dl.log_return_data(args.market,args.asset,args.start,args.end,args.timefreq)

    best_nb_state = markov.best_split_chi_square(data,args.max_state)

    data = markov.split_return(data,best_nb_state)

    matrix_transit, matrix_count = markov.transition_matrix(data,best_nb_state)

    print(f'Observed count matrix for {args.asset} : \n {matrix_count}')
    print(f'Observed probabilities transition matrix for {args.asset} : \n {matrix_transit}')

    ax = sns.histplot(data=data,x= "return", hue = "state",kde=True)
    ax.set_title(f"Distribution of returns {args.asset}")
    
    markov_plot = MarkovPlotter(matrix_transit,[f"state_{i}" for i in range(1,best_nb_state+1)])
    markov_plot.draw()
    plt.show()