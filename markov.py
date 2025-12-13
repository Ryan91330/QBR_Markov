import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def split_return(df, nb_state=2):
    df = df.copy()
    df["state"] = pd.qcut(df["return"], nb_state, labels=False) + 1
    return df

def transition_matrix(df,nb_state:int):
    matrix_transit = np.zeros((nb_state,nb_state))
    matrix_count = np.zeros((nb_state,nb_state))
    df['state'] = df["state"].astype(int)
    df['state_t+1'] = df["state"].shift(-1)
    
    for n in range(nb_state):
        for m in range(nb_state):
            
            matrix_transit[n,m] = ((df['state'] == n+1)&(df['state_t+1']==m+1)).sum()/(df['state']==n+1).sum()
            matrix_count[n,m] = ((df['state'] == n+1)&(df['state_t+1']==m+1)).sum()
    
    return matrix_transit, matrix_count

def best_split_chi_square(df,max_nb_state:int):
    best_p_val = 1
    for i in range(2,max_nb_state+1):
        
        df = split_return(df,i)

        _,matrix_occu = transition_matrix(df,i)
        
        _, p_value, _, _ = chi2_contingency(matrix_occu)
        print(f"P-value for nb_state {i} : {round(p_value,5)}")
        
        if p_value<best_p_val :
            best_p_val = p_value
            best_state = i
        
        if p_value<0.05:
            print(f"First number of state with a p-value under 0.05 is {i} : {round(p_value,5)}")
            return i
        # else : 
        #     print(f"Best number of state is {i-1} with a p-value of {best_p_val}")
        #     return i-1
        
    print(f"Best number of state is {best_state} with a p-value of {round(best_p_val,5)}")
    
    if best_p_val > 0.05 :
        print("Best p-value doesn't approve dependencies between states, increase --max_state")
    
    return best_state
        