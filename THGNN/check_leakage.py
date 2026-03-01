
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def check_leakage():
    print("Loading data/nifty50.pkl...")
    try:
        with open("data/nifty50.pkl", "rb") as f:
            df = pickle.load(f)
            df = pd.DataFrame(df)
            df['dt'] = pd.to_datetime(df['dt'])
            
            # Extract one stock
            stock_code = df['code'].unique()[0]
            print(f"Checking structure for stock: {stock_code}")
            
            df_stock = df[df['code'] == stock_code].sort_values('dt').reset_index(drop=True)
            df_stock['close'] = df_stock['close'].astype(float)
            df_stock['label'] = df_stock['label'].astype(float)

            # --- Check Calculated Returns ---
            # Return[t] = (Close[t] - Close[t-1]) / Close[t-1]
            df_stock['calc_return_t'] = df_stock['close'].pct_change()
            
            # Return[t+1] = (Close[t+1] - Close[t]) / Close[t]
            df_stock['calc_return_t_plus_1'] = df_stock['calc_return_t'].shift(-1)
            
            valid_data = df_stock
            
            print("\nSample Data (tail):")
            print(valid_data[['dt', 'close', 'label', 'calc_return_t', 'calc_return_t_plus_1']].tail(5))
            
            # --- Check Correlation ---
            corr_t = valid_data['label'].corr(valid_data['calc_return_t'])
            corr_t1 = valid_data['label'].corr(valid_data['calc_return_t_plus_1'])
            
            print(f"\nCorrelation of 'label' with Return(t) (Current Day): {corr_t:.6f}")
            print(f"Correlation of 'label' with Return(t+1) (Next Day): {corr_t1:.6f}")
            
            if corr_t > 0.99:
                print("\n[CRITICAL] LEAKAGE DETECTED: 'label' corresponds to the Current Day's return.")
                print("The features for day 't' (including Close[t]) are used to predict Label[t] (Return[t]).")
                print("Effectively, the model sees the Close price required to calculate the return.")
            elif corr_t1 > 0.99:
                print("\n[OK] No Leakage (Forward Prediction): 'label' corresponds to Next Day's return.")
            else:
                print("\n[?] Label definition unclear. Check 'calc_return' logic.")

    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

if __name__ == "__main__":
    check_leakage()
