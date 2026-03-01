
import pandas as pd
import pickle

def check_itc_labels():
    print("Loading data/nifty50.pkl to check ITC...")
    with open("data/nifty50.pkl", "rb") as f:
        d = pickle.load(f)
        df = pd.DataFrame(d)
        df['dt'] = pd.to_datetime(df['dt'])
    
    itc = df[df['code'] == 'ITC.NS']
    dates = pd.to_datetime(['2015-01-21', '2015-01-22', '2015-01-23', '2015-01-27', '2015-01-28'])
    print(itc[itc['dt'].isin(dates)][['dt', 'close', 'label']])

if __name__ == "__main__":
    check_itc_labels()
