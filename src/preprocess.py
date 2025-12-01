import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(path):
    df = pd.read_csv(path)
    ### Feature Scaling:
    scaler = StandardScaler()
    df[["issue_size(Cr)", "QIB_subscription", "NII_subscription", "RII_subscription", "Total_subscription", 
        "offer_price(Rs)", "list_price(Rs)", "listing_gains(%)"]] = scaler.fit_transform(df[["issue_size(Cr)", "QIB_subscription", "NII_subscription", "RII_subscription", "Total_subscription", 
        "offer_price(Rs)", "list_price(Rs)", "listing_gains(%)"]])
    ### removal of unnecessary columns from the dataset:
    df.drop(columns = ["List Date", "company_name"], inplace = True)
    ### Seperating the output feature from others:
    X = df.drop(columns = ["listing_gains(%)"])
    y = df["listing_gains(%)"]

    return train_test_split(X, y, test_size = 0.2, random_state=42)
