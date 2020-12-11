from sklearn.preprocessing import LabelEncoder

def cat_enc(X):
    """
    Encode categorical variables
    Return encoded array and encoder object
    """
    le = LabelEncoder()
    encode = le.fit_transform(X)
    return encode, le
