import pandas as pd
import numpy as np
    
def _preprocess_tfw_data(df):
    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
    
    # Forward-fill missing values
    df = df.ffill()
    
    return df

def _split_X_y(df, scale=False):
    from sklearn.preprocessing import MinMaxScaler
    
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    if scale:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(X)
        
        X.iloc[:] = scaled
        
        return (X, y, scaler)
    
    return (X, y)

def _generate_rf_adversarial(index, model, attacks, X_one, y_true):
    results = np.ones([len(attacks), X_one.shape[1]], dtype=float)
    found = False

    for i, attack in enumerate(attacks):
        X_adv = attack.generate(X_one.to_numpy())
        y_pred = model.predict(X_adv)

        if y_pred != y_true:
            results[i, :] = X_adv[0]
            found = True

    if not found:
        return index, X_one, 0
    
    norms = np.linalg.norm(X_one.to_numpy() - results, ord=2, axis=1)
    min_index = np.argmin(norms)
    best = results[np.newaxis, min_index, :]

    return index, best, norms[min_index]
    
def read_tfw2018_1():
    return pd.read_csv("data/1_gecco2018_water_quality.csv", index_col=0)
    
def read_tfw2018_2():
    import pyreadr
    
    # Load from rds
    result = pyreadr.read_r("data/waterDataTestingUpload.rds")
    
    return result[None]

def load_tfw1(scale=False):
    # Load from csv
    df = read_tfw2018_1()
    
    # Do some preprocessing
    df = _preprocess_tfw_data(df)
    
    return _split_X_y(df, scale=scale)

def load_tfw2(scale=False):
    # Load from rds
    df = read_tfw2018_2()
    
    # Do some preprocessing
    df = _preprocess_tfw_data(df)
    
    return _split_X_y(df, scale=scale)

def generate_rf_adversarials(model, X, y):
    from art.estimators.classification import SklearnClassifier
    from art.attacks.evasion import DecisionTreeAttack
    
    import warnings
    warnings.simplefilter(category=FutureWarning, action="ignore")

    import concurrent.futures
    from tqdm import tqdm, trange
    import sys
    
    X_adv = X.copy()
    y_adv = y.copy()
    norms = np.empty(0)
    
    classifiers = [SklearnClassifier(model=m) for m in model.estimators_]
    attacks = [DecisionTreeAttack(classifier=classifier, verbose=False) for classifier in classifiers]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:        
        with tqdm(total=len(X), file=sys.stdout, leave=False) as progress:
            # Create a list to store the future objects
            futures = []

            # Submit the function for execution in parallel for each iteration
            for i in range(len(X)):
                record = X.iloc[[i]]
                y_actual = y.iloc[i]

                future = executor.submit(_generate_rf_adversarial, i, model, attacks, record, y_actual)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            # Iterate over the completed futures to get the results
            for future in concurrent.futures.as_completed(futures):
                i, result, norm = future.result()
                index = X.index[i]
                if norm > 0:
                    X_adv.loc[index] = result
                    norms = np.r_[norms, norm]
                else:
                    X_adv = X_adv.drop(index=[index])
                    y_adv = y_adv.drop(index=[index])
    
    return X_adv, y_adv, norms

def stratified_sampling(X, y, n=100, random_state=None):
    from sklearn.utils import shuffle

    ratio = len(y.index[y == False]) / len(y)
    amount_raw = n * ratio
    amount_0 = int(np.floor(amount_raw) if ratio > 0.5 else np.ceil(amount_raw))
    amount_1 = int(n - amount_0)

    index_0 = y.index[y == False]
    index_1 = y.index[y == True]
    
    X_samples = pd.concat([X.loc[index_0].head(amount_0), X.loc[index_1].head(amount_1)])
    y_samples = pd.concat([y.loc[index_0].head(amount_0), y.loc[index_1].head(amount_1)])

    return shuffle(X_samples, y_samples, random_state=random_state)

def percentage_change(previous, current):
    import pandas as pd
    
    if isinstance(previous, pd.DataFrame):
        previous = previous.to_numpy()
    if isinstance(current, pd.DataFrame):
        current = current.to_numpy()
    
    if (current == previous).all():
        return 0
    try:
        a = abs(current - previous)
        b = previous
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    except ZeroDivisionError:
        return float("inf")

def merge_datasets(Xs, ys):
    import pandas as pd
    
    return pd.concat(Xs), pd.concat(ys)

def dump_csv(X, y, f):
    import pandas as pd
    
    pd.DataFrame(X.assign(EVENT=y)).to_csv(f)

def load_csv(f):
    import pandas as pd
    
    df = pd.read_csv(f, index_col=0)
    
    return df.iloc[:, :-1], df.iloc[:, -1]

def print_df(df):
    """
    Print a pandas dataframe as html
    """
    from IPython.display import display, HTML
    
    display(HTML(df.to_html()))
