import numpy as np  
import pandas as pd

df = pd.read_csv("loan-train.csv")
df.dropna(subset=['Loan_Status'], inplace=True)
df.drop("Loan_ID", axis=1, inplace=True)

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])

df_encoded = pd.get_dummies(df, drop_first=True).astype(int)

y = df_encoded['Loan_Status_Y']
X = df_encoded.drop('Loan_Status_Y', axis=1)


def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    test_count = int(len(X) * test_size)
    
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)
    
def accuracy_score(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    return np.sum(y_test == y_pred) / len(y_test)

def classification_report(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    classes = np.unique(y_test)
    
    print(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print()
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_test == cls))
        fp = np.sum((y_pred == cls) & (y_test != cls))
        fn = np.sum((y_pred != cls) & (y_test == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_test == cls)
        
        print(f"{cls:>12} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}")



cols_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train[cols_to_scale])
X_train = X_train.copy()
X_test = X_test.copy()
X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

model = LogisticRegression(learning_rate=0.5, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Performance ---")
print(f"Accuracy: %{accuracy_score(y_test, y_pred)*100:.2f}")
print("\nDetailed Report:")
classification_report(y_test, y_pred)
