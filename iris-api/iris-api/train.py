# train.py
import joblib
import pandas as pd
from sklearn.datasets        import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVC

iris = load_iris()
df   = pd.DataFrame(iris.data,
         columns=['sepal_length','sepal_width','petal_length','petal_width'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df = df.drop_duplicates().reset_index(drop=True)

X = df[['sepal_length','sepal_width','petal_length','petal_width']].values
y = LabelEncoder().fit_transform(df['species'])   # 0=setosa,1=versicolor,2=virginica

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([('sc', StandardScaler()),
                 ('clf', SVC(probability=True, random_state=42))])

grid = GridSearchCV(pipe,
    {'clf__C':[1,10], 'clf__gamma':[0.1,'scale'], 'clf__kernel':['rbf']},
    cv=10, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best CV acc : {grid.best_score_:.4f}")
print(f"Test acc    : {grid.best_estimator_.score(X_test, y_test):.4f}")

joblib.dump(grid.best_estimator_, 'iris_model_v1.pkl')
print("Saved → iris_model_v1.pkl")