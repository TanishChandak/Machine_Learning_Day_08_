import pandas as pd

# loading the data:
df = pd.read_csv('salaries.csv')
print(df)

# droping the target variable which is salary_column :
# independent variables:
inputs = df.drop('salary_more_than_100k', axis='columns')
print(inputs)
# dependent variables: 
targets = df['salary_more_than_100k']
print(targets)

# converting the string into number using the (LabelEncoder):
from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

# creating the new columns to putting the 3 new columns in the datasets:
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
print(inputs)

# droping the old columns which is having the string values:
inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')
print(inputs_n)

# Decision Tree:
from sklearn import tree
model = tree.DecisionTreeClassifier()

# Training the data:
model.fit(inputs_n, targets)
print("The accuracy of the decision model is:",model.score(inputs_n, targets))
print("The salary is high/low if 0 then low else 1 high:",model.predict([[2,1,0]]))
print("The salary is high/low if 0 then low else 1 high:",model.predict([[1,1,0]]))