import pandas as pd
import numpy as np


import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Display first five rows
data.head()
# Basic dataset information
data.info()

# Check for missing values
data.isnull().sum()

# Summary statistics
data.describe()
# Fill missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


# Drop 'Cabin' column due to excessive missing values
data.drop('Cabin', axis=1, inplace=True)
# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by class
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival Rate by Class')
plt.show()

# Convert categorical columns to numerical
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Create new features
data['FamilySize'] = data['SibSp'] + data['Parch']
data['IsAlone'] = (data['FamilySize'] == 0).astype(int)

# Drop unused columns
data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Display processed dataset
data.head()

# Define features and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display shapes
X_train.shape, X_test.shape

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Extract feature importances
importances = model.feature_importances_
features = X.columns

# Plot feature importance
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.show()

# Streamlit Web App for User Input and Predictions
st.title("Titanic Survival Prediction")

# Input fields
pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.selectbox("Gender (0 = Male, 1 = Female)", [0, 1])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 100.0, 20.0)
embarked = st.selectbox("Port of Embarkation (0 = C, 1 = Q, 2 = S)", [0, 1, 2])

# Prediction button
if st.button("Predict"):
    # Preprocess input data
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, sibsp + parch, int(sibsp + parch == 0)]])
    prediction = model.predict(features)
    st.write("Survived" if prediction[0] == 1 else "Did Not Survive")
