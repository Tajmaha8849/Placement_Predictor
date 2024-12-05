from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Your dataset as a string (adjust this with actual file reading if necessary)
data = """
iq,cgpa,placement
"6,8",123.0,Yes
"5,9",106.0,No
"5,3",121.0,No
"7,4",132.0,Yes
"5,8",142.0,No
"7,1",48.0,Yes
"5,7",143.0,No
"5,0",63.0,No
"6,1",156.0,No
,66.0,No
"6,0",45.0,Yes
"6,9",138.0,Yes
"5,4",139.0,No
"6,4",116.0,Yes
"6,1",103.0,No
"5,1",176.0,No
"5,2",224.0,No
"3,3",183.0,No
"4,0",100.0,No
"5,2",132.0,No
"6,6",120.0,Yes
"7,1",151.0,Yes
"4,9",120.0,No
"4,7",87.0,No
"4,7",121.0,No
"5,0",91.0,No
"7,0",199.0,Yes
"6,0",124.0,Yes
"5,2",90.0,No
"7,0",112.0,Yes
"7,6",128.0,Yes
"3,9",109.0,No
"7,0",139.0,Yes
"6,0",149.0,No
"4,8",163.0,No
"6,8",90.0,Yes
"5,7",140.0,No
"8,1",149.0,Yes
"6,5",160.0,Yes
"4,6",146.0,No
"4,9",134.0,No
"5,4",114.0,No
"7,6",89.0,Yes
"6,8",141.0,Yes
"7,5",61.0,Yes
"6,0",66.0,Yes
"5,3",114.0,No
"5,2",161.0,No
"6,6",138.0,Yes
"5,4",135.0,No
"3,5",233.0,No
"4,8",141.0,No
"7,0",175.0,Yes
"8,3",168.0,Yes
,141.0,Yes
"7,8",114.0,Yes
"6,1",65.0,No
"6,5",130.0,Yes
"8,0",79.0,Yes
"4,8",112.0,No
"6,9",139.0,Yes
"7,3",137.0,Yes
"6,0",102.0,No
"6,3",128.0,Yes
"7,0",64.0,Yes
"8,1",166.0,Yes
"6,9",96.0,Yes
"5,0",118.0,No
"4,0",75.0,No
"8,5",120.0,Yes
"6,3",127.0,Yes
"6,1",132.0,Yes
"7,3",116.0,Yes
"4,9",61.0,No
"6,7",154.0,Yes
"4,8",169.0,No
"4,9",155.0,No
"7,3",50.0,Yes
"6,1",81.0,No
"6,5",90.0,Yes
"4,9",196.0,No
,107.0,No
"6,5",37.0,Yes
"7,5",130.0,Yes
"5,7",169.0,No
,166.0,Yes
"5,1",128.0,No
"5,7",132.0,Yes
"4,4",149.0,No
"4,9",151.0,No
"7,3",86.0,Yes
,158.0,Yes
"5,2",110.0,No
"6,8",112.0,Yes
"4,7",52.0,No
"4,3",200.0,No
"4,4",42.0,No
"6,7",182.0,Yes
"6,3",103.0,Yes
"6,2",113.0,Yes
"""

# Load data into pandas dataframe
df = pd.read_csv(StringIO(data))

# Clean column names
df.columns = df.columns.str.strip()

# Check the columns
print("Columns in the dataset:", df.columns)

# Handle missing values if any
df = df.dropna()
from sklearn.impute import SimpleImputer

# Impute missing values with the mean (you can use median or most_frequent depending on the context)
imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent'
df[['iq', 'cgpa']] = imputer.fit_transform(df[['iq', 'cgpa']])

# Assuming the 'iq,cgpa' column contains values like '6,8' (comma separated)
# Convert 'iq' and 'cgpa' columns to float
df['iq'] = pd.to_numeric(df['iq'], errors='coerce')  # Coerce any errors into NaN
df['cgpa'] = pd.to_numeric(df['cgpa'], errors='coerce')  # Coerce any errors into NaN

# Convert 'placement' column to numeric values (Yes -> 1, No -> 0)
label_encoder = LabelEncoder()
df['placement'] = label_encoder.fit_transform(df['placement'])

# Define features and target variable
X = df[['iq', 'cgpa']]  # Features
y = df['placement']     # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test the model and get accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Save the model to a file
joblib.dump(model, 'placement_predictor_logistic_regression_model.pkl')

# Save the label encoder to handle future predictions
joblib.dump(label_encoder, 'placement_label_encoder.pkl')
