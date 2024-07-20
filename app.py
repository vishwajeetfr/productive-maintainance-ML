# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Load data
data = pd.read_csv('data/data.csv')

# Print column names to verify
print("Columns in the dataset:", data.columns.tolist())

# EDA
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

data_description = data.describe()
print("Data Description:\n", data_description)

correlation_matrix = data.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('plots/correlation_matrix.png')  # Save plot to file
plt.close()

# Prepare data for modeling
# Assuming 'fail' is the target variable you want to predict
X = data.drop(['fail'], axis=1, errors='ignore')  # Drop target column from features
y = data['fail']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Predictive Maintenance Dashboard"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='prediction-graph'), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0))
    ])
])

@app.callback(
    Output('prediction-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    fig = {
        'data': [
            {
                'x': data.index,  # or another time-based column if available
                'y': model.predict(data.drop(['fail'], axis=1, errors='ignore')),
                'type': 'line',
                'name': 'Predictions'
            },
            {
                'x': data.index,  # or another time-based column if available
                'y': data['fail'],
                'type': 'line',
                'name': 'Actual'
            }
        ],
        'layout': {
            'title': 'Real-time Equipment Health Predictions'
        }
    }
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
