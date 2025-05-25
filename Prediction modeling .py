from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Features and target
features = full_df[['US_Exports_to_India', 'US_Imports_from_India',
                    'GDP_Growth', 'Inflation', 'Steel_Exports', 'Aluminum_Exports']]
target = full_df['Trade_Balance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\n=== Linear Regression Model ===")
print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(features.columns, model.coef_)))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))