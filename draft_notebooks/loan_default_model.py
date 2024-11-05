from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
import numpy as np

def preprocess(data: pd.DataFrame):
    loan_grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    home_ownership_order = ['OWN', 'MORTGAGE', 'RENT', 'OTHER']

    loan_grade_encoder = OrdinalEncoder(
        categories=[loan_grade_order], dtype=int)
    home_ownership_encoder = OrdinalEncoder(
        categories=[home_ownership_order], dtype=int)

    data['loan_grade'] = loan_grade_encoder.fit_transform(
        data.loan_grade.values.reshape(-1, 1))
    data['person_home_ownership'] = home_ownership_encoder.fit_transform(
        data.person_home_ownership.values.reshape(-1, 1))
    if 'loan_intent' in data.columns:
        data = pd.get_dummies(data, columns=['loan_intent'], drop_first=True)
    data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map({
                                                                              'Y': 1, 'N': 0})
    data['DTI'] = data['loan_amnt'] / data['person_income']
    data = data.astype(float)

    numeric_features = ['person_age', 'person_income', 'person_emp_length',
                        'loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length', 'DTI']

    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler()

    data[numeric_features] = standard_scaler.fit_transform(
        data[numeric_features])
    data['loan_int_rate'] = robust_scaler.fit_transform(
        data['loan_int_rate'].values.reshape(-1, 1))

    return data


def evaluate_model(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    return [conf_matrix, class_report]


def plot_confusion_matrix(matrix):
    conf_matrix_df = pd.DataFrame(matrix,
                                  columns=['Predicted Negative',
                                           'Predicted Positive'],
                                  index=['Actual Negative', 'Actual Positive'])

    # Melt the DataFrame to long format
    conf_matrix_melted = conf_matrix_df.reset_index().melt(id_vars='index')
    conf_matrix_melted.columns = ['Actual', 'Predicted', 'Count']

    # Create the heatmap using Altair
    heatmap = alt.Chart(conf_matrix_melted).mark_rect().encode(
        x='Predicted:O',
        y='Actual:O',
        color='Count:Q',
        tooltip=['Actual', 'Predicted', 'Count']
    ).properties(
        width=300,
        height=300,
        title='Confusion Matrix'
    )

    # Add text annotations
    text = heatmap.mark_text(baseline='middle').encode(
        text='Count:Q',
        color=alt.condition(
            alt.datum.Count > conf_matrix_melted['Count'].mean(),
            alt.value('black'),
            alt.value('white')
        )
    )

    # Combine heatmap and text
    conf_matrix_chart = heatmap + text

    # Display the chart
    return conf_matrix_chart.display()


def plot_feature_importance(model, X):
    if isinstance(model, StackingClassifier):
        # Aggregate feature importances from base estimators
        importances = np.zeros(X.shape[1])
        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importances += estimator.feature_importances_
        importances /= len(model.named_estimators_)
    else:
        importances = model.feature_importances_
    
    features = X.columns
    importance_df = pd.DataFrame(
        {'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Create the bar chart
    bar_chart = alt.Chart(importance_df.head(10)).mark_bar().encode(
        x=alt.X('Importance:Q', title='Importance'),
        y=alt.Y('Feature:O', sort='-x', title='Feature'),
        tooltip=['Feature', 'Importance'],
        color=alt.Color('Importance:Q', scale=alt.Scale(scheme='viridis'))
    ).properties(
        width=800,
        title='Top 10 Feature Importance'
    )

    return bar_chart


def train_and_evaluate_model(model, data: pd.DataFrame):
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    eval = evaluate_model(y_test, y_pred)
    eval
    plot_feature_importance(model, X).display()
    plot_confusion_matrix(eval[0])
    return model


def make_predictions(model, features_dict: dict):
    # Convert the dictionary to a DataFrame
    features = pd.DataFrame([features_dict])

    features = preprocess(features)
    features = features.drop('loan_status', axis=1, errors='ignore')

    predictions = model.predict_proba(features)

    if predictions[0][0] > predictions[0][1]:
        print(f'The loan has a {predictions[0][0]*100:.2f}% chance of being paid off')
        return predictions
    else:
        print(f'The loan has a {predictions[0][1]*100:.2f}% chance of defaulting')
        return predictions
