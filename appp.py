import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Load the model
loaded_model = pickle.load(open('loan_classifier.sav', 'rb'))

# Load the dataset
loan = pd.read_csv('loan.csv')

# Function for loan prediction
def loan_prediction(input_data):
    convert_data_to_numpy = np.asarray(input_data)
    reshape_input_data = convert_data_to_numpy.reshape(1, -1)
    
    prediction = loaded_model.predict(reshape_input_data)

    return "We're sorry, Your Loan has not been Approved." if prediction == 0 else "Congratulations, Your Loan has been Approved."

# Chat page (with vertical charts and insights)
def chart_page():
    st.title("Overview")
    
    # Create vertical count plot using Plotly for Gender
    fig_gender = px.histogram(
        loan, 
        x='Gender',  # Use 'x' for vertical charts
        color='Loan_Status', 
        barmode='group', 
        title='Gender Status of Loan Applicants',
        labels={'Loan_Status': 'Loan Status'},
        color_discrete_sequence=px.colors.sequential.Blues,
        orientation='v'  # Set orientation to vertical
    )
    
    st.plotly_chart(fig_gender)

    # Create vertical count plot using Plotly for Education
    fig_education = px.histogram(
        loan, 
        x='Education',  # Use 'x' for vertical charts
        color='Loan_Status',
        barmode='group',
        title='Educational Status vs Loan Status',
        labels={'Loan_Status': 'Loan Status'},
        color_discrete_sequence=px.colors.sequential.Blues,
        orientation='v'  # Set orientation to vertical
    )

    st.plotly_chart(fig_education)

    # Displaying value counts
    education_counts = loan['Education'].value_counts()
    st.write("Value Counts of Education Status:")
    st.write(education_counts)

    # Add insights about the charts
    st.subheader("Insights")
    st.markdown("""
    - *Gender Distribution*: The chart shows that the proportion of loan approvals is relatively balanced between genders, with some variations.
    - *Educational Status*: The educational status chart indicates that applicants with higher education levels tend to have a higher loan approval rate.
    - *Overall Trends*: It's important to consider these factors when analyzing loan approval rates, as they can influence lending decisions.
    """)

# Dashboard page
def dashboard_page():
    st.title("Dashboard Page")
    st.markdown("<h3 style='text-align: center;'>Bank Loan Prediction</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: green; font-size: 16px;'>Input the required values:</h5>", unsafe_allow_html=True)

    # Collecting user input values
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Gender = st.selectbox("Gender (Select 0 or 1)", options=[0, 1])
        Education = st.selectbox("Education (0 or 1)", options=[0, 1])
        Coapplicant_Income = st.number_input('Coapplicant Income', value=0.0)

    with col2:
        Married = st.selectbox("Married (Select 0 or 1)", options=[0, 1])
        Self_Employed = st.selectbox("Employment Status (0 or 1)", options=[0, 1])
        Loan_Amount = st.number_input('Loan Amount', value=0.0)

    with col3:
        Dependents = st.number_input('Dependents', value=0)
        Applicant_Income = st.number_input('Applicant Income', value=0.0)
        Loan_Amount_Term = st.number_input('Loan Amount Term', value=0)

    Credit_History = st.selectbox('Credit History (Select 0 or 1)', options=[0, 1])
    Property_Area = st.selectbox('Property Area (Select a category)', options=[0, 1, 2])  # Adjust based on actual categories

    # When user clicks "Predict"
    if st.button('Bank Loan Application'):
        try:
            input_data = [
                int(Gender),
                int(Married),
                int(Dependents),
                int(Education),
                int(Self_Employed),
                float(Applicant_Income),
                float(Coapplicant_Income),
                float(Loan_Amount),
                float(Loan_Amount_Term),
                int(Credit_History),
                int(Property_Area)
            ]
            
            result = loan_prediction(input_data)
            st.success(result)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Main function to switch between pages
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ["Chart", "Dashboard"])

    if page == "Chart":
        chart_page()
    elif page == "Dashboard":
        dashboard_page()

# Run the app
if __name__ == "__main__":
    main()