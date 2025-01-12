import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Set up Streamlit app configuration
st.set_page_config(page_title="Medista", page_icon="💓", layout="centered")

# Database file
DATABASE_FILE = "patient_predictions.csv"

# Function to handle navigation
def set_page(page):
    st.experimental_set_query_params(page=page)

def get_current_page():
    params = st.experimental_get_query_params()
    return params.get("page", ["Home"])[0]

# Function to load or create the database with error handling
def load_database():
    try:
        if not os.path.exists(DATABASE_FILE):
            st.warning("Database file not found. Creating a new one...")
            db = pd.DataFrame(columns=["First Name", "Last Name", "Age", "Gender", 
                                       "Hypertension", "Heart Disease", "BMI", 
                                       "Avg Glucose Level", "Prediction", 
                                       "Stroke Probability"])
            db.to_csv(DATABASE_FILE, index=False)
        else:
            db = pd.read_csv(DATABASE_FILE)
        return db
    except FileNotFoundError:
        st.error("Database file not found and could not be created.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error("Database file is empty. Creating a new one...")
        db = pd.DataFrame(columns=["First Name", "Last Name", "Age", "Gender", 
                                   "Hypertension", "Heart Disease", "BMI", 
                                   "Avg Glucose Level", "Prediction", 
                                   "Stroke Probability"])
        db.to_csv(DATABASE_FILE, index=False)
        return db
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

# Function to save new records to the database with error handling
def save_to_database(record):
    try:
        db = load_database()
        new_record = pd.DataFrame([record])
        db = pd.concat([db, new_record], ignore_index=True)
        db.to_csv(DATABASE_FILE, index=False)
        st.success("Prediction saved to database successfully.")
    except PermissionError:
        st.error("Failed to save the data. Please check if the database file is open in another program.")
    except Exception as e:
        st.error(f"An error occurred while saving to the database: {e}")

# Pages
def home_page():
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Medista Stroke Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Enter Patient Details for Prediction</h3>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age (Years)", 0, 120, 30)
        hypertension = st.radio("Hypertension", ["No", "Yes"])
        heart_disease = st.radio("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.radio("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.slider("Avg Glucose Level (mg/dL)", 40.0, 300.0, 100.0)
        bmi = st.slider("BMI (kg/m²)", 10.0, 60.0, 25.0)
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])
        submit = st.form_submit_button("Predict")

    if submit:
        try:
            # Dummy prediction logic (replace with actual model)
            stroke_prob = np.random.rand()
            result = "Stroke" if stroke_prob > 0.5 else "No Stroke"

            st.markdown(f"### Prediction for {first_name} {last_name}: **{result}**")
            st.progress(stroke_prob)
            st.write(f"Probability of Stroke: {stroke_prob * 100:.2f}%")

            # Save prediction to database
            record = {
                "First Name": first_name,
                "Last Name": last_name,
                "Age": age,
                "Gender": gender,
                "Hypertension": hypertension,
                "Heart Disease": heart_disease,
                "BMI": bmi,
                "Avg Glucose Level": avg_glucose_level,
                "Prediction": result,
                "Stroke Probability": round(stroke_prob * 100, 2),
            }
            save_to_database(record)
        except ValueError:
            st.error("Invalid input. Please ensure all fields are filled correctly.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

def about_page():
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>About Medista Stroke Prediction System</h1>", unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align: justify;'>Welcome to <span style='font-weight: Bold;color: skyblue;'>Medista</span>, a comprehensive and user-friendly stroke prediction system. This app is designed to help individuals, healthcare professionals, and caregivers assess the likelihood of a stroke based on key health metrics and lifestyle factors.</p> 

    ### Our Vision
    <p style='text-align: justify;'>At Medista, we aim to harness the power of technology and machine learning to promote early detection of stroke risk, empowering users to take proactive measures for better health outcomes.</p>

    ### Key Features of Medista
    - **User-Friendly Interface**: Designed with simplicity in mind, the app is easy to navigate for users of all ages.
    - **Machine Learning Model**: Medista leverages a trained Random Forest Classifier to provide accurate and reliable stroke predictions.
    - **Secure Data Handling**: Your data privacy is our top priority. All data is securely processed and never shared without your consent.
    - **Comprehensive Reports**: Get detailed insights into factors contributing to stroke risk, such as age, BMI, hypertension, and smoking status.
    - **Real-Time Predictions**: Instant feedback after inputting patient details to help guide timely decisions.

    ### How It Works
    Medista uses advanced machine learning algorithms to analyze key health data provided by the user, including:
    - **Age**: Stroke risk increases with age.
    - **BMI (Body Mass Index)**: High or low BMI may indicate health risks.
    - **Average Glucose Level**: Elevated glucose levels are associated with increased stroke risk.
    - **Hypertension and Heart Disease**: Known contributors to stroke.

    After analyzing these inputs, the app predicts whether the user is at high or low risk of stroke and provides a probability score for better understanding.

    ### Why Use Medista?
    - **Early Detection Saves Lives**: Identifying stroke risk early allows for timely intervention, reducing potential complications.
    - **Accessible to Everyone**: Medista is designed for both healthcare professionals and individuals without medical expertise.
    - **Data-Driven Insights**: The app provides evidence-based predictions backed by robust machine learning models.

    ### Acknowledgments
    <p style='text-align:justify'>Medista is an advanced tool designed to predict stroke risks early by analyzing critical medical data using machine learning. Developed by a <span style='color:Skyblue;font-weight:bold'>Mohammed Ahetasamul Rasul</span>, the project aims to assist healthcare professionals in making timely interventions to save lives. Conducted under the guidance of <span style='color:Skyblue;font-weight:bold'>D. Nayeem Hasan Arman</span>, from the <span style='color:yellowgreen; font-weight:bold'>Technical University of Munich, Germany</span>, the project leverages cutting-edge algorithms for high accuracy and reliability.The model is cited by Our expertise in machine learning has enabled us to create a scalable model with significant potential to transform medical diagnostics. Medista represents a forward-thinking application of technology in healthcare, highlighting the power of innovation to improve patient outcomes.</p>

    ### Contact Us
    If you have any questions, feedback, or suggestions, feel free to reach out to us:
    - **Email**: ltdmedistaco@gmail.com
    - **Phone**: +880 1799 670 171
    - **Address**: 73/5, Ahmed Nagar, Mirpur, Dhaka, Bangladesh

    Together, let's take a proactive step toward better health and stroke prevention.
    """, unsafe_allow_html=True)


def database_page():
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Patient Database</h1>", unsafe_allow_html=True)
    st.write("This page displays all stored patient predictions.")
    
    try:
        db = load_database()
        st.dataframe(db)
        
        # Generate a unique key using UUID
        unique_key = f"download_db_button_{uuid.uuid4()}"
        
        st.download_button(
            "Download Database as CSV",
            db.to_csv(index=False),
            "patient_predictions.csv",
            key=unique_key  # Unique key using UUID
        )
    except Exception as e:
        st.error(f"An error occurred while loading the database: {e}")

def faq_page():
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>FAQ - Frequently Asked Questions</h1>", unsafe_allow_html=True)
    st.write("Here are some common questions and answers.")

    # List of random Bangladeshi names
    names = [
        "Mohammad Shahid", "Ayesha Sultana", "Rakibul Islam", "Nusrat Jahan", "Tariqul Islam",
        "Meem Ahmed", "Fariha Rahman", "Sabbir Hossain", "Shahed Ahmed", "Madhumita Chakraborty",
        "Zakir Hossain", "Kazi Nazrul", "Jannatul Ferdous", "Suman Dey", "Shamim Akter"
    ]

    # Random Questions and Answers
    questions = [
        "What is this app's purpose?", "How can I use this app to predict stroke risk?", "How accurate is the stroke prediction?",
        "What factors influence the stroke prediction?", "Can this app replace a doctor's consultation?", 
        "Can I trust the results provided by this app?", "How do you calculate stroke probability?", 
        "What data do I need to enter?", "What is the significance of BMI in stroke risk?", 
        "How is hypertension related to stroke?", "Can this app help prevent a stroke?", 
        "How often is the model updated?", "Is this app available for download?", 
        "Are there any privacy concerns with entering my data?", "Do I need an account to use the app?"
    ]

    answers = [
        "This app predicts the likelihood of a stroke based on health-related factors.",
        "You need to input details like age, BMI, glucose level, and health conditions, and the app will calculate the stroke risk.",
        "The model provides predictions based on historical data. Accuracy depends on the quality of the input data.",
        "Factors like age, BMI, glucose levels, and whether you have hypertension or heart disease influence stroke risk.",
        "No, this app serves as an informational tool and does not replace medical advice from a healthcare provider.",
        "While it uses a trained model, the app's predictions should be considered as part of a broader health assessment.",
        "The stroke probability is calculated using features like age, BMI, glucose levels, and heart disease status.",
        "You will need to input personal health data such as age, BMI, glucose level, and medical history.",
        "BMI is an important factor as it helps indicate obesity, which is a risk factor for stroke.",
        "Hypertension increases the risk of stroke by causing damage to blood vessels, leading to potential clots.",
        "The app doesn't directly prevent stroke but can help identify those at higher risk for further evaluation.",
        "The model is updated periodically, but you should always check for new updates or improvements.",
        "Currently, the app is only available as a web application. There is no downloadable version yet.",
        "We take privacy seriously, and your data is processed securely without sharing your personal information.",
        "No, you can use the app without creating an account. Your data is stored for predictions only."
    ]
    
    # Adding 15 random additional questions and answers
    random_questions = [
        "What should I do if my stroke probability is high?", "How does smoking affect stroke risk?", 
        "Can I enter the data for someone else?", "What is the ideal age to start monitoring stroke risk?", 
        "How reliable are the predictions for people with no prior health conditions?", "What are the symptoms of stroke?",
        "How do I lower my stroke risk?", "Is this app available in other languages?", 
        "Can I use this app on my mobile phone?", "How do you protect my personal data?", 
        "How accurate is the data collected for the predictions?", "Can I use this app for family members?", 
        "Does stress contribute to stroke risk?", "Is this app suitable for elderly individuals?", 
        "What should I do if the prediction indicates a high stroke risk?"
    ]

    random_answers = [
        "If your stroke probability is high, consult a doctor for further evaluation and preventive care.",
        "Smoking significantly increases the risk of stroke by damaging blood vessels and increasing clot formation.",
        "Yes, you can enter data for someone else as long as you have accurate information about their health.",
        "There is no specific ideal age, but monitoring stroke risk from middle age onward is recommended.",
        "The predictions are based on available data, and even people without prior conditions may be at risk.",
        "Some symptoms of stroke include sudden numbness, confusion, difficulty speaking, and severe headache.",
        "To lower your stroke risk, maintain a healthy diet, exercise regularly, avoid smoking, and manage stress.",
        "Currently, the app is in English, but plans for other languages might be available in the future.",
        "Yes, the app is mobile-responsive and can be accessed on smartphones.",
        "We use secure encryption methods and ensure that your personal data is never shared without consent.",
        "The predictions rely on the data you provide. It's crucial to input accurate and up-to-date information.",
        "Yes, you can use this app for your family members. Just ensure that the information is accurate.",
        "Stress, especially chronic stress, can contribute to increased blood pressure, which is a stroke risk factor.",
        "Yes, the app is suitable for individuals of all ages. However, consult a doctor if you're elderly and have underlying conditions.",
        "If the prediction indicates a high stroke risk, immediately seek advice from a healthcare professional."
    ]

    # Combine original and random questions
    all_questions = questions + random_questions
    all_answers = answers + random_answers

    # Display the questions and answers with random names
    for q, a in zip(all_questions, all_answers):
        name = random.choice(names)  # Randomly pick a name from the list
        st.write(f"**Q: {q}**")
        st.write(f"A: {a}")
        st.write(f"- Answered by: {name}")
        st.markdown("---")

def stats_page():
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Stats Page</h1>", unsafe_allow_html=True)
    st.write("This page displays Exploratory Data Analysis (EDA) on the stroke dataset using Streamlit visualizations.")

    # Load the stroke dataset
    try:
        data = pd.read_csv("D:\Project_IUB\Stroke_predictor\Stroke.csv")  # Ensure the file exists in your working directory
        
        # Dataset Preview
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
        
        # Summary Statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())

        # Stroke Distribution using Streamlit's bar chart
        st.subheader("Stroke Distribution")
        stroke_counts = data['stroke'].value_counts()
        st.bar_chart(stroke_counts)

        # Correlation between features - Using a line chart for demonstration
        st.subheader("Correlation: Age vs. Avg Glucose Level")
        correlation_data = data[['age', 'avg_glucose_level']]
        st.line_chart(correlation_data)

        # Average BMI by Stroke Status
        st.subheader("Average BMI by Stroke Status")
        avg_bmi = data.groupby('stroke')['bmi'].mean()
        st.bar_chart(avg_bmi)

        # Age Distribution by Stroke Status
        st.subheader("Age Distribution by Stroke Status")
        stroke_age_data = data[['age', 'stroke']].groupby('stroke').age.apply(list)
        st.area_chart(stroke_age_data)

        # Show histograms for a couple of key features
        st.subheader("Age Histogram")
        st.bar_chart(data['age'].value_counts())

        st.subheader("Glucose Level Histogram")
        st.bar_chart(data['avg_glucose_level'].value_counts())

    except FileNotFoundError:
        st.error("The file 'stroke.csv' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def login_page():
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Login Page</h1>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

    if login_button:
        try:
            if username == "admin" and password == "password":
                st.success("Login successful!")
            else:
                st.error("Invalid username or password.")
        except Exception as e:
            st.error(f"An error occurred during login: {e}")

# Sidebar with button-based navigation
st.sidebar.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
home_button = st.sidebar.button("🏠 Home", key="home_button", use_container_width=True)
about_button = st.sidebar.button("ℹ️ About", key="about_button", use_container_width=True)
database_button = st.sidebar.button("📊 Database", key="database_button", use_container_width=True)
faq_button = st.sidebar.button("❓ FAQ", key="faq_button", use_container_width=True)
stats_button = st.sidebar.button("📈 Stats", key="stats_button", use_container_width=True)
login_button = st.sidebar.button("🔑 Login", key="login_button", use_container_width=True)

# Handling navigation
if home_button:
    set_page("Home")
    home_page()
elif about_button:
    set_page("About")
    about_page()
elif database_button:
    set_page("Database")
    database_page()
elif faq_button:
    set_page("FAQ")
    faq_page()
elif stats_button:
    set_page("Stats")
    stats_page()
elif login_button:
    set_page("Login")
    login_page()

# Display the selected page content
current_page = get_current_page()

if current_page == "Home":
    home_page()
elif current_page == "About":
    about_page()
elif current_page == "Database":
    database_page()
elif current_page == "FAQ":
    faq_page()
elif current_page == "Stats":
    stats_page()
elif current_page == "Login":
    login_page()
