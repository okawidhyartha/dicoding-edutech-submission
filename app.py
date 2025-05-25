import pandas as pd
import joblib
import streamlit as st

scaler = joblib.load("./model/standard_scaler.joblib")
course_encoder = joblib.load("./model/label_encoder_Course.joblib")
marital_status_encoder = joblib.load(
    "./model/label_encoder_Marital_status.joblib")
model = joblib.load("./model/best_random_forest_model.joblib")

def data_preprocessing(data):
    df = pd.DataFrame(data)

    df["Approval_rate_1st_sem"] = df["Curricular_units_1st_sem_approved"] / \
        df["Curricular_units_1st_sem_enrolled"]
    df["Approval_rate_2nd_sem"] = df["Curricular_units_2nd_sem_approved"] / \
        df["Curricular_units_2nd_sem_enrolled"]

    df["Approval_rate_1st_sem"].fillna(0, inplace=True)
    df["Approval_rate_2nd_sem"].fillna(0, inplace=True)

    df.drop(columns=[
        "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_enrolled",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_enrolled",
    ], inplace=True)

    df["Course"] = course_encoder.fit_transform(df["Course"])
    df["Marital_status"] = marital_status_encoder.fit_transform(
        df["Marital_status"])
    df = scaler.transform(df)

    return df


def predict(data):
    preprocessed_data = data_preprocessing(data)
    predictions = model.predict(preprocessed_data)

    if (predictions == 1):
        return "Graduate"

    return "Dropout"


st.title("Student Graduation Prediction")

marital_status = st.selectbox(
    "Marital Status",
    [
        "1 - Single",
        "2 - Married",
        "3 - Widower",
        "4 - Divorced",
        "5 - Facto union",
        "6 - Legally separated"
    ]
)
application_order = st.number_input("Application Order", min_value=0, step=1, max_value=9)
course = st.selectbox(
    "Course",
    [
        "33 - Biofuel Production Technologies",
        "171 - Animation and Multimedia Design",
        "8014 - Social Service (evening attendance)",
        "9003 - Agronomy",
        "9070 - Communication Design",
        "9085 - Veterinary Nursing",
        "9119 - Informatics Engineering",
        "9130 - Equinculture",
        "9147 - Management",
        "9238 - Social Service",
        "9254 - Tourism",
        "9500 - Nursing",
        "9556 - Oral Hygiene",
        "9670 - Advertising and Marketing Management",
        "9773 - Journalism and Communication",
        "9853 - Basic Education",
        "9991 - Management (evening attendance)"
    ]
)
daytime_evening_attendance = st.selectbox(
    "Daytime/Evening Attendance",
    ["1 - Daytime", "0 - Evening"]
)
previous_qualification_grade = st.number_input(
    "Previous Qualification Grade", min_value=0.0, max_value=200.0)
admission_grade = st.number_input(
    "Admission Grade", min_value=0.0, max_value=200.0)
displaced = st.selectbox("Displaced", ["0 - No", "1 - Yes"])
debtor = st.selectbox("Debtor", ["0 - No", "1 - Yes"])
tuition_fees_up_to_date = st.selectbox(
    "Tuition Fees Up To Date", ["0 - No", "1 - Yes"])
gender = st.selectbox("Gender", ["1 - Male", "0 - Female"])
scholarship_holder = st.selectbox("Scholarship Holder", ["0 - No", "1 - Yes"])
age_at_enrollment = st.number_input(
    "Age at Enrollment", min_value=15, max_value=100)
curricular_units_1st_sem_approved = st.number_input(
    "Curricular Units 1st Sem Approved", min_value=0, step=1)
curricular_units_1st_sem_enrolled = st.number_input(
    "Curricular Units 1st Sem Enrolled", min_value=0, step=1)
curricular_units_1st_sem_grade = st.number_input(
    "Curricular Units 1st Sem Grade", min_value=0.0, max_value=20.0)
curricular_units_2nd_sem_approved = st.number_input(
    "Curricular Units 2nd Sem Approved", min_value=0, step=1)
curricular_units_2nd_sem_enrolled = st.number_input(
    "Curricular Units 2nd Sem Enrolled", min_value=0, step=1)
curricular_units_2nd_sem_grade = st.number_input(
    "Curricular Units 2nd Sem Grade", min_value=0.0, max_value=20.0)


if st.button("Predict"):
    input_data = [{
        "Marital_status": int(marital_status.split(" - ")[0]),
        "Application_order": application_order,
        "Course": int(course.split(" - ")[0]),
        "Daytime_evening_attendance": int(daytime_evening_attendance.split(" - ")[0]),
        "Previous_qualification_grade": previous_qualification_grade,
        "Admission_grade": admission_grade,
        "Displaced": int(displaced.split(" - ")[0]),
        "Debtor": int(debtor.split(" - ")[0]),
        "Tuition_fees_up_to_date": int(tuition_fees_up_to_date.split(" - ")[0]),
        "Gender": int(gender.split(" - ")[0]),
        "Scholarship_holder": int(scholarship_holder.split(" - ")[0]),
        "Age_at_enrollment": age_at_enrollment,
        "Curricular_units_1st_sem_grade": curricular_units_1st_sem_grade,
        "Curricular_units_2nd_sem_grade": curricular_units_2nd_sem_grade,
        "Curricular_units_1st_sem_approved": curricular_units_1st_sem_approved,
        "Curricular_units_1st_sem_enrolled": curricular_units_1st_sem_enrolled,
        "Curricular_units_2nd_sem_approved": curricular_units_2nd_sem_approved,
        "Curricular_units_2nd_sem_enrolled": curricular_units_2nd_sem_enrolled
    }]
    result = predict(input_data)
    st.success(f"Prediction: {result}")
