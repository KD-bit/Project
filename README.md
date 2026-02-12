# 🏨 Hotel Booking Demand Analysis & Cancellation Prediction  
## Data Analytics – Machine Learning Project  

---

## 👥 Team Members  

- Shoakbarov Muhammadqodir (Leader) – 202490299  
- Raxmatulayev Humoyun – 202490257  
- Pak Andrey – 202490242  
- Ahmadjonov Shohruzbek – 202490028  

---

# 📌 Project Overview  

This project analyzes hotel booking data to understand customer behavior, booking trends, and cancellation patterns.

Using Data Analytics and Machine Learning techniques, we transform raw booking data into meaningful business insights that support better decision-making in hotel management.

---

# 📊 Dataset Information  

Dataset Title: Hotel Booking Demand Dataset  
Source: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand  

Description:  
The dataset contains booking information for both a City Hotel and a Resort Hotel.  
It includes booking dates, stay duration, guest details, cancellation status, booking channels, country of origin, and lead time.

Dataset Size:  
- 119,000+ rows  
- 32 columns  

Why this dataset:  
- Real-world business data  
- Contains categorical and numerical features  
- Suitable for time-based analysis  
- Allows building a Machine Learning model for cancellation prediction  

---

# 🎯 Project Objectives  

The main objective is to analyze booking behavior and predict cancellations.

Business Questions:

- Which hotel type generates more bookings?
- What is the cancellation rate?
- Which months have the highest demand?
- Does lead time affect cancellation probability?
- Which customers are more likely to cancel?

Expected Outcomes:

- Identify peak and low seasons  
- Understand cancellation behavior  
- Improve revenue management strategies  
- Support better booking policies  

---

# 🧹 Data Preparation  

## Load Dataset

import pandas as pd

df = pd.read_csv("hotel_bookings.csv")
df.head()
## Data Cleaning

- Remove duplicates  
- Handle missing values  
- Convert date columns  

df = df.drop_duplicates()
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
## Feature Engineering

df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
df['total_guests'] = df['adults'] + df['children'] + df['babies']
---

# 📈 Exploratory Data Analysis (EDA)

## Hotel Distribution

df['hotel'].value_counts()
## Cancellation Rate

cancellation_rate = df['is_canceled'].mean()
## Monthly Booking Trend

monthly_bookings = df.groupby('arrival_date_month')['hotel'].count()
## Lead Time vs Cancellation

df.groupby('is_canceled')['lead_time'].mean()
## Top Booking Countries

df['country'].value_counts().head(10)
---

# 🤖 Machine Learning Model  

Objective: Predict Booking Cancellation  

Target variable:

is_canceled
## Modeling Example

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
---

# 📊 Key Findings  

- Resort hotels tend to have longer stays  
- High lead time bookings are more likely to be canceled  
- Summer months show peak demand  
- Repeat guests cancel less frequently  
- Some countries dominate total bookings  

---

# 💼 Business Recommendations  

- Apply stricter policies for long lead-time reservations  
- Offer promotions during low-demand months  
- Focus marketing on repeat customers  
- Improve forecasting for peak seasons  

---

# 🗓 Project Timeline
Week 1 → Dataset selection & planning  
Week 2 → Data cleaning  
Week 3 → Exploratory analysis  
Week 4 → Machine Learning modeling  
Week 5 → Final report & presentation  

---

# 🎓 Skills Gained  

- Data preprocessing using Pandas  
- Exploratory Data Analysis  
- Feature engineering  
- Classification modeling  
- Business interpretation of results  
- GitHub collaboration  

---

# 📌 Conclusion  

This project demonstrates how hotel booking data can be transformed into actionable insights.

By combining data analysis and machine learning, we provide strategic recommendations that can help hotels reduce cancellations, optimize revenue, and improve operational planning.

---

# 📚 References  

Pandas Documentation: https://pandas.pydata.org/docs/  
Scikit-learn Documentation: https://scikit-learn.org/
