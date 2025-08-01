import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('household_energy.csv', parse_dates=['timestamp'])
st.title("Energy Consumption Forecasting ")

st.write(df.head())

df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['weekday'] = df['timestamp'].dt.weekday
df['date'] = df['timestamp'].dt.date 

daily_consumption = df.groupby('date')['energy_consumption'].sum()
weekly_consumption = df.groupby(df['timestamp'].dt.isocalendar().week)['energy_consumption'].sum()
fig1, ax1 = plt.subplots()

daily_df = daily_consumption.reset_index()
daily_df.columns = ['date', 'energy_consumption']

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.barplot(data=daily_df, x='date', y='energy_consumption', ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)


features = ['temperature','outside_temperature','device_usage', 'hour', 'weekday']
X = df[features] #X is holding the input features for training purpose
y = df['energy_consumption'] #y is o/p variable or we have to predict the energy consumption continously

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model = LinearRegression()

model.fit(X_train, y_train)


st.header("Plot Regression Fit for Multiple Features")
selected_feature = st.selectbox("Select a feature to visualize regression:", features)

fig2, ax2 = plt.subplots()
sns.scatterplot(x=df[selected_feature], y=df["energy_consumption"], marker ='o', alpha=0.3, color='green', ax=ax2)
ax2.set_xlabel(f"No of {selected_feature.replace('_', ' ')}")
ax2.set_ylabel("Daily Energy Consumption (kWh)")
st.pyplot(fig2)



#Detect inefficiencies: consumption 25% higher than moving average(6-hour window)
df['moving_avg'] = df['energy_consumption'].rolling(window=6, min_periods=1).mean()
df['inefficient'] = df['energy_consumption'] > 1.25 * df['moving_avg']


# Prediction Section
st.subheader("Energy Consumption Prediction")
temperature = st.number_input("Enter the temperature: ", value=30.0)
outside_temperature = st.number_input("Enter the outside temperature: ",value=30.0)
device_usage = st.number_input("Enter the device usage: ", value=0)
hour = st.number_input("Enter hour value: ", value=6)
weekday =st.number_input("Enter weekday (0-Mon, 6-Sun): ")

if st.button("Predict"):
    inp_df = pd.DataFrame([[temperature,outside_temperature,device_usage,hour,weekday]], columns=['temperature','outside_temperature','device_usage','hour','weekday'])
    prediction = model.predict(inp_df)[0]
    st.success(f"Predicted Energy Consumption(kWh): {prediction:.3f} kWh")
    st.header("Recommendation")
    if inp_df['device_usage'].iloc[0] > 0 and prediction > 3:
        st.info("Consider turning off idle devices")
    if inp_df['outside_temperature'].iloc[0] > 30 and prediction > 3:
        st.info("High outside temp + high usage: Adjust thermost")
    if inp_df['hour'].iloc[0] in [0, 1, 2, 3, 4] and prediction > 2:
        st.warning("High energy use detected at night. Investigate")