import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("==================================================")
print("     ADVANCED HOUSE PRICE PREDICTOR SYSTEM        ")
print("==================================================")

# 1. डेटासेट तैयार करना
size = [1000, 1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000]
rooms = [2, 3, 3, 4, 4, 5, 5, 5, 6]
age = [10, 5, 8, 3, 6, 2, 4, 1, 5]
price = [4500000, 7000000, 8200000, 12000000, 14000000, 18000000, 21000000, 24000000, 26000000]

data = {
    'Size_SqFt': size,
    'Bedrooms': rooms,
    'Age_Years': age,
    'Price_INR': price
}
df = pd.DataFrame(data)

# 2. मॉडल को ट्रेन करना
X = df[['Size_SqFt', 'Bedrooms', 'Age_Years']]
y = df['Price_INR']

model = LinearRegression()
model.fit(X, y)
print("🤖 AI मॉडल सफलतापूर्वक ट्रेन हो गया है।")

# 3. डेटा का ग्राफ (Visualization) बनाकर सेव करना
print("📊 डेटा ग्राफ जनरेट किया जा रहा है...")
plt.figure(figsize=(8, 5))
plt.scatter(df['Size_SqFt'], df['Price_INR'] / 100000, color='blue', marker='o', label='Actual Houses')
plt.plot(df['Size_SqFt'], model.predict(X) / 100000, color='red', linewidth=2, label='AI Prediction Line')
plt.title('House Size vs Price (in Lakhs)')
plt.xlabel('Size of House (SqFt)')
plt.ylabel('Price (₹ in Lakhs)')
plt.grid(True)
plt.legend()

# ग्राफ को F: ड्राइव के फोल्डर में इमेज के रूप में सेव करना
graph_filename = 'house_price_chart.png'
plt.savefig(graph_filename)
plt.close()
print(f"✅ ग्राफ सफलतापूर्वक '{graph_filename}' के नाम से सेव हो गया है।")
print("--------------------------------------------------")

# 4. यूजर से इनपुट लेना (Interactive System)
print("\n🔮 अपने सपनों के घर की जानकारी डालें:")
try:
    user_size = float(input("➡️ घर का साइज डालें (SqFt में, उदा. 2500): "))
    user_rooms = float(input("➡️ बेडरूम की संख्या डालें (उदा. 3): "))
    user_age = float(input("➡️ घर कितना पुराना है? (साल में, उदा. 2): "))

    # यूजर के इनपुट पर अनुमान लगाना
    user_input = [[user_size, user_rooms, user_age]]
    predicted_price = model.predict(user_input)[0]

    print("\n==================================================")
    print("🏠 आपके इनपुट के आधार पर AI का अनुमान:")
    print(f"💰 अनुमानित कीमत (Predicted Price): ₹{predicted_price:,.2f}")
    print("==================================================")

except ValueError:
    print("\n❌ कृपया केवल सही नंबर्स ही इनपुट करें!")
