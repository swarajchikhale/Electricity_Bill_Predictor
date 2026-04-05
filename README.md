# ⚡ Electricity Bill & Unit Consumption Predictor

A Machine Learning project that predicts **monthly electricity unit consumption (kWh)** and **electricity bill (₹)** based on household appliance usage.

---

## 🚀 Project Overview

Electricity usage is increasing with modern appliances, making it difficult for users to estimate their monthly bill. This project uses **Machine Learning (Linear Regression)** to predict:

- 🔹 Monthly electricity units consumed (kWh)
- 🔹 Monthly electricity bill (₹)

It also includes a **Streamlit web application** where users can input their usage details and get instant predictions.

---

## 🧠 Features

- ✅ Predicts electricity **unit consumption**
- ✅ Predicts electricity **bill amount**
- ✅ User-friendly **Streamlit web interface**
- ✅ Uses real-world input parameters
- ✅ Fast and lightweight ML model (Linear Regression)

---

## 📊 Dataset

The dataset is stored in CSV format and contains the following columns:

### 🔹 Input Features
- `people`
- `house_size` (small / medium / large)
- `ac_hours`
- `fan_hours`
- `fridge` (0 or 1)
- `washing_machine`
- `tv_hours`
- `laptop_hours`
- `season` (summer / winter / monsoon)

### 🔹 Output Targets
- `unit_consume` (kWh)
- `bill` (₹)

---

## ⚙️ Tech Stack

- 🐍 Python
- 📊 Pandas, NumPy
- 🤖 Scikit-learn (Linear Regression, OneHotEncoder)
- 💾 Joblib (model saving/loading)
- 🌐 Streamlit (web app)

---

## 🏗️ Project Structure


ElectricityBillPredictor/
│── data/
│ └── electricity_data.csv
│
│── notebooks/
│ └── train_models.py
│
│── model/
│ ├── preprocessor.pkl
│ ├── bill_model.pkl
│ └── unit_model.pkl
│
│── app/
│ └── app.py
│
│── requirements.txt
│── README.md


---

## 🔄 Workflow

1. Collect and prepare dataset (CSV)
2. Load dataset using Pandas
3. Preprocess categorical data using OneHotEncoder
4. Split data into training and testing sets
5. Train two Linear Regression models:
   - Unit Consumption Model
   - Electricity Bill Model
6. Evaluate models using MAE and R² Score
7. Save models using Joblib (`.pkl`)
8. Build Streamlit app for user interaction

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Train the Models
cd notebooks
python train_models.py
3️⃣ Run the Streamlit App
cd app
streamlit run app.py
💡 Output Example
🔹 Predicted Unit Consumption: 250 kWh
🔹 Predicted Electricity Bill: ₹ 2500
🔹 Approx per unit rate shown in UI
📈 Model Used
Linear Regression
Simple and efficient for continuous prediction
Works well for small to medium datasets
📌 Future Improvements
🔹 Use advanced models (Random Forest, XGBoost)
🔹 Add appliance-wise cost breakdown
🔹 Provide electricity saving tips
🔹 Deploy app on cloud (Render / Streamlit Cloud)
🔹 Integrate with smart meter (IoT)
📚 References
Scikit-learn Documentation
Streamlit Documentation
Python Documentation
```

📞 Support & Contact
Getting Help
Documentation: Check the /docs folder for detailed guides
Issues: Report bugs via GitHub Issues
Discussions: Join our community discussions
Community
Discord Server: [Join our community](https://discord.com/invite/EtQdFmED)
Telegram : [Community discussions](https://t.me/collabcoderx)

🌟 If you found this project helpful, please consider giving it a star! 🌟
Built with ❤️ for the developer community

Happy Coding! 🚀

👨‍💻 Author
Swaraj Chikhale
B.Tech IT Student
