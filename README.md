**Machine Learning Model for Heart Failure predictions**

I used open source dataset of heart failure patients. I trained model in notebook by testing few different algorithms (XGBoost and Random forest) that seemed most appropriate to me, observing their performance when tuning and picking the one with best results/performance for this dataset. I tested with GridSearch and OptimisedSearch. I also experimented with Voting Classifier but decided to leave that out for now. I decided not to complicate interface with Streamlit, so i used prebuilt FastAPI interface for it.

**Using:**
1.Python
2. FastAPI and UVICORN
3. Scikitlearn and XGBoost (left out of main file)
4. Pandas and NumPy
5. Render (for deployment)

Website: [https://trained-model-jltq.onrender.com](https://trained-model-jltq.onrender.com/docs)
