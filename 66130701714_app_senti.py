import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the sentiment analysis model
model_name = "poom-sci/WangchanBERTa-finetuned-sentiment"
sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)

# Streamlit app
st.title("Thai Sentiment Analysis App")

# File upload section
uploaded_file = st.file_uploader("Upload a CSV file with comments", type=["csv"])

# If a file is uploaded
if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Check if the DataFrame has the expected column
    if 'comment' not in df.columns:
        st.error("The uploaded CSV file must contain a 'comment' column.")
    else:
        # Process each comment for sentiment analysis
        st.subheader("Sentiment Analysis Results:")
        predictions = []
        
        # Analyze sentiment for each comment
        for comment in df['comment']:
            results = sentiment_analyzer([comment])
            sentiment = results[0]['label']
            score = results[0]['score']
            predictions.append({'comment': comment, 'sentiment': sentiment, 'score': score})

        # Create a DataFrame for the predictions
        predictions_df = pd.DataFrame(predictions)
        
        # Display the results in the app
        st.write(predictions_df)

        # Save predictions to a new CSV file
        output_file = "sentiment_analysis_results.csv"
        predictions_df.to_csv(output_file, index=False)
        
        # Provide a download link for the results
        st.success(f"Sentiment analysis completed! Results saved to {output_file}.")
        st.download_button(label="Download Results", data=predictions_df.to_csv(index=False), file_name=output_file, mime='text/csv')
else:
    # Input text area for single comment analysis (if no file is uploaded)
    text_input = st.text_area("Enter Thai text for sentiment analysis", "ขอความเห็นหน่อย... ")

    # Button to trigger analysis for single input
    if st.button("Analyze Sentiment"):
        results = sentiment_analyzer([text_input])
        sentiment = results[0]['label']
        score = results[0]['score']
        
        # Display result as progress bars
        st.subheader("Sentiment Analysis Result:")
        if sentiment == 'pos':
            st.success(f"Positive Sentiment (Score: {score:.2f})")
            st.progress(score)
        elif sentiment == 'neg':
            st.error(f"Negative Sentiment (Score: {score:.2f})")
            st.progress(score)
        else:
            st.warning(f"Neutral Sentiment (Score: {score:.2f})")
            st.progress(score)
