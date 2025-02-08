import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import cleantext

st.set_page_config(page_title="Sentiment Analysis Web App", layout="wide")

# Header
st.title("Sentiment Analysis Web App")
st.write("Upload your dataset and perform sentiment analysis with multiple models and detailed visualizations.")

# Function Definitions
def preprocess_text(text):
    return cleantext.clean(
        text, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True
    )

def generate_wordcloud(data, title):
    wordcloud = WordCloud(
        max_words=1000, width=1600, height=800, collocations=False
    ).generate(" ".join(data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    st.pyplot(plt)

def polarity_score(text):
    from textblob import TextBlob
    blob = TextBlob(text)
    return blob.sentiment.polarity

def sentiment_label(score):
    if score >= 0.5:
        return "Positive"
    elif score <= -0.5:
        return "Negative"
    else:
        return "Neutral"

def model_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.write("#### Confusion Matrix")
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

    # ROC Curve
    st.write("#### ROC Curve")
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    if y_pred_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='blue')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        st.pyplot(plt)

# CSV Analysis
st.header("Upload Dataset for Sentiment Analysis")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.write(df.head())

    text_column = st.selectbox("Select the column containing text", options=df.columns)

    if st.button("Preprocess and Analyze"):
        # Preprocessing
        df[text_column] = df[text_column].apply(preprocess_text)

        # # Word Cloud
        # st.write("## Word Clouds")
        # st.write("### Overall Word Cloud")
        # generate_wordcloud(df[text_column], "Overall Word Cloud")
        
        # # Positive and Negative word clouds:
        # # Positive and Negative Word Clouds
        # st.write("### Positive and Negative Word Clouds")
        # positive_text = df[df['Sentiment'] == "Positive"][text_column]
        # negative_text = df[df['Sentiment'] == "Negative"][text_column]

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.write("#### Positive Word Cloud")
        #     wordcloud_positive = WordCloud(
        #         max_words=1000, width=1600, height=800, collocations=False, colormap="Greens"
        #     ).generate(" ".join(positive_text))
        #     plt.figure(figsize=(8, 5))
        #     plt.imshow(wordcloud_positive, interpolation='bilinear')
        #     plt.axis("off")
        #     st.pyplot(plt)

        # with col2:
        #     st.write("#### Negative Word Cloud")
        #     wordcloud_negative = WordCloud(
        #         max_words=1000, width=1600, height=800, collocations=False, colormap="Reds"
        #     ).generate(" ".join(negative_text))
        #     plt.figure(figsize=(8, 5))
        #     plt.imshow(wordcloud_negative, interpolation='bilinear')
        #     plt.axis("off")
        #     st.pyplot(plt)
        
        # Word Cloud -  combined
        df['Polarity'] = df[text_column].apply(polarity_score)
        df['Sentiment'] = df['Polarity'].apply(sentiment_label)

        # Word Cloud
        st.write("## Word Clouds")
        st.write("### Positive and Negative Word Clouds")

        positive_text = df[df['Sentiment'] == "Positive"][text_column]
        negative_text = df[df['Sentiment'] == "Negative"][text_column]

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Positive Word Cloud")
            wordcloud_positive = WordCloud(
                max_words=1000, width=1600, height=800, collocations=False, colormap="Greens"
            ).generate(" ".join(positive_text))
            plt.figure(figsize=(8, 5))
            plt.imshow(wordcloud_positive, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

        with col2:
            st.write("#### Negative Word Cloud")
            wordcloud_negative = WordCloud(
                max_words=1000, width=1600, height=800, collocations=False, colormap="Reds"
            ).generate(" ".join(negative_text))
            plt.figure(figsize=(8, 5))
            plt.imshow(wordcloud_negative, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

        

        # Sentiment Analysis
        df['Polarity'] = df[text_column].apply(polarity_score)
        df['Sentiment'] = df['Polarity'].apply(sentiment_label)

        st.write("## Sentiment Distribution")
        st.write("### Bar Chart of Sentiments")
        sentiment_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="Set2")
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)

        st.write("### Pie Chart of Sentiments")
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel", len(sentiment_counts)))
        ax.set_title("Sentiment Proportions")
        st.pyplot(fig)

        st.write("### Histogram of Polarity Scores")
        fig, ax = plt.subplots()
        sns.histplot(df['Polarity'], kde=True, bins=30, ax=ax, color="skyblue")
        ax.set_title("Polarity Score Distribution")
        ax.set_xlabel("Polarity Score")
        st.pyplot(fig)

        st.write("### Box Plot of Polarity by Sentiment")
        fig, ax = plt.subplots()
        sns.boxplot(x="Sentiment", y="Polarity", data=df, palette="coolwarm", ax=ax)
        ax.set_title("Polarity Spread by Sentiment")
        st.pyplot(fig)

        st.write("### Sentiment Count Comparison")
        fig, ax = plt.subplots()
        sns.countplot(x="Sentiment", data=df, palette="viridis", ax=ax)
        ax.set_title("Sentiment Count Comparison")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Train-Test Split and Modeling
        st.write("## Model Training")
        X = df[text_column]
        y = df['Sentiment'].apply(lambda x: 1 if x == "Positive" else 0)  # Binary classification
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Naive Bayes": BernoulliNB(),
            "Support Vector Machine": LinearSVC(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }

        for model_name, model in models.items():
            st.write(f"## {model_name}")
            model.fit(X_train, y_train)
            model_evaluation(model, X_test, y_test)

        # Allow file download
        st.write("### Download Analyzed Data")
        @st.cache
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df)
        st.download_button("Download as CSV", data=csv, file_name="analyzed_data.csv", mime="text/csv")
