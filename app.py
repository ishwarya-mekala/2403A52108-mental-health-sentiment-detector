import streamlit as st

from model import train_model, predict_sentiment


@st.cache_resource
def load_trained_model():
    model, acc, report = train_model("data/mental_health_dataset.csv")
    return model, acc, report


def main():
    st.set_page_config(
        page_title="Mental Health Sentiment Detector",
        page_icon="🧠",
        layout="centered",
    )

    st.title("🧠 Mental Health Sentiment Detector")
    st.write(
        "This mini app uses a TF-IDF + Logistic Regression model to classify mental-health related text as "
        "**positive**, **neutral**, or **negative**."
    )

    with st.expander("Model details", expanded=False):
        model, acc, report = load_trained_model()
        st.markdown(f"**Test accuracy on small held-out set:** `{acc:.3f}`")
        st.text("Classification report:\n" + report)

    st.subheader("Try it out")
    user_text = st.text_area(
        "Enter a mental-health related text (thought, feeling, or message):",
        height=150,
        placeholder="Example: I feel overwhelmed and anxious about everything lately.",
    )

    if st.button("Analyse sentiment"):
        if not user_text.strip():
            st.warning("Please enter some text to analyse.")
        else:
            model, _, _ = load_trained_model()
            label = predict_sentiment(model, user_text)
            if label == "positive":
                color = "green"
            elif label == "negative":
                color = "red"
            else:
                color = "orange"

            st.markdown(
                f"**Predicted sentiment:** "
                f"<span style='color:{color}; font-size: 1.2em; font-weight: 700;'>{label.upper()}</span>",
                unsafe_allow_html=True,
            )

            st.caption(
                "Note: This is a small educational model and **not** a substitute for professional help."
            )


if __name__ == "__main__":
    main()

