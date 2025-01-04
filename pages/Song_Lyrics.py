import streamlit as st

# Streamlit App
st.title("Song Lyrics Uploader")

# Sidebar for Navigation
st.sidebar.header("Options")
upload_option = st.sidebar.radio("Upload Option", ["Upload a File", "Paste Text"])

if upload_option == "Upload a File":
    uploaded_file = st.file_uploader("Choose a text file containing song lyrics", type=["txt"])
    if uploaded_file:
        # Read the file
        lyrics = uploaded_file.read().decode("utf-8")
        st.subheader("Uploaded Lyrics:")
        st.text_area("Lyrics Content", lyrics, height=300)
        
        # Save the lyrics for further processing
        with open("uploaded_lyrics.txt", "w") as f:
            f.write(lyrics)
        st.success("Lyrics saved for further analysis!")

elif upload_option == "Paste Text":
    st.subheader("Paste Song Lyrics Below:")
    lyrics = st.text_area("Lyrics Content", height=300)
    if st.button("Save Lyrics"):
        # Save the lyrics for further processing
        with open("pasted_lyrics.txt", "w") as f:
            f.write(lyrics)
        st.success("Lyrics saved for further analysis!")
