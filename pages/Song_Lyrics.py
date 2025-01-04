import streamlit as st
from pydub import AudioSegment
from pydub.playback import play

def main():
    st.title("Song Uploader and Lyrics Analyzer")
    
    # Tabbed Interface
    tab1, tab2 = st.tabs(["Upload Song File", "Paste Song Lyrics"])

    # Tab 1: Upload Song File
    with tab1:
        st.subheader("Upload Your Song")
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "flac"])
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            
            # Play audio file (convert to WAV if not)
            try:
                audio = AudioSegment.from_file(uploaded_file)
                st.audio(uploaded_file, format='audio/wav')
                st.write("Audio uploaded successfully. Now processing...")
            except Exception as e:
                st.error(f"Error processing audio: {e}")
        else:
            st.info("No file uploaded yet. Please upload a song file.")

    # Tab 2: Paste Song Lyrics
    with tab2:
        st.subheader("Paste Your Song Lyrics")
        lyrics = st.text_area("Paste the lyrics of the song here:")
        if lyrics:
            st.write("You provided the following lyrics:")
            st.text_area("Lyrics Output", value=lyrics, height=300)
        else:
            st.info("No lyrics pasted yet. Please paste song lyrics in the box above.")

if __name__ == "__main__":
    main()
