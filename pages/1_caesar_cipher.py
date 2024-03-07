import streamlit as st

st.header("Caesar Cipher")

def encrypt_decrypt(text, shift_keys, ifdecrypt):
    shiftkey = shift_keys.split()
    letters = list(text)
    key = []
    result = []

        # Check if shift_keys is valid
    if not shiftkey or all(s.isspace() for s in shiftkey):
        st.error("Shift keys cannot be empty or contain only spaces")
        return ""  # Return an empty string
    
    for i in range(len(text)): 
        key += (shiftkey[i % len(shiftkey)]).split()
    for i in range(len(text)):
        if ifdecrypt: result += chr((ord(letters[i]) - int(key[i]) - 32) % 94 + 32)
        else: result += chr((ord(letters[i]) + int(key[i]) - 32 + 94) % 94 + 32)
        st.write(i, letters[i], key[i], result[i])
    st.write("----------")
    
    fresult = ""
    for i in range(len(text)):
        st.write(i, result[i], key[i], letters[i])
        fresult = "".join(result)
    st.write("----------")
    
    return fresult
    
     
# Example usage
text = st.text_area("Text")
shift_keys = st.text_area("Shift Keys")

if st.button("Submit"):
    st.balloons()
    st.write("Cipher:", encrypt_decrypt(text, shift_keys, ifdecrypt=False))
    st.write("Text:", text)
    st.write("Shift keys:", shift_keys)
    st.write("Decrypted text:", text)




