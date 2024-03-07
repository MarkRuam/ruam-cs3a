import streamlit as st

st.header("Caesar Cipher")

def encrypt_decrypt(text, shift_keys, ifdecrypt):
    """
    Encrypts a text using Caesar Cipher with a list of shift keys.
    Args:
        text: The text to encrypt.
        shift_keys: A list of integers representing the shift values for each character.
        ifdecrypt: flag if decrypt or encrypt
    Returns:
        A string containing the encrypted text if encrypt and plain text if decrypt
    """
    shiftkey = shift_keys.split()
    letters = list(text)
    key = []
    result = []
    for i in range(len(text)): 
        key += (shiftkey[i % len(shiftkey)]).split()
    for i in range(len(text)):
        if ifdecrypt: result += chr((ord(letters[i]) - int(key[i]) - 32) % 94 + 32)
        else: result += chr((ord(letters[i]) + int(key[i]) - 32 + 94) % 94 + 32)
        print(i, letters[i], key[i], result[i])
    print("----------")
    
    for i in range(len(text)):
        print(i, result[i], key[i], letters[i])
        fresult = "".join(result)
    print("----------")
    
    return fresult
    
     
# Example usage
text = st.text_area(input())
shift_keys = st.text_area(input())
x = encrypt_decrypt(text, shift_keys, ifdecrypt=False)

print("Text:", text)
print("Shift keys:", shift_keys)
print("Cipher:", x)
print("Decrypted text:", text)




