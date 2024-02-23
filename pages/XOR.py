import streamlit as st

st.header("XOR")

def xor_encrypt(plaintext, key):
    """Encrypts plaintext using XOR cipher with the given key, printing bits involved."""

    ciphertext = bytearray()
    for i in range(len(plaintext)):
        plaintext_byte = plaintext[i]
        key_byte = key[i % len(key)]
        

        xor_result = plaintext_byte ^ key_byte
        
        st.write(f"Plaintext byte: {format(plaintext_byte, '08b')} = {chr(plaintext_byte)}")
        st.write(f"Key byte:       {format(key_byte, '08b')} = {chr(key_byte)}")
        st.write(f"XOR result:     {format(xor_result, '08b')} = {chr(xor_result)}")
        
        ciphertext.append(xor_result)
        st.write("--------------------")

    return ciphertext

def xor_decrypt(ciphertext, key):
    """Decrypts ciphertext using XOR cipher with the given key."""
    return   xor_encrypt(ciphertext, key)  # XOR decryption is the same as encryption

plaintext = bytes(st.text_area("Plain Text:").encode())
key = bytes(st.text_input("Key:").encode())


if st.button("Submit"):

    
    if not (1 < len(plaintext) >= len(key) >= 1):
        st.write("Plaintext length should be equal or greater than the length of key")
    elif not (plaintext != key):
        st.write("Plaintext should not be equal to the key")
    else:
        st.balloons()
        ciphertext = xor_encrypt(plaintext, key)
        st.write("Ciphertext: ", ciphertext.decode())
        
        decrypted_text = xor_decrypt(ciphertext, key)
        st.write("Decrypted: ", decrypted_text.decode())