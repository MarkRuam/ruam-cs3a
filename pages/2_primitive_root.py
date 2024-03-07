import streamlit as st

st.header("Primitive Root")

def primitive(q, g):
    
    count = 0
    for i in range(1, q+1):
        if q % i == 0:
           count += 1
    if count > 2:
        st.write(f"{q} is not a prime number!!")
    elif count == 2:
        isprimitive = bool
        x = []
        isp = []
        for i in range(q-1):
            for j in range(q):
                n = i + 1
                e = j + 1
                z = n**e % q 
                if z not in x:
                    x.append(z)
                    if e <= (q-2) :
                        st.write(f"{n}^{e} mod {q} = {z}", end = ", ")
                    if e == (q-1):
                        st.write(f"{n}^{e} mod {q} = {z} ==> {n} is primitive root of {q}",end=", ")
                        isprimitive = True
                        isp.append(n)
                elif z in x:
                    x.clear()
                    st.write("")
                    break
        if g in isp:
            st.write(f"{g} is primitive root: {isprimitive} {isp}")    
        else:
            st.write(f"{g} is NOT primitive root of {q} - List of Primitive roots: {isp}")
        
    return


q = st.number_input("Prime number",min_value=0, max_value=100, value=None, placeholder= "type number...")
g = st.number_input("Primitive root",min_value=0, max_value=100, value=None, placeholder= "type number...")

if st.button("Submit"):

    primitive(q, g)





