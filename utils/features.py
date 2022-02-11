import streamlit as st
from torch import unsafe_chunk

header_temp = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <div class="table">
    <table>
        <tr>
            <th>ID</th>
            <th>Column name</th>
            <th>Column represented Id</th>
            <th>Possible values</th>
        </tr>
    </table>
"""

html_temp = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <div class="table"><table>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
    </table>
    </div>
"""

class features:
    def __init__(self, index, value, key, possible_values):
        self.index = index
        self.value = value
        self.key = key
        self.possible_values = possible_values

    def __str__(self):

        return html_temp.format(self.index, self.value, self.key, self.possible_values)  
            # add a streamlit button to add a job to the list

def display_features(fea):
    st.markdown(header_temp,unsafe_allow_html=True)
    j = 1
    for key, value in fea.items():
        st.markdown(features(j, value, key , ''),unsafe_allow_html=True)
        j += 1
    # for i in range(len(list(fea.keys()))):
    #     st.markdown(features(i, fea), unsafe_allow_html=True)