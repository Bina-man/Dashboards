import streamlit as st


html_temp = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        <div class="base_Card" style = "
            font-family: Roboto, sans-serif;
            display: flex;
            align-items: start;
            background-color: #9370db36;
            box-shadow: 0 1px 5px rgb(0 0 0 / 90%);
            margin: 10px;
            height: 450px;
            justify-content: start;
            justify-content: start;">
                <div class="base_Card_details" 
                    style = "
                        padding: 10px;
                        margin: 10px;
                        position: relative;
                    ">
                    <div class="name" style = "
                        align-items: start;
                        position: absolute;
                        padding: 15px;
                        display: flex;
                        border-radius: 10px;
                        height: 100px;      
                        width: 350px;                  
                        font-size: x-large;
                        background-color: #ff00c8;
                    ">{}
                    </div>
                    <div class="params">
                        <div class= "param_title" style = "
                            align-items: start;
                            position: absolute;
                            display: flex;
                            border-radius: 10px;
                            margin-top: 150px;
                            height: 250px;      
                            width: 350px;                  
                            font-size: x-large;
                            color: white;
                            padding: 10px;
                            background-color: rgb(49 48 114);"
                        >Parameters 
                        </div>
                        <div class="param_value" style = "
                        align-items: start;
                            position: absolute;
                            display: flex;
                            padding-left: 10px;
                            margin-top: 200px;     
                            width: 350px;                  
                            font-size: x-large;
                            color: white;
                            background-color: rgb(49 48 114);"
                        >{}
                        </div>
                    </div>
                <div class="accuracy" style = "
                        align-items: start;
                            position: relative;
                            display: flex;
                            padding-left: 10px;
                            margin-top: 15px;   
                            margin-left: 400px;     
                            width: 180px;            
                            height: 380px;      
                            font-size: x-large;
                            background-color: lawngreen;
                            color: white;"
                        >Accuracy
                        <div class="accuracy_value_0" style = "
                            align-items: start;
                            position: absolute;
                            display: flex;
                            margin-top: 50px;"
                        >0 => {}
                        </div>
                        <div class="accuracy_value_1" style = "
                            align-items: start;
                            position: absolute;
                            display: flex;
                            margin-top: 100px;"
                        >1 => {}
                        </div>
                </div>
        </div>
"""

class models:
    def __init__(self, value):
        self.value = value[0]
        self.parameters = value[1]
        self.accuracy_0 = list(value[2].values())

    def __str__(self):
        return html_temp.format(self.value, self.parameters, self.accuracy_0[0], self.accuracy_0[1])

def display_models(fea):
    st.markdown(models(fea), unsafe_allow_html=True)