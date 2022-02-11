from operator import mod
from tkinter import font
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.features import *
from utils.models import *
import numpy as np
import joblib

class all_data:
    def __init__(self):
        pass

    def read_data():
        # df = pd.read_csv('../../data/processed/edhs/Unimputed_CHM_1.csv')
        df = pd.read_csv('./data/sample_edhs.csv')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        return df
    

    def missing_values():
        with open('./data/missing_values.txt', 'r') as f:
                missing_values = f.read()
                print(missing_values)

        # change the missing_values to a dictionary
        missing_values = {i.split(':')[0]: i.split(':')[1] for i in missing_values}
        return missing_values
    
    def read_corr():
        val = pd.read_csv('./data/CHM_corr.csv')
        val.drop(['Unnamed: 0'], axis=1, inplace=True)
        # drop the first column of val
        val = val.iloc[:, 1:]
        return val

def missing_bar_chart():
    #Creating the dataset
    data = all_data.missing_values()
    Courses = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize = (10, 5))
    plt.bar(Courses, values)
    plt.xlabel("Programming Environment")
    plt.ylabel("Number of Students")
    plt.title("Students enrolled in different courses")
    st.pyplot(fig)

def correlation_matrix(data):
    fig = plt.figure(figsize = (10, 5))
    sns.heatmap(data, annot=True, cmap='RdYlGn', linewidths=0.5)
    plt.title('Correlation Matrix')
    st.pyplot(fig)

def features_selected():
    ftrr = {
    # Grographical features
    'v024': 'Region',
    'v025': 'Rural/Urban',    
    'v001': 'Cluster Number',
    'v002': 'Household Number',

    # Demographic features
    'v137' : '# of Children Under 5',
    'bord' : 'Birth Order',
    'v219' : 'Total # of Children + Current Pregnancy',
    
    # Age features
    'v212' : 'Age @ 1st Birth',
    'b11' : 'Preceeding Birth Interval',
    'b3' : 'CMC date of birth of child', #Replaced
    'b6' : 'Chids age at Death in days',
    'b7' : 'Chids age at Death in Months (Imputed)',
    'b18': 'DOB',
    'b19': 'Age of child',

    # Health features
    'm14' : 'No of Antenatal Care Visit',
    'v116' : 'Toilet Fascilty of House Hold(Related to Sanitation)',
    'v113' : 'Source of Drinking Water',
    'v157' : 'Household reads Newspaper',
    'v158' : 'Household has Radio',
    'v159' : 'HOusehold has TV',    
    'm70': 'Baby Postnatal Check within two months',    
    'm15': 'place of delivery',

    # Education features
    'v106' : 'Highest Education',
    'v149' : 'Educational Attainment',

    
    # Child information features
    'b4' : 'Sex of Child',
    'b5' : 'Child is Alive (Out Outcome Variable)', 
    'm34': 'when child was put to breast (early vs Late within first Hour)',
    'm55': 'Child given anything other than breast milk',
    
    'm77': 'Was Child put to bare Mothers skin after Birth',
    'v190': 'Wealth Index',
    'm18': 'Size of Child at birth',
    'm19': 'Birth Weight in Kg',
    'v717': 'Employment Women',
    'm4': 'Breastfed',
    'v008': 'CMC date of interview',
    'v228':'Terminated pregnancy',
    'v312': 'Current contraceptive method',
    'h11': 'Had diarrhea recently',
    'b8': 'Age of child in years',
    'b9': 'Age of child in months',
    'Year': 'Year',
    }
    
    return ftrr

def feature_importance(data):
    Courses = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize = (10, 7))
    sns.barplot(values, Courses)    
    plt.xlabel("Features", )
    plt.ylabel("Importance")
    plt.xticks(rotation=45,fontsize=12)
    # change the font size of the y-axis
    plt.yticks(fontsize=12)
    plt.title("Most important factors")
    st.pyplot(fig)
    
def show_results():
    st.subheader('Most Important Factors')
    imputation = st.selectbox('Select Imputation', ['Iterative imputation', 'Ordinary Mean | Mode'])
    st.write("Here are the dominant factors. You can select for multiple models and see the results")
    if imputation == 'Iterative imputation':
        model = st.selectbox('Select Model', ['Random Forest', 'XGBoost'])
        if model == 'Random Forest':
            key =  ['Region',
                    'Rural_Urban',
                    'No_of_Children_under_5',
                    'Age_at_1st_Birth',
                    'BirthOrder',
                    'Prec.BirthInterval',
                    'TotalChildren',
                    'HighestEducation',
                    'SexofChild',
                    'childSize',
                    'WealthQuantile',
                    'prev_termin_pregnancy',
                    'Year',
                    'Toiletfascility',
                    'HandwashFascility',
                    'Place_of_delivery',
                    'Contraceptive_use',
                    'Mediaexposure',
                    'age']
            value = [0.00248893, 0.00162279, 0.21864789, 0.00160378, 0.00747543,
                    0.05029216, 0.06618739, 0.0044109 , 0.0009565 , 0.00825019,
                    0.00733631, 0.00139968, 0.01394231, 0.00554112, 0.00283784,
                    0.00967358, 0.00742899, 0.00596421, 0.58394]
            feature_importance_dict = dict(zip(key, value))
            feature_importance(feature_importance_dict)
            st.subheader('Based on the analysis above the age is the dominant factor, we removed age and see which factors are important')
            key =  ['Region',
                    'Rural_Urban',
                    'No_of_Children_under_5',
                    'Age_at_1st_Birth',
                    'BirthOrder',
                    'Prec.BirthInterval',
                    'TotalChildren',
                    'HighestEducation',
                    'SexofChild',
                    'childSize',
                    'WealthQuantile',
                    'prev_termin_pregnancy',
                    'Year',
                    'Toiletfascility',
                    'HandwashFascility',
                    'Place_of_delivery',
                    'Contraceptive_use',
                    'Mediaexposure']
            value = [0.00575667, 0.00480238, 0.51024392, 0.00397095, 0.02203449,
                    0.12426632, 0.18566448, 0.01075796, 0.00316215, 0.01815575,
                    0.02125358, 0.00368085, 0.03154911, 0.0114833 , 0.00539888,
                    0.01489044, 0.01251673, 0.01041205]
            feature_importance_dict = dict(zip(key, value))
            feature_importance(feature_importance_dict)
        elif model == 'XGBoost':
            key =  ['Region',
                    'Rural_Urban',
                    'No_of_Children_under_5',
                    'Age_at_1st_Birth',
                    'BirthOrder',
                    'Prec.BirthInterval',
                    'TotalChildren',
                    'HighestEducation',
                    'SexofChild',
                    'childSize',
                    'WealthQuantile',
                    'prev_termin_pregnancy',
                    'Year',
                    'Toiletfascility',
                    'HandwashFascility',
                    'Place_of_delivery',
                    'Contraceptive_use',
                    'Mediaexposure',
                    'age']
            value = [0.01890436, 0.0100456 , 0.18065244, 0.02054358, 0.03191504,
                    0.05067764, 0.03781533, 0.02703498, 0.01344586, 0.02132381,
                    0.01946213, 0.01261234, 0.03564634, 0.02768292, 0.01637699,
                    0.03826184, 0.02883448, 0.02537146, 0.38339284]
            feature_importance_dict = dict(zip(key, value))
            feature_importance(feature_importance_dict)
            st.subheader('Based on the analysis above the age is the dominant factor, we removed age and see which factors are important')
            key =  ['Region',
                    'Rural_Urban',
                    'No_of_Children_under_5',
                    'Age_at_1st_Birth',
                    'BirthOrder',
                    'Prec.BirthInterval',
                    'TotalChildren',
                    'HighestEducation',
                    'SexofChild',
                    'childSize',
                    'WealthQuantile',
                    'prev_termin_pregnancy',
                    'Year',
                    'Toiletfascility',
                    'HandwashFascility',
                    'Place_of_delivery',
                    'Contraceptive_use',
                    'Mediaexposure']
            value = [0.02312679, 0.03162296, 0.37141356, 0.0279589 , 0.03996997,
                    0.06859119, 0.06028451, 0.03949566, 0.0232997 , 0.03432206,
                    0.02745046, 0.03267637, 0.03903816, 0.04441945, 0.02808973,
                    0.02511491, 0.05675178, 0.02637392]
            feature_importance_dict = dict(zip(key, value))
            feature_importance(feature_importance_dict)
    elif imputation == 'Ordinary Mean | Mode':
        model = st.selectbox('Select Model', ['Random Forest', 'XGBoost'])
        if model == 'Random Forest':
            key =  ['Region',
                    'Rural_Urban',
                    'No_of_Children_under_5',
                    'Age_at_1st_Birth',
                    'BirthOrder',
                    'Prec.BirthInterval',
                    'TotalChildren',
                    'HighestEducation',
                    'SexofChild',
                    'childSize',
                    'WealthQuantile',
                    'prev_termin_pregnancy',
                    'Year',
                    'Toiletfascility',
                    'HandwashFascility',
                    'Place_of_delivery',
                    'Contraceptive_use',
                    'Mediaexposure',
                    'age']
            value = [3.28409886e-03, 2.45556319e-03, 2.40633496e-01, 1.72564929e-03,
                    1.00915070e-02, 5.26607934e-02, 6.75481399e-02, 6.52129089e-03,
                    8.06661148e-04, 7.73073786e-03, 8.36168538e-03, 3.40182630e-04,
                    1.33735678e-02, 1.23620818e-03, 6.61443354e-04, 3.91043195e-03,
                    7.66269631e-03, 1.41043598e-03, 5.69585410e-01]
            feature_importance_dict = dict(zip(key, value))
            feature_importance(feature_importance_dict)
            st.subheader('Based on the analysis above the age is the dominant factor, we removed age and see which factors are important')
            key =  ['Region',
                    'Rural_Urban',
                    'No_of_Children_under_5',
                    'Age_at_1st_Birth',
                    'BirthOrder',
                    'Prec.BirthInterval',
                    'TotalChildren',
                    'HighestEducation',
                    'SexofChild',
                    'childSize',
                    'WealthQuantile',
                    'prev_termin_pregnancy',
                    'Year',
                    'Toiletfascility',
                    'HandwashFascility',
                    'Place_of_delivery',
                    'Contraceptive_use',
                    'Mediaexposure']
            value = [0.00696809, 0.00781772, 0.52102941, 0.00371032, 0.0249815 ,
                    0.12528112, 0.19942863, 0.01504861, 0.00265252, 0.01656583,
                    0.01842359, 0.00063972, 0.03023446, 0.00284251, 0.00169945,
                    0.00517499, 0.01371196, 0.00378957]
            feature_importance_dict = dict(zip(key, value))
            feature_importance(feature_importance_dict)
        elif model == 'XGBoost':
            key =  ['Region',
                    'Rural_Urban',
                    'No_of_Children_under_5',
                    'Age_at_1st_Birth',
                    'BirthOrder',
                    'Prec.BirthInterval',
                    'TotalChildren',
                    'HighestEducation',
                    'SexofChild',
                    'childSize',
                    'WealthQuantile',
                    'prev_termin_pregnancy',
                    'Year',
                    'Toiletfascility',
                    'HandwashFascility',
                    'Place_of_delivery',
                    'Contraceptive_use',
                    'Mediaexposure',
                    'age']
            value = [0.02048751, 0.02059695, 0.1710822 , 0.0221486 , 0.03821605,
                    0.04928231, 0.04065862, 0.03674648, 0.01840717, 0.02287741,
                    0.02336911, 0.0163441 , 0.03586599, 0.01894557, 0.0187406 ,
                    0.04268981, 0.03703905, 0.02104646, 0.34545597]
            feature_importance_dict = dict(zip(key, value))
            feature_importance(feature_importance_dict)
            st.subheader('Based on the analysis above the age is the dominant factor, we removed age and see which factors are important')
            key =  ['Region',
                    'Rural_Urban',
                    'No_of_Children_under_5',
                    'Age_at_1st_Birth',
                    'BirthOrder',
                    'Prec.BirthInterval',
                    'TotalChildren',
                    'HighestEducation',
                    'SexofChild',
                    'childSize',
                    'WealthQuantile',
                    'prev_termin_pregnancy',
                    'Year',
                    'Toiletfascility',
                    'HandwashFascility',
                    'Place_of_delivery',
                    'Contraceptive_use',
                    'Mediaexposure']
            value = [0.02909926, 0.02814793, 0.33333072, 0.0269317 , 0.05487187,
                    0.07157773, 0.06127196, 0.05252262, 0.02581846, 0.0376559 ,
                    0.03707837, 0.02400167, 0.03929391, 0.02809054, 0.02526036,
                    0.0332862 , 0.05963827, 0.03212255]
            feature_importance_dict = dict(zip(key, value))
            feature_importance(feature_importance_dict)

def main():
    st.sidebar.title('Select a page')
    dropdown = st.sidebar.selectbox('Pages', ['Home', 'Data Highlights','EDA','Strategies','Results' ,'Predcit','About Us'])
    df = all_data.read_data()
    if dropdown == 'Home':
        st.image('./assets/Logo.jpg', width=300,  use_column_width=True)
        st.title('Under 5 Mortality ')
        st.markdown('This app shows the under 5 mortality rate in the Ethiopia based on EDHS dataset.')
        st.markdown('**Data Source:** [World Health Organization](https://dhsprogram.com/)')
        st.subheader('Background')
        st.write("""
        A substantial reduction in Child mortality has been achieved across the globe.
        Ethiopia too has made good progress in reducing child mortality since 1990.
        The overall Under-five mortalities has reduced from about 203 to 67 per thousand 
        live birth, showing a reduction of about 66%. Despite the remarkable progress in
         mortality reduction for Under-five children, strong and coordinated measures shall 
         be taken to meet the SDG target
        """)
        st.subheader('Objective')
        st.write("""
        Various studies have been conducted for the case of Ethiopia. In this study,
         we aim to identify various Socioeconomic, Demographic, WASH (Environmental) 
         and clinical parameters as attributing factors for Child mortality using advanced
          Machine Learning algorithms. IT aims mainly in identify feature-importance of these risk factors.
           Therefore, this study can be helpful for policymakers to design effective multi-sectoral interventions
           to reduce child mortality rate.
        """)




    elif dropdown == 'Data Highlights':
        df = all_data.read_data()
        btn = st.selectbox('Select an what to see about the data', ["Baisc information", 'Years', 'Study Selected features'])
        if btn == "Baisc information":
            st.dataframe(df.head(20))
            # adv_btn = st.checkbox('Show Advanced Information')
            # if adv_btn:
            #     st.subheader('Advanced Information')
            #     st.table(df.describe())
            #     st.subheader('Missing values')
                
        elif btn == 'Years':
            st.subheader('The data is available for the years 2000 - 2019')
            st.success('**2000               Used**')
            st.success('**2005               Used**')
            st.success('**2011               Used**')
            st.error('**2014               Not used**')
            st.success('**2016               Used**')
            st.warning('**2019 -                mini Edhs Used**')
        elif btn == 'Study Selected features':
            # st.markdown('The selected features are the following:')
            # display_features(features_selected())
            st.subheader('Coming soon...')
            st.warning("Please contact EPHI....")
            st.markdown('**Ethiopian Public Health Institue:** [EPHI](https://ephi.com/)')   
    elif dropdown == 'Strategies':
        st.markdown('### Strategies')
        if st.checkbox('Imputation of missing values'):
            st.markdown('#### Imputation of missing values')
            st.markdown('The missing values were imputed using the two baisc strategies and the corresping results were provided below:')
            result = [
            ["Imputation Strategy", "Logistic Regression accuracy", "Random forest classifier score" , "XGBOOST","Neural Network result" ], 
            ["Ordinary Mean | Median","0.0  0.0", "0.0  0.0" , "0.0  0.0" , "0.0  0.0"],
            ["Iterative Imputer", "0.0", "0.0","0.0  0.0", "0.0  0.0",  "0.0  0.0"]]
            st.table(result)
            st.warning('**Note:** You can see the VIF score below')
            st.image('./assets/VIF.png', width=300,  use_column_width=True)
        if st.checkbox('Balancing strategies used'):
            balancing_strategies = ["Over_sample", "Under_sample", "SMOTE", "ADASYN"]
            st.markdown('#### Since the target variable is so much unbalanced we used these strategies to balance the data')
            for i in balancing_strategies:
                st.markdown(i)   
        if st.checkbox('Models used'):
            models = [["Logistic Regression accuracy", " - ", {0:90, 1:89}, "guide"], ["Random Forest classfier", " - ", {0:90, 1:89}, "guide"]
            , ["XGBOOST", " - ", {0:90, 1:89}, "guide"], ["Neural Network", " - ", {0:90, 1:89}, "guide"]]
            st.markdown('#### Models used')
            for i in models:
                print(i)
                display_models(i)      
    elif dropdown == 'Results':
        st.markdown('### Results')
        show_results()
    elif dropdown == 'Predcit':
        st.write('### Predcit')
        loaded_model = joblib.load('./model/classifier_model')
        # create a testvar and put the first element in df to testvar
        age = st.slider('Age in months', 0, 60, 1)
        total_childeren = st.slider('Total Childerens', 0, 6, 1)
        place_of_delivery = st.selectbox('Place of delivery', ['Delivery at Health Fascilities', 'Delivery outside of Health Fascilities'])
        if place_of_delivery == 'Delivery at Health Fascilities':
            place_of_delivery = 1
        elif place_of_delivery == 'Delivery outside of Health Fascilities':
            place_of_delivery = 0
        contraceptive_use = st.selectbox('Contraceptive_use', ['No', 'Yes'])
        if contraceptive_use == 'No':
            contraceptive_use = 0
        elif contraceptive_use == 'Yes':
            contraceptive_use = 1
        preciding_bith_interval = st.slider('Preciding_bith_interval in month', 9, 200, 1)
        highest_ed = st.selectbox('Highest Education', ['No education', 'Primary', 'Secondary', 'Higher'])
        if highest_ed == 'No education':
            highest_ed = 0
        elif highest_ed == 'Primary':
            highest_ed = 1
        elif highest_ed == 'Secondary':
            highest_ed = 2
        else:
            highest_ed = 3
        if st.button('Predict'):
            testvar = [age, total_childeren, place_of_delivery, contraceptive_use, preciding_bith_interval, highest_ed]
            testvar = np.array(testvar).reshape(1, -1)
            st.write(loaded_model.predict(testvar))
            st.write(loaded_model.predict_proba(testvar))
    elif dropdown == 'EDA':
        st.markdown('### Exploratory Data Analysis')
        correlation_matrix(all_data.read_corr())
    elif dropdown == 'About Us':
        st.markdown('### About')
main()


