import pandas as pd
import os 
import numpy as np

# Encoder libraries 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def read_json_to_df(location):
    """
    Gets location of the json file and converts it to a pd.DataFrame 
    :param:location:
    :type:str
    :Return:rtn: pd.DataFrame 
    """
    with open(location) as f:
        f.readlines(1)
        df = pd.read_json(f) 
        name=f.readlines(0)
        df.style.set_caption(str(name))
    return df

def read_folder_json(DataFolder):
    '''
    Reads the names of the files and craetes 8 data Frames 
    :Param: DataFolder(str): The location of the data 

    '''
    List_File_Names=os.listdir(DataFolder)
    file_types=['MEDICATIONS','VITALS','DIAGNOSES','LABS','ENCOUNTERS','PROCEDURES','HHSFINANCIALDATA','DEMOGRAPHICS']
    Column_list=[['PATIENTHASHMRN','MEDDATEDIFFNO','GENERICNAME',  'MEDNAME', 'PHARMACEUTICALCLASS', 'THERAPEUTICCLASS'],
                ['PATIENTHASHMRN', 'VITALDATEDIFFNO','UNIT', 'VALUE', 'VALUETYPE', 'VITALNAME'],
                ['PATIENTHASHMRN','ENCDATEDIFFNO','ENCOUNTERHASHKEY','CHRONIC',   'ICD10CODE', 'ICD10DESCRIPTION', 'ISPRIMARY'],
                ['PATIENTHASHMRN','LABDATEDIFFNO', 'LAB_NAME', 'ORDERINGDEPTSPECIALTY', 'RESULTNUMERICVALUE', 'RESULTSTATUS', 'RESULTUNIT', 'RESULTVALUE','SOURCETYPE'],
                ['PATIENTHASHMRN','ENCDATEDIFFNO', 'ENCOUNTERHASHKEY','ADMISSIONTYPE', 'DEPTLOCATION', 'DEPTSPECIALTY',  'TYPE', 'VISITTYPE','ADMITDEPTSPECIALTY'],
                ['PATIENTHASHMRN','PROCDATEDIFFNO','CODESET',   'PROCEDURECODE','PROCEDURENAME', 'QUANTITY', 'TYPE'],
                ['PATIENTHASHMRN','MEDIANINCOME','ASTHMAHOSPRATE', 'FINANCIALCLASS', 'HHSREGION', 'POPULATION'],
                ['PATIENTHASHMRN','AGE', 'ETHNICITY', 'MARITALSTATUS', 'RACE', 'SEX', 'SMOKINGSTATUS', 'STATUS']]

    for i, type in enumerate(file_types):
        Files=[x for x in  List_File_Names if type in x ]
        # print(Files)
        df_list=[]
        for ii, name in enumerate(Files):
            file=DataFolder+name
            if ii==0:
                exec('df_'+ type + " = read_json_to_df(file)")          
            exec('df_'+ type + " = read_json_to_df(file)")
            exec('df_'+ type + ' = df_'+ type +'.reindex(Column_list[i], axis=1)')
            # exec('print(type,df_'+ type +'.columns)')
            index_with_date = next((i for i, s in enumerate(Column_list[i]) if "DATE" in s), None)
            if "DATE" in str(Column_list[i]):
                exec('df_'+type+'.rename(columns={"'+str(Column_list[i][index_with_date])+'": "DATE"}, inplace=True)')
            exec('df_list.append(df_'+ type+')')
        exec('global d_'+type)     
        exec('d_'+type+' = pd.concat(df_list, axis=0)')
    # return df_MEDICATIONS,df_VITALS,df_DIAGNOSES,df_LABS,df_ENCOUNTERS,df_PROCEDURES,df_HHSFINANCIALDATA,df_DEMOGRAPHICS
# Seperate Blood Presure 
def split_BP(data_frame,function='max'):
    high_list=[]
    low_list=[]
    if 'BP' in data_frame.columns:
        for value in data_frame['BP']:
            High=None
            Low=None
            if not (pd.isna(value) or value==''):
                if function=="max":
                    bp=sorted(value.split(' '))[-1]
                elif function=="min":
                    bp=sorted(value.split(' '))[0]
                High,Low=bp.split('/')
            high_list.append(High)
            low_list.append(Low)
        data_frame['High_Pressure']=pd.to_numeric(high_list)
        data_frame['Low_Pressure']=pd.to_numeric(low_list)

        data_frame=data_frame.drop('BP',axis=1)
    return data_frame

def singulate_data(data_frame):
    # Reduce multi values in the data 
    split_values = data_frame['BMI (Calculated)'].str.split(' ')
    data_frame['BMI (Calculated)']=split_values.str[0]

    split_values = data_frame['Pulse'].str.split(' ')
    data_frame['Pulse']=split_values.str[0]

    split_values = data_frame['Resp'].str.split(' ')
    data_frame['Resp']=split_values.str[0]

    split_values = data_frame['SpO2'].str.split(' ')
    data_frame['SpO2']=split_values.str[0]

    split_values = data_frame['Temp'].str.split(' ')
    data_frame['Temp']=split_values.str[0]

    split_values = data_frame['Weight'].str.split(' ')
    data_frame['Weight']=split_values.str[0]

    split_values = data_frame['Height'].str.split(' ')
    data_frame['Height']=split_values.str[0]

    split_values = data_frame['O2 Flow (L/min)'].str.split(' ')
    data_frame['O2 Flow (L/min)']=split_values.str[0]

    split_values = data_frame['O2 Flow Rate (L/min)'].str.split(' ')
    data_frame['O2 Flow Rate (L/min)']=split_values.str[0]

    split_values = data_frame['Respiration'].str.split(' ')
    data_frame['Respiration']=split_values.str[0]

    data_frame['BMI (Calculated)'] = pd.to_numeric(data_frame['BMI (Calculated)'])
    data_frame['Pulse'] = pd.to_numeric(data_frame['Pulse'])
    data_frame['Resp'] = pd.to_numeric(data_frame['Resp'])
    data_frame['SpO2'] = pd.to_numeric(data_frame['SpO2'])
    data_frame['Temp'] = pd.to_numeric(data_frame['Temp'])
    data_frame['Weight'] = pd.to_numeric(data_frame['Weight'])
    data_frame['Height']=pd.to_numeric(data_frame['Height'])
    data_frame['O2 Flow (L/min)']=pd.to_numeric(data_frame['O2 Flow (L/min)'])
    data_frame['O2 Flow Rate (L/min)']=pd.to_numeric(data_frame['O2 Flow Rate (L/min)'])
    data_frame['Respiration']=pd.to_numeric(data_frame['Respiration'])
    return data_frame

# VITALS_pivot_clean=split_BP(VITALS_pivot)
# VITALS_pivot_clean=singulate_data(VITALS_pivot_clean)
def one_hot_encoder_create(Data):
    ''' 
    Function crates integer and one_hot encoder with the data and returns both of the encoders
    :Param:Data: Seri, array or list
    :return:One_encoder:
    :return:Int_encoder 
    '''
    data = np.array(Data)
    # integer encode
    Int_encoder = LabelEncoder()
    integer_encoded = Int_encoder.fit_transform(data)
    # binary encode
    One_encoder =  OneHotEncoder(sparse=False)
    data_integer_encoded = integer_encoded.reshape(len(data), 1)
    Data_onehot_encoded =  One_encoder.fit_transform(data_integer_encoded)
    return One_encoder,Int_encoder

def one_hot_encoder(Data, One_encoder,Int_encoder):
    ''' 
    Function converts data into One_hot encode version
    :Param:Data: Seri, array or list
    :Param:One_encoder:
    :Param:Int_encoder:

    :Return: rtn: Encoded data 
    '''
    one_hot=One_encoder.transform(Int_encoder.transform(Data).reshape(-1,1))

    rtn=np.where(one_hot==0,None,one_hot )
    return rtn

def reverse_one_hot_encoder(Data, One_encoder,Int_encoder):
    ''' 
    Function reconverts  One_hot encode to original form
    :Param:Data: Seri, array or list
    :Param:One_encoder:
    :Param:Int_encoder:

    :Return: rtn: decoded data 
    '''
    
    Data=np.where(Data==None,0,Data )

    rtn=Int_encoder.inverse_transform(np.argmax(Data, axis=1))
    return rtn