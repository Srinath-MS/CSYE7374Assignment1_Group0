import pandas as pd
wc = pd.read_csv('/Users/shwetaanchan/Desktop/Midterm/Adult_Data.csv’)
df = pd.DataFrame(wc)
dummy = pd.get_dummies(df,columns=['Workclass','Education','Maritalstatus','occupation','Relationship','Race','Sex','Country'])
dummy.to_csv(‘Income_Class.csv',sep=',')