#coding=utf-8
import pandas as pd
root = '/home/zsc/workspace/git/Tencent_CTR_Competition/data/'

train_df = pd.read_csv(root+"train.csv")
test_df=  pd.read_csv(root+'test.csv')

train_df['source']='train'
test_df['source']='submit'

data = pd.concat([train_df,test_df],axis=0)

print data.head()


#训练集和测试集的数据分别从23 - 28(预测28号的label), 24-29(预测29号的label)
dfTrain = data[data.clickTime.between(230000,272359)]
dfTest = data[data.clickTime.between(240000,292359)]

dfSubmit = data[data.source=='submit']



