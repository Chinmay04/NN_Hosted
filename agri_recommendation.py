import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random as rnd

def get_recomm_npk(c_val):
  df=pd.read_csv("Data/crop_recommendation.csv")

  def combine_features(row):
      return str(row['N'])+" "+str(row['P'])+" "+str(row['K'])

  df["combined_features"] = df.apply(combine_features,axis=1)

  cv = CountVectorizer()
  cosim = cosine_similarity(cv.fit_transform(df["combined_features"]))

  def get_lfi(index):
      try: return df[df.index == index]["label"].values[0] 
      except: ''
  def get_inpk(npk):
      npk = list(map(int,npk.split(" ")))
      return [df.index[df.N == npk[0]][0]]+[df.index[df.P == npk[1]][0]]+[df.index[df.K == npk[2]][0]]

  # c_val= input() #crop name input rice etc
  choice_npk=list(zip(df[df.label==c_val]["N"],df[df.label==c_val]["P"],df[df.label==c_val]["K"]))

  try:    npki = get_inpk(input()) #n value p value k value in string format eg '90 30 50'
  except: npki = get_inpk(" ".join(map(str,rnd.choice(choice_npk))))

  try:
    sort_n = sorted(list(enumerate(cosim[npki[0]])),key=lambda x:x[1],reverse=True)[1:][:5]
    sort_p = sorted(list(enumerate(cosim[npki[1]])),key=lambda x:x[1],reverse=True)[1:][:5]
    sort_k = sorted(list(enumerate(cosim[npki[2]])),key=lambda x:x[1],reverse=True)[1:][:5]
  except:
    npki=get_inpk(" ".join(map(str,choice_npk[0])))
    sort_n = sorted(list(enumerate(cosim[npki[0]])),key=lambda x:x[1],reverse=True)[1:][:5]
    sort_p = sorted(list(enumerate(cosim[npki[1]])),key=lambda x:x[1],reverse=True)[1:][:5]
    sort_k = sorted(list(enumerate(cosim[npki[2]])),key=lambda x:x[1],reverse=True)[1:][:5]

  recom_n=" ".join(set(filter(lambda x: x is not None,map(lambda x: get_lfi(x[0]),sort_n))))
  recom_p=" ".join(set(filter(lambda x: x is not None,map(lambda x: get_lfi(x[0]),sort_p))))
  recom_k=" ".join(set(filter(lambda x: x is not None,map(lambda x: get_lfi(x[0]),sort_k))))

  
  return [recom_n, recom_p, recom_k]