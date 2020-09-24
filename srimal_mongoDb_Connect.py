# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:27:04 2019

@author: Amit Srimal
"""
import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["Sample"]
mycol = mydb["Sam_Coll1"]

for x in mycol.find():
  print(x)
