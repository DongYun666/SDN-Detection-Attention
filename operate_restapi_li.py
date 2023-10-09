import requests
import time
import re
from math import log
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from datetime import datetime
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import joblib

class GetInformation:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def get_switch_id(self):
        url = 'http://' + self.ip + ':' + self.port + '/stats/switches'
        re_switch_id = requests.get(url=url).json()
        switch_id_hex = []
        for i in re_switch_id:
            switch_id_hex.append(hex(i))

        return switch_id_hex

    def get_flow_table(self):
        url = 'http://' + self.ip + ':' + self.port + '/stats/flow/%d'
        list_switch = self.get_switch_id()
        all_flow = []
        for switch in list_switch:
            new_url = format(url % int(switch, 16))
            re_switch_flow = requests.get(url=new_url).json()
            all_flow.append(re_switch_flow)

        return all_flow

    def show_flow(self):
        list_flow = self.get_flow_table()
        for flow in list_flow:
            for dpid in flow.keys():
                dp_id = dpid
                print('switch_id:{0}({1})'.format(hex(int(dp_id)), int(dp_id)))
            for list_table in flow.values():
                for table in list_table:
                    print(table)


class PostOperation:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def post_add_flow(self, dpid=None, cookie=0, priority=0, eth_src=None, eth_dst=None,type='OUTPUT', port='CONTROLLER'):
        url = 'http://' + self.ip + ':' + self.port + '/stats/flowentry/add'
        if eth_src== 'None':
            # 添加的默认流表项数据信息
            data = {
                "dpid": dpid,
                "cookie": cookie,
                "cookie_mask": 0,
                "table_id": 0,
                "priority": priority,
                "flags": 0,
                "actions": [
                    {
                        "type": type,
                        "port": port
                    }
                ]
            }
        else:
            data = {
                "dpid": dpid,
                "cookie": cookie,
                "cookie_mask": 0,
                "table_id": 0,
                "priority": priority,
                "flags": 0,
                "match": {
                    "eth_src": eth_src,
                    "eth_dst": eth_dst
                },
                "actions": [
                        
                    
                ]
            }

        response = requests.post(url=url, json=data)
        if response.status_code == 200:
            print('Successfully Add!')
        else:
            print('Fail!')

    def post_del_flow(self, dpid=None, cookie=0, priority=0,eth_src=None, eth_dst=None):
        url = url = 'http://' + self.ip + ':' + self.port + '/stats/flowentry/delete_strict'
        data = {
            "dpid": dpid,
            "cookie": cookie,
            "cookie_mask": 1,
            "table_id": 0,
            "priority": priority,
            "flags": 1,
            "match": {
                "eth_src": eth_src,
                "eth_dst": eth_dst
            },
            "actions": [
                
            ]
        }

        response = requests.post(url=url, json=data)
        if response.status_code == 200:
            print('Successfully Delete!')
        else:
            print('Fail!')

    def post_clear_flow(self, dpid=None):
        url = 'http://' + self.ip + ':' + self.port + '/stats/flowentry/clear/' + str(dpid)
        response = requests.delete(url=url)
        if response.status_code == 200:
            print('Successfully Clear!')
        else:
            print('Fail!')

class Shannon:      
    def calcShannonEnt(self,dataSet):
        numEntries = len(dataSet) # 样本数
        labelCounts = {} # 该数据集每个类别的频数
        for featVec in dataSet:  # 对每一行样本
            currentLabel = featVec # 该样本的标签
            if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1 
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries # 计算p(xi)
            shannonEnt -= prob * log(prob, 2)  # log base 2
        return shannonEnt
    
    
    def get_shannon(self,nog):                      # nog  number of groups

        data=pd.read_csv('DDoS_data.csv')

        length1 = len(data)

        if length1 <= 50*nog:

            print("Network Normal... ... \n")

        else: 

            for i in range(nog): 

                x = length1 - (i+1)*50
                y = x+50
                list1 = data.iloc[x:y,5]
                list2 = data.iloc[x:y,7]
                ent1 = self.calcShannonEnt(list1)
                ent2 = self.calcShannonEnt(list2)

                if ent1 > 5  and  ent2 < 0.1:
                    print ("DDoS Attacking......",'熵：',ent1,ent2,sep="   ")
                    defense = self.Trace(x)
                    print("DDoS_eth_stc: ",defense[0],"  DDoS_ip_dst: ",defense[1],"\n")
                    return defense
		     
                else:
                    print("Network Normal... ...",'熵：',ent1,ent2,sep="   ")
                    
        return 1
		
                    

    def Trace(self,entx):
        data=pd.read_csv('DDoS_data.csv')
        x = entx
        y = x+50
        ethsrc = data.iloc[x:y,3]          #代替实际应用的边缘交换机端口
        ipdst = data.iloc[x:y,7]
        item1 = self.maxElement(ethsrc)
        ethsrc_max = item1[0][0]
        item2 = self.maxElement(ipdst)
        ipdst_max = item2[0][0]
        return ethsrc_max,ipdst_max
         
        
    def maxElement(self,listA):
        labelCounts = {}
        for featVec in listA:  # 对每一行样本
            currentLabel = featVec # 该样本的标签
            if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1 
        labelmax =  sorted(labelCounts.items(), key=lambda item:item[1], reverse = True)
        return list(labelmax)

