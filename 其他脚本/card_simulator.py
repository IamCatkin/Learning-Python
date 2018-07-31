#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Jul 26 18:08:58 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import numpy as np

class Card():
    def __init__(self):
        self.rarity = ''
    
    def updatecard(self,i):
        if i == 0:
            self.rarity = 'UR'
        elif i == 1:
            self.rarity = 'SSR'
        elif i == 2:
            self.rarity = 'SR'
        else:
            self.rarity = 'R'
            
def unpackingcard(ele_cont=False):
    data = []
    if ele_cont is True:
        _ = np.random.randint(20)
        if _ == 0:
            data.append(0)
        elif _ < 5:
            data.append(1)
        else:
            data.append(2)
        rds = np.random.randint(100,size=10)
        for j in rds:
            if j == 0:
                data.append(0)
            elif j < 5:
                data.append(1)
            elif j < 20:
                data.append(2)
            else:
                data.append(3)
    else:
        k = np.random.randint(100)
        if k == 0:
            data.append(0)
        elif k < 5:
            data.append(1)
        elif k < 20:
            data.append(2)
        else:
            data.append(3)
    return sorted(data)
                
def main():
    start = input('是否要十一连抽?(Y or N):')
    cards = []
    if start == 'Y' or start == 'y':
        data = unpackingcard(True)
        for i in data:
            card = Card()
            card.updatecard(i)
            cards.append(card.rarity)
    else:
        data = unpackingcard()
        for i in data:
            card = Card()
            card.updatecard(i)
            cards.append(card.rarity)
    print('你抽奖的结果为：' + repr(cards))
            
if __name__ == '__main__':
	main()