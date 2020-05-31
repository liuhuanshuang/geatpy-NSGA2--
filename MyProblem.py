import numpy as np
import geatpy as ea
import random

'''
1. 工序链中每个工件的工序总数有限制
2. 若要用多染色体，则使用多个种群、并把每个种群对应个体关联起来即可

'''

class MyProblem(ea.Problem):
	def __init__(self,steps=[3,1,4,2,3,2,2,2,2,1]): # steps=[2,3,3,4]工件的工序数
		self.a = np.array([
			[3,4,5,14],
			[4,5,7,14],
			[5,6,7,9],
			[5,7,14,14],
			[4,14,14,14],
			[5,7,14,14],
			[3,4,6,14],
			[5,6,8,14],
			[4,5,14,14],
			[6,7,8,9],
			[6,8,10,14],
			[3,5,6,7],
			[5,6,8,9],
			[3,5,7,14],
			[4,6,6,7],
			[4,14,14,14],
			[5,7,8,14],
			[4,6,14,14],
			[5,6,7,7],
			[6,7,7,8]
			])
		self.n = 10  # 工件数
		self.m = 4  # 机器数
		self.H = np.array([3,1,4,2,3,2,2,2,2,1]) # 工件的工序数
		self.H_1 = np.cumsum(self.H) - self.H #[1 3 6 9]
		self.sumH = sum(self.H)
		self.xianbiankucun = 15  #线边库存
		self.yunshufeiyong = 400 #运输费用
		self.lilv = 0.0035
		self.tiaozhengshijian = np.array([0.2,0.3,0.15,0.2]) # 调整时间
		self.yuancailiaofei = np.array([2400,3000,2800,4000]) # 原材料费
		self.jiaohuoqi = np.array([15,24,25,30]) # 交货期  小时
		self.tiqianchengfa = np.array([30,30,30,30]) # 提前惩罚 元/小时
		self.tuoqichengfa = np.array([60,70,50,50])  # 拖期惩罚 元/小时

		self.yunshushijian = np.array([0.15,0.2,0.25,0.3]) # 运输时间 小时

		self.gongshifei = np.array([60,30,50,40,30])  # 机器工时费
		self.gongrenfei = np.array([25,30,25,25,25])  # 工人费

		name='MyProblem'
		M=2
		maxormins=[1]*M # f1成本最低 f2时间最短 
		Dim=2*sum(steps)  # 工序编码 机器编码
		varTypes=[1]*Dim
		lb=[1]*Dim
		ub=[self.n]*(self.sumH)+[self.m]*(self.sumH)
		lbin=[1]*Dim
		ubin=[1]*Dim
		ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

	def aimFunc(self,pop): #目标函数
		x = pop.Phen.astype(int)
		init_step = [1,1,1,2,3,3,3,3,4,4,4,5,5,6,6,7,7,8,8,9,9,10]
		for i in range(40):
			np.random.shuffle(init_step)
			x[i][:self.sumH] = init_step
		process = x[:,:self.sumH]
		machine = x[:,self.sumH:]
		f1 = np.zeros(np.size(process,0))
		f2 = np.zeros(np.size(process,0))
		for population in range(np.size(process,0)):
			try:
				step_1 = self.H_1
				time_array = np.zeros((self.sumH,self.m),dtype=int)
				for i in range(np.size(process,1)):
					time_array[step_1[process[population][i]-1], machine[population][i]] = 1
					step_1[process[population][i]-1] +=1
				# 计算时间
				f1[population] = max(np.sum(time_array*self.a, axis=0))
				# 计算成本
				f2[population] = np.sum((time_array*self.a)*(self.gongshifei+self.gongrenfei))
			except:
				f1[population] = 10000000000
				f2[population] = 10000000000

		

		# 可行性法则约束工序数  行代表个体 列代表约束条件  CV[1]>0 代表1号个体不满足约束条件
		pop.CV = np.abs(np.abs(self.sumH-np.count_nonzero(process-1,axis=1)-self.H[0])+ 
			np.abs(self.sumH-np.count_nonzero(process-2,axis=1)-self.H[1])+
			np.abs(self.sumH-np.count_nonzero(process-3,axis=1)-self.H[2])+
			np.abs(self.sumH-np.count_nonzero(process-4,axis=1)-self.H[3])+
			np.abs(self.sumH-np.count_nonzero(process-5,axis=1)-self.H[4])+
			np.abs(self.sumH-np.count_nonzero(process-6,axis=1)-self.H[5])+
			np.abs(self.sumH-np.count_nonzero(process-7,axis=1)-self.H[6])+
			np.abs(self.sumH-np.count_nonzero(process-8,axis=1)-self.H[7])+
			np.abs(self.sumH-np.count_nonzero(process-9,axis=1)-self.H[8])+
			np.abs(self.sumH-np.count_nonzero(process-10,axis=1)-self.H[9]))
		pop.CV = np.reshape(pop.CV,(np.size(process,0),1))

		pop.ObjV = np.vstack([f1,f2]).T