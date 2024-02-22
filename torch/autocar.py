import socket
import struct
import torch,gym
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt

buffer_size = 1024
sent_data_size = 32

server_cmd_train = 10
server_cmd_save = 11
server_cmd_end = 12
server_cmd_selective_train = 13
server_cmd_reset_training = 14

client_cmd_action = 0
client_cmd_reset = 1
map_segment_index = 0

save_eps=[50,100,150]
auto_save_interval = 100

good_reward = 30
good_data_ratio = 0.3

isTraining=1

fig = plt.figure()
actor_loss_plot = fig.add_subplot(3,1,1)
critic_loss_plot = fig.add_subplot(3,1,2)
reward_plot = fig.add_subplot(3,1,3)
# q_plot = fig.add_subplot(2,2,4)

actor_loss_plot.set_title('actor loss')
critic_loss_plot.set_title('critic loss')
reward_plot.set_title('reward')
# q_plot.set_title('q value mean')

actor_loss_data = []
critic_loss = []
rewards = []
# q_mean = []

def read_state(buffer,offset):
	return struct.unpack_from('17f',buffer,offset),offset+68

def read_vector(buffer, offset):
	return struct.unpack_from('ff',buffer, offset),offset+8

def write_vector(vec, buffer, offset):
	struct.pack_into('2f',buffer,offset,*vec)
	return offset+8

def read_sensor(buffer, offset):
	return struct.unpack_from('12f',buffer,offset),offset+48

def read_float(buffer, offset):
	return struct.unpack_from('f',buffer,offset)[0],offset+4

def write_float(val,buffer,offset):
	struct.pack_into('f',buffer,offset,val)
	return offset+4

def read_int(buffer,offset):
	return struct.unpack_from('i',buffer,offset)[0],offset+4

def write_int(val,buffer,offset):
	struct.pack_into('i',buffer,offset,val)
	return offset+4

def write_action(a, buffer,offset):
	struct.pack_into('2i',buffer,offset,*a)
	return offset+8

###params
lr_actor = 6e-4
lr_critic = 1e-3
gamma = 0.9 # reward discount
memory_capacity = 10000
min_training_mem = 1
batch_size = 32
tau = 0.001 #soft replacement

actor_net_file_prefix = 'actor_'
actor_target_net_file_prefix = 'actor_target_'
critic_net_file_prefix = 'critic_'
critic_target_net_file_prefix = 'critic_target_'

first_layer=200
second_layer=360
###ddpg
class ActorNet(nn.Module):
	"""docstring for ActorNet"""
	def __init__(self, state_dim,action_dim):
		super(ActorNet, self).__init__()

		self.model = nn.Sequential(*[
			nn.Linear(state_dim, first_layer),
			nn.ReLU(),
			nn.Linear(first_layer,second_layer),
			nn.ReLU(),
			nn.Linear(second_layer,action_dim),
			nn.Tanh()
			])

	def forward(self, state):
		return self.model(state)

		
class CriticNet(nn.Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()
		self.s_net = nn.Sequential(*[
			nn.Linear(state_dim, first_layer),
			nn.ReLU(inplace=True),
			nn.Linear(first_layer,second_layer),
			])
		self.a_net = nn.Linear(action_dim, second_layer)
		
		self.out_net = nn.Linear(second_layer, 1)

	def forward(self, state, action):
		s = self.s_net(state)
		a = self.a_net(action)
		mix = torch.relu(s+a)
		q_val = self.out_net(mix)
		return q_val

class DDPGNet(object):
	"""docstring for DDPGNet"""
	def __init__(self, state_dim, action_dim):
		super(DDPGNet, self).__init__()
		self.s_dim,self.a_dim = state_dim,action_dim

		self.actor_net = ActorNet(state_dim,action_dim)
		self.actor_target_net = ActorNet(state_dim,action_dim)

		self.critic_net = CriticNet(state_dim,action_dim)
		self.critic_target_net = CriticNet(state_dim,action_dim)

		self.action_opt = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)
		self.critic_opt = torch.optim.Adam(self.critic_net.parameters(), lr=lr_critic)

		self.memory = np.zeros((memory_capacity,state_dim*2+action_dim+2))
		self.good_memories = np.zeros((memory_capacity,state_dim*2+action_dim+2))
		self.good_memory_index = 0
		self.good_memory_amount = 0
		self.memory_index = 0
		self.memory_amount = 0

	def choose_action(self, state):
		s = torch.FloatTensor(state)
		action = self.actor_net(s)
		return action.detach().numpy()

	def learn(self):		
		#soft replace
		for target_param, eval_param in zip(self.actor_target_net.parameters(),self.actor_net.parameters()):
			target_param.data.copy_(target_param.data*(1-tau)+eval_param.data*tau)

		for target_param, eval_param in zip(self.critic_target_net.parameters(),self.critic_net.parameters()):
			target_param.data.copy_(target_param.data*(1-tau)+eval_param.data*tau)

		#多组数据batch是为了加快训练速度

		if self.memory_amount>batch_size:
			if self.good_memory_amount<batch_size*good_data_ratio:
				good_data = self.good_memories[:self.good_memory_amount,:]
				indices = np.random.choice(self.memory_amount, size=batch_size-self.good_memory_amount)
			else:
				good_data_indices = np.random.choice(self.good_memory_amount, size=int(batch_size*good_data_ratio))
				good_data = self.good_memories[good_data_indices,:]
				indices = np.random.choice(self.memory_amount, size=int(batch_size*(1-good_data_ratio)))
			normal_data = self.memory[indices,:]
			batch_data=np.concatenate((normal_data,good_data))
		else:
			indices = np.random.choice(self.memory_amount, size=batch_size)
			batch_data = self.memory[indices,:]

		states = torch.FloatTensor(batch_data[:,:self.s_dim])
		actions = torch.FloatTensor(batch_data[: ,self.s_dim:self.s_dim+self.a_dim])
		rewards = torch.FloatTensor(batch_data[:,-self.s_dim-2:-self.s_dim-1])
		next_states = torch.FloatTensor(batch_data[:,-self.s_dim-1:-1])
		is_done = torch.FloatTensor(batch_data[:,-1:])

		a_target = self.actor_target_net(next_states)
		q_ = self.critic_target_net(next_states,a_target)
		
		q_target = rewards + gamma*q_
		#出界/边缘直接舍弃累加，确定q值，这样边缘梯度就会比中间小
		for k in range(len(batch_data)):
			if is_done[k] > 0:
				q_target[k] = rewards[k]		
		

		a = self.actor_net(states)
		q = self.critic_net(states,a)
		a_loss = -torch.mean(q)

		actor_loss_data.append(a_loss.detach().numpy())

		self.action_opt.zero_grad()
		a_loss.backward()
		self.action_opt.step()
		
		q_eval = self.critic_net(states,actions)


		td_error = F.mse_loss(q_eval,q_target)

		critic_loss.append(td_error.detach().numpy())

		self.critic_opt.zero_grad()
		td_error.backward()
		self.critic_opt.step()

	def store_transition(self, state, action, reward, next_state, done):
		global memory_capacity
		transition = np.hstack((state,action,[reward],next_state,[done]))
		self.memory_index = self.memory_index % memory_capacity
		self.memory[self.memory_index, :] = transition		
		self.memory_index += 1
		if self.memory_amount<memory_capacity:
			self.memory_amount += 1

	def store_good_transition(self, state, action, reward, next_state, done):
		global memory_capacity
		transition = np.hstack((state,action,[reward],next_state,[done]))
		self.good_memory_index = self.good_memory_index % memory_capacity
		self.good_memories[self.good_memory_index, :] = transition		
		self.good_memory_index += 1
		if self.good_memory_amount<memory_capacity:
			self.good_memory_amount += 1

def other_operation(cmd):
	global map_segment_index
	global episode
	global running
	def save(root_folder=''):
		root = ''
		if root_folder:
			root='%s\\'%root_folder
			if not os.path.exists(root_folder):
				os.mkdir(root_folder)
		folder_name = '%sep_%i'%(root,episode)
		if not os.path.exists(folder_name):
			os.mkdir(folder_name)
		torch.save(ddpg.actor_net.state_dict(),folder_name + '\\' + actor_net_file_prefix+str(episode)+'.nnp')
		torch.save(ddpg.actor_target_net.state_dict(),folder_name + '\\' + actor_target_net_file_prefix+str(episode)+'.nnp')
		torch.save(ddpg.critic_net.state_dict(),folder_name + '\\' + critic_net_file_prefix+str(episode)+'.nnp')
		torch.save(ddpg.critic_target_net.state_dict(),folder_name + '\\' + critic_target_net_file_prefix+str(episode)+'.nnp')
		print('saved nn to \\'+folder_name)
	if cmd==server_cmd_save:
		save()
	elif cmd == server_cmd_end:
		running = False
	elif cmd == server_cmd_reset_training:
		randomness=1
		episode=0
	else:
		print('invalid cmd:'+str(cmd))
		running = False

def OUNoise(x,mu,theta,sigma):
	return theta*(mu-x)+sigma*np.random.randn(1)

def explore_action(action_collection,original_action):
	original_action = np.clip(original_action,-1,1)
	original_action = (original_action+1)*0.5
	p = action_collection[-1]-action_collection[0]
	original_action = action_collection[-1] + p * original_action
	per = p / len(action_collection)
	base = action_collection[-1]
	for x in action_collection:
		base+=per
		if original_action<base:
			return x

	return action_collection[-1]
#=======setup server=====
running = True
state_d = 17
action_d = 2
ddpg = DDPGNet(state_d,action_d)
episode = 1
brake_possibiliy = 0.1

if os.path.exists('net'):
	folder_name = 'net\\'
	if os.path.exists(folder_name + '\\' + 'actor.nnp') and \
	os.path.exists(folder_name + '\\' + 'actor_target.nnp') and \
	os.path.exists(folder_name + '\\' + 'critic.nnp') and \
	os.path.exists(folder_name + '\\' + 'critic_target.nnp'):

		print('loading nn')
		
		ddpg.actor_net.load_state_dict(torch.load(folder_name + '\\' + 'actor.nnp'))
		ddpg.actor_target_net.load_state_dict(torch.load(folder_name + '\\' + 'actor_target.nnp'))
		ddpg.critic_net.load_state_dict(torch.load(folder_name + '\\' + 'critic.nnp'))
		ddpg.critic_target_net.load_state_dict(torch.load(folder_name + '\\' + 'critic_target.nnp'))
		print('load nn success')

randomness = 1
r_epsil = randomness

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(('localhost',6666))
server.listen(1)

print('waiting for unity...')
#=====main=========
connect, addr = server.accept()
print('unity client conneted from:', addr)

data_buffer=bytearray(sent_data_size)
#send reset
offset = write_int(client_cmd_reset,data_buffer,0)
offset = write_int(isTraining,data_buffer,offset)
connect.send(data_buffer)

while running:
	data = connect.recv(buffer_size)
	offset = 0	
	cmd,offset = read_int(data, offset)
	if cmd ==server_cmd_train or cmd == server_cmd_selective_train:
		# sensor_data,offset = read_sensor(data,offset)
		# speed,offset = read_vector(data,offset)
		# angular_speed,offset = read_float(data,offset)

		#receive reset state
		state,offset = read_state(data,offset)
		reward,offset = read_float(data,offset)
		done,offset = read_int(data,offset)	
		#print('state',state,'reward',reward)
		
		ep_reward = 0
		frame = 0
		while done==0 and running:
			a = ddpg.choose_action(state)
			# randomness
			noise = np.zeros([1,action_d])
			noise[0][0]=OUNoise(a[0],0.0,0.58,0.34)
			noise[0][1]=OUNoise(a[1],0.5,1.0,0.1)

			# break for curve:
			if random.random()<=brake_possibiliy:
				noise[0][1] = OUNoise(a[1],-0.1,1.0,0.05)

			alpha = max(randomness/r_epsil,0)
			a[0]+=alpha*noise[0][0]
			a[1]+=alpha*noise[0][1]


			a[0] = explore_action([0,1,2],a[0])
			a[1] = explore_action([0,1],a[1])			
			

			#send action to unity
			offset = write_int(client_cmd_action,data_buffer,0)
			offset = write_action(a.astype(int),data_buffer,offset)
			connect.send(data_buffer)
			
			#print('action', a)
			
			#receive state,reward
			cmd=-1
			while cmd!=server_cmd_train and cmd != server_cmd_selective_train and running:
				data = connect.recv(buffer_size)
				offset = 0	
				cmd,offset = read_int(data, offset)
				if cmd ==server_cmd_train or cmd == server_cmd_selective_train:
					obs,offset = read_state(data,offset)
					reward,offset = read_float(data,offset)
					done,offset = read_int(data,offset)

					if cmd != server_cmd_selective_train:
						ddpg.store_transition(state,a,reward/10.0,obs,done)
						if reward>=good_reward:
							ddpg.store_good_transition(state,a,reward/10.0,obs,done)

						if ddpg.memory_amount > min_training_mem:
							# 	#积累了足够的数据之后才开始单步训练
							if randomness>0.00001:
								randomness-=0.00003
							else:
								randomness = 0				

					state=obs
					ep_reward += reward
					frame += 1
					if done>0:
						if ddpg.memory_amount > min_training_mem:
							print("episode",episode,' frame: %i reward: %.2f'%(frame, ep_reward),'explore: %.2f'%randomness)
						
						else:
							print("Memory progress: %i/%i"%(ddpg.memory_amount,min_training_mem))

						need_save = False
						if episode in save_eps:
							need_save = True
						elif episode % auto_save_interval==0:
							need_save = True

						if need_save:
							other_operation(server_cmd_save)

					if cmd != server_cmd_selective_train:
						ddpg.learn()
						rewards.append(reward)
						
					if done > 0:						
						episode += 1
						
						offset = write_int(client_cmd_reset,data_buffer,0)
						offset = write_int(isTraining,data_buffer,offset)
						connect.send(data_buffer)

						break
				else:
					other_operation(cmd)
		
	else:
		other_operation(cmd)	

connect.close()
server.close()

actor_loss_plot.plot(actor_loss_data)
critic_loss_plot.plot(critic_loss)
reward_plot.plot(rewards)
plt.show()

