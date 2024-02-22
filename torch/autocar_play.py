import socket
import struct
import torch,gym
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

buffer_size = 1024
sent_data_size = 32

server_cmd_train = 10
server_cmd_save = 11
server_cmd_end = 12
server_cmd_selective_train = 13

client_cmd_action = 0
client_cmd_reset = 1

running = True
state_d = 17
action_d = 2

first_layer=200
second_layer=360

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

class ActorNet(nn.Module):
	"""docstring for ActorNet"""
	def __init__(self, state_dim,action_dim):
		super(ActorNet, self).__init__()

		self.model = nn.Sequential(*[
			nn.Linear(state_dim, first_layer),
			nn.ReLU(inplace=True),
			nn.Linear(first_layer, second_layer),
			nn.ReLU(inplace=True),
			nn.Linear(second_layer,action_dim),
			nn.Tanh()
			])

	def forward(self, state):
		return self.model(state)

def other_operation(cmd):
	global running
	if cmd==server_cmd_save:
		pass
	elif cmd == server_cmd_end:
		running = False
	else:
		print('invalid cmd:'+str(cmd))
		running = False

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
#========main=======

actor_net = ActorNet(state_d,action_d)
if os.path.exists('actor.nnp'):
	print('loading nn')
	
	actor_net.load_state_dict(torch.load('actor.nnp'))
	print('load nn success')
else:
	print('nn file not found : actor.nnp')
	exit()

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(('localhost',6666))
server.listen(1)

print('waiting for unity...')
#=====main=========
connect, addr = server.accept()
print('unity client conneted from:', addr)

isTraining=0
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

		#receive reset state
		state,offset = read_state(data,offset)
		reward,offset = read_float(data,offset)
		done,offset = read_int(data,offset)	
		
		ep_reward = 0
		while done==0 and running:
			s = torch.FloatTensor(state)
			action = actor_net(s)
			a = action.detach().numpy()

			a[0] = explore_action([0,1,2],a[0])
			a[1] = explore_action([0,1],a[1])

			#send action to unity
			offset = write_int(client_cmd_action,data_buffer,0)
			offset = write_action(a.astype(int),data_buffer,offset)
			connect.send(data_buffer)
			
			#receive state,reward
			data = connect.recv(buffer_size)
			offset = 0	
			cmd,offset = read_int(data, offset)
			if cmd ==server_cmd_train or cmd == server_cmd_selective_train:
				obs,offset = read_state(data,offset)
				reward,offset = read_float(data,offset)
				done,offset = read_int(data,offset)	

				state=obs
				#ep_reward += reward
				ep_reward += 1
				if done>0:
					print(' time: %.2f'%(ep_reward/20.0))

					offset = write_int(client_cmd_reset,data_buffer,0)
					offset = write_int(isTraining,data_buffer,offset)
					connect.send(data_buffer)
			else:
				other_operation(cmd)				
		
	else:
		other_operation(cmd)

connect.close()
server.close()