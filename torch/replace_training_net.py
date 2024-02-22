import os
import sys
import re

if __name__ == '__main__':
	folder_name_pattern = r'ep_(\d+)'
	folder = sys.argv[1]
	if not os.path.exists(folder):
		print('invalid folder ', folder)
		exit()

	ep = re.match(folder_name_pattern, folder).groups()
	if len(ep)<=0:
		print('invalide folder name')
		exit()

	ep = ep[0]
	destFolder = 'net'
	if not os.path.exists(destFolder):
		os.mkdir(destFolder)

	file_names = {'actor','actor_target','critic','critic_target'}
	files=[]
	for f in file_names:
		fname = '%s\\%s_%s.nnp'%(folder,f,ep)
		if not os.path.exists(fname):
			print('file not found: %s'%fname)
			exit()
		else:
			files.append((fname,"%s\\%s.nnp"%(destFolder,f)))

	for path in files:
		os.system('copy %s %s'%path)