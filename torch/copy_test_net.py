import os
import sys
import re

folder_name_pattern = r'ep_(\d+)'

if __name__ == '__main__':
	if len(sys.argv)<2:
		print('no folder selected')
		exit()

	folder = sys.argv[1]
	if not os.path.exists(folder):
		print('invalid folder ', folder)
		exit()

	ep = re.match(folder_name_pattern, folder).groups()
	if len(ep)<=0:
		print('invalide folder name')
		exit()

	ep = ep[0]
	src = '%s\\actor_%s.nnp'%(folder, ep)
	dest = 'actor.nnp'

	if not os.path.exists(src):
		print('no actor net file found: %s'%src)
		exit()

	
	print('copy file from %s to %s'%(src,dest))
	os.system('copy %s %s'%(src,dest))
	print('done')