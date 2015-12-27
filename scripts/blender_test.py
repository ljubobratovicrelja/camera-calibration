#!/usr/bin/python3

import os
import bpy
from mathutils import Matrix


def readKs(path):
	with open(path, 'r') as f:
		data = f.read().split("\n")
		#a_data = data[0].split(",")
		#k_data = data[1].split(",")
		ks_data = []
		for i in range(2, len(data)-1):
			K_data = data[i].split(",")
			m = Matrix()
			idx = 0
			for j in range(0, 3):
				for k in range(0, 4):
					m[j][k] = float(K_data[idx])
					idx += 1
			ks_data.append(m)
			print(m)
		
		return ks_data
	
m = readKs("/home/relja/git/camera_calibration/a.out")
bpy.data.objects['camera_test'].matrix_local = m[0]
bpy.data.objects['camera_test'].scale = [1, -1, -1]