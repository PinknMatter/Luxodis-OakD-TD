# me - this DAT
# oakDeviceOp - the OP which is cooking

import depthai as dai
import numpy as np
from HandFaceTracker import HandFaceTracker as Tracker

def onInitialize(oakDeviceOp, callCount):

	if callCount == 1:
		tracker = Tracker.create(parent.OakProject)		
		oakDeviceOp.store('tracker', tracker)
		oakDeviceOp.storeStartupValue('tracker', None)
	else:
		tracker = oakDeviceOp.fetch('tracker', None)
		
	return tracker.onInitialize()
	
def onInitializeFail(oakDeviceOp):
	parent().addScriptError(oakDeviceOp.scriptErrors())
	return
	
def onReady(oakDeviceOp):
	tracker = oakDeviceOp.fetch('tracker', None)
	tracker.onReady(oakDeviceOp)
	return
	
def onStart(oakDeviceOp):
	return

def whileRunning(oakDeviceOp):
	return

def onDone(oakDeviceOp):
	return

def createPipeline(oakDeviceOp):	
	tracker = oakDeviceOp.fetch('tracker', None)
	return tracker.create_pipeline()
