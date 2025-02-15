# me - this DAT
# oakDeviceOp - the OP which is cooking

import depthai as dai

def onInitialize(oakDeviceOp, callCount):
	r = op.TDBD.From_zoo('deeplab_v3_mnv2_256x256', shaves=6, zoo_type="depthai")
	if r.wait < 0:
		raise ValueError("Unable to download.")
	return r.wait

def onInitializeFail(oakDeviceOp):
	parent().addScriptError(oakDeviceOp.scriptErrors())
	return

def onReady(oakDeviceOp):
	return
	
def onStart(oakDeviceOp):
	return

def whileRunning(oakDeviceOp):
	return

def onDone(oakDeviceOp):
	return

def createPipeline(oakDeviceOp):
	# https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_depth#gen2-deeplabv3-on-depthai---depth-cropping	
	blobPath = op.TDBD.From_zoo('deeplab_v3_mnv2_256x256', shaves=6, zoo_type="depthai").path

	blob = dai.OpenVINO.Blob(blobPath)
	# for name,tensorInfo in blob.networkInputs.items(): print(name, tensorInfo.dims)
	INPUT_SHAPE = blob.networkInputs['Input'].dims[:2]
	TARGET_SHAPE = (400,400)
	 if hasattr(parent().par, 'Fps'):
        fps = parent().par.Fps.eval()
    else:
        fps = 30

	
	# This example creates an RGB camera. (CAM_A)

	pipeline = dai.Pipeline()

	# Define source and output
	camRgb = pipeline.create(dai.node.ColorCamera)
	camRgb.setFps(fps)
	camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
	camRgb.setPreviewKeepAspectRatio(False)
	camRgb.setIspScale(2,3)
	camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

	# for deeplabv3
	camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
	camRgb.setPreviewSize(*INPUT_SHAPE)
	camRgb.setInterleaved(False)

	# NN output linked to XLinkOut
	xOutISP = pipeline.create(dai.node.XLinkOut)
	xOutISP.setStreamName('cam')
	camRgb.isp.link(xOutISP.input)

	# Define a neural network that will make predictions based on the source frames
	detectionNN = pipeline.create(dai.node.NeuralNetwork)
	detectionNN.setBlob(blob)
	detectionNN.input.setBlocking(False)
	detectionNN.setNumInferenceThreads(2)
	camRgb.preview.link(detectionNN.input)

	# NN output linked to XLinkOut
	xOutNN = pipeline.create(dai.node.XLinkOut)
	xOutNN.setStreamName('nn')
	detectionNN.out.link(xOutNN.input)

	# Create left mono camera (CAM_B)
	left = pipeline.create(dai.node.MonoCamera)
	left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

	# create right mono camera (CAM_C)
	right = pipeline.create(dai.node.MonoCamera)
	right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

	# Create stereo depth
	stereo = pipeline.create(dai.node.StereoDepth)
	stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
	# allocates resources for worst case scenario
	# allowing runtime switch of stereo modes
	stereo.setRuntimeModeSwitch(True)
	stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
	stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
	left.out.link(stereo.left)
	right.out.link(stereo.right)

	xOutDepth = pipeline.create(dai.node.XLinkOut)
	xOutDepth.setStreamName('depth')
	stereo.depth.link(xOutDepth.input)

	return pipeline