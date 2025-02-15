import depthai as dai
import numpy as np
import cv2

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
    # Get the NN output from the input channels
    nn_data = op('nn')
    depth_data = op('depth')
    
    if nn_data is not None and depth_data is not None:
        # Get NN output and reshape
        layer1 = nn_data.numpyArray()
        lay1 = layer1.reshape(*INPUT_SHAPE)
        
        # Get depth frame
        depth_frame = depth_data.numpyArray()
        depth_frame = cv2.resize(depth_frame, TARGET_SHAPE)
        
        # Create binary mask from segmentation
        mask = (lay1 > 0).astype(np.uint8)
        mask = cv2.resize(mask, TARGET_SHAPE)
        
        # Apply mask to depth
        depth_overlay = depth_frame * mask
        
        # Send the masked depth frame to output
        return depth_overlay
    return

def onDone(oakDeviceOp):
    return

def createPipeline(oakDeviceOp):
    # Get the blob path from TDBD
    blobPath = op.TDBD.From_zoo('deeplab_v3_mnv2_256x256', shaves=6, zoo_type="depthai").path
    blob = dai.OpenVINO.Blob(blobPath)
    
    # Get input shape from the blob
    global INPUT_SHAPE, TARGET_SHAPE
    INPUT_SHAPE = blob.networkInputs['Input'].dims[:2]
    TARGET_SHAPE = (400, 400)
    
    # Set default FPS if not specified
    if hasattr(parent().par, 'Fps'):
        fps = parent().par.Fps.eval()
    else:
        fps = 30

    # Create pipeline
    pipeline = dai.Pipeline()

    # RGB Camera setup
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setFps(fps)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setPreviewKeepAspectRatio(False)
    camRgb.setIspScale(2, 3)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setPreviewSize(*INPUT_SHAPE)
    camRgb.setInterleaved(False)

    # Neural Network setup
    detectionNN = pipeline.create(dai.node.NeuralNetwork)
    detectionNN.setBlob(blob)
    detectionNN.input.setBlocking(False)
    detectionNN.setNumInferenceThreads(2)
    camRgb.preview.link(detectionNN.input)

    # Create mono cameras for depth
    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Create stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setRectifyEdgeFillColor(0)
    stereo.setRuntimeModeSwitch(True)
    
    # Link mono cameras to stereo
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    # Create outputs
    xOutNN = pipeline.create(dai.node.XLinkOut)
    xOutNN.setStreamName('nn')
    detectionNN.out.link(xOutNN.input)

    xOutDepth = pipeline.create(dai.node.XLinkOut)
    xOutDepth.setStreamName('depth1')
    stereo.depth.link(xOutDepth.input)

    return pipeline
