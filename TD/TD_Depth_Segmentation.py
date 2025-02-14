import depthai as dai
import numpy as np
import blobconverter

def onInitialize(oakDeviceOp, callCount):
    r = blobconverter.from_zoo('deeplab_v3_mnv2_256x256', shaves=6, zoo_type="depthai")
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

def decode_deeplabv3p(output_tensor):
    class_colors = [[0,0,0], [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    output = output_tensor.reshape(*INPUT_SHAPE)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def get_multiplier(output_tensor):
    class_binary = [[0], [1]]
    class_binary = np.asarray(class_binary, dtype=np.uint8)
    output = output_tensor.reshape(*INPUT_SHAPE)
    output_colors = np.take(class_binary, output, axis=0)
    return output_colors

def createPipeline(oakDeviceOp):
    # Get the blob using blobconverter
    blob = dai.OpenVINO.Blob(blobconverter.from_zoo(name="deeplab_v3_mnv2_256x256", zoo_type="depthai", shaves=6))
    
    # Get input shape from the blob
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
    xOutDepth.setStreamName('depth')
    stereo.depth.link(xOutDepth.input)

    return pipeline
