"""grocery controller."""

# Nov 2, 2022

from controller import Robot
from controller import Display
from controller import Camera
import math
import numpy as np
import copy

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()
#display = Display("camera")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")
#display.height = camera.getHeight()
#display.width = camera.getWidth()
#display.enable(timestep)

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None

arm_joint = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint","arm_5_joint","arm_6_joint", "arm_7_joint"]
middle_shelf = [1.6, -0.92, -3, 1.3, 1.32, 1.39, 0.5] 
top_shelf = [1.6, -0.18, -3.2, 1, -1.9, -1, 0]
overhead = [2.68, 1.02, 0, -0.32, 0, 0, 0]
basket = [1.6, 1.02, 0, 2.29, -2.07, 1.39, 2.07] 

yellow_range = [[210, 210, 0], [255, 255, 30]] #RGB, might be an issue
MASK_WIDTH = 240
MASK_HEIGHT = 135

bounding_box_x = [0,0,240,240]
bounding_box_y = [0,135,135,0]

# ------------------------------------------------------------------
# Helper Functions

#states/substates:
#1. mapping
#2. traverse
    #a. move not in shelf
    #b. right and left looking
    #c. yellow found (turn to look)
    #d. pick up 

#Color detection
def recalculating_img(img_test):
    img_new = np.zeros([MASK_HEIGHT,MASK_WIDTH,3])
    img_mask = np.zeros([MASK_HEIGHT, MASK_WIDTH])
    for i in range(MASK_HEIGHT):
        for j in range(MASK_WIDTH):
            img_new[i][j][0] = Camera.imageGetRed(img_test, MASK_WIDTH, j, i) 
            img_new[i][j][1] = Camera.imageGetGreen(img_test, MASK_WIDTH, j, i) 
            img_new[i][j][2] = Camera.imageGetBlue(img_test, MASK_WIDTH, j, i)
            if pixelYellow(img_new[i][j][0], img_new[i][j][1], img_new[i][j][2]):
                img_mask[i][j] = 1 
    return img_mask
       
def pixelYellow(r, g, b, yellow = yellow_range):
    if r>=yellow[0][0] and r<=yellow[1][0]:
        if g>=yellow[0][1] and g<=yellow[1][1]:
            if b>=yellow[0][2] and b<=yellow[1][2]:
                return True
    return False
    
#remove once we have mapping    
def print_img(img_mask, display):
    for i in range(MASK_HEIGHT):
        for j in range(MASK_WIDTH):
            if img_mask[i][j]==1:
                display.drawPixel(j, i)
      
def clear_display(display = display):
    display.setColor(0x000000)
    display.fillRectangle(0,0,MASK_WIDTH, MASK_HEIGHT)
    display.setColor(0xFFFFFF)


def expand_nr(img_mask, cur_coord, coordinates_in_blob):
    #copy and pasted from McCarthy HW3
    coordinates_in_blob = []
    coordinate_list = [cur_coord]
    height = img_mask.shape[0]
    width = img_mask.shape[1]
    while len(coordinate_list) > 0:
        cur_coordinate = coordinate_list.pop()

        if cur_coordinate[0]>=0 and cur_coordinate[0]<height and cur_coordinate[1]>=0 and cur_coordinate[1]<width:
            if img_mask[cur_coordinate[0], cur_coordinate[1]] == 0:
                continue
            else:
                coordinates_in_blob.append(cur_coordinate)
            img_mask[cur_coordinate[0], cur_coordinate[1]] = 0
            coordinate_list.append([cur_coordinate[0]-1, cur_coordinate[1]])
            coordinate_list.append([cur_coordinate[0], cur_coordinate[1]-1])
            coordinate_list.append([cur_coordinate[0]+1, cur_coordinate[1]])
            coordinate_list.append([cur_coordinate[0], cur_coordinate[1]+1])
        else:
            continue
    return coordinates_in_blob
    
    
def get_blobs(img_mask):
    mask_copy = copy.copy(img_mask)
    blobs_list = []  
    for y in range(MASK_HEIGHT):
        for x in range(MASK_WIDTH):
            if(mask_copy[y,x]==1):
                blob = expand_nr(mask_copy, [y,x], [])
                if len(blob)>5:
                    blobs_list.append(blob)
    return blobs_list

def get_blob_centroids(blobs_list):
    object_positions_list = []
    for blob in blobs_list:
        sumy = 0
        sumx = 0
        for el in blob:
            sumy = sumy+el[0]
            sumx = sumx+el[1]
        sumy = sumy/len(blob)
        sumx = sumx/len(blob)
        object_positions_list.append([sumy,sumx])
    return object_positions_list  
    
def color_handler(display = display):
    img_mask = recalculating_img(camera.getImage())
    print_img(img_mask, display)
    blobs = get_blobs(img_mask)
    centroids = get_blob_centroids(blobs)
    if(len(blobs)>0):
        print(centroids)      
gripper_status="closed"
arm_status=0
arm_statuses = [overhead, middle_shelf, top_shelf, basket]
counter = 0
#display = robot.getDevice('display');
width = camera.getWidth()
height = camera.getHeight()
print(height,width)
display.drawLine(0, height, width, height)
display.drawLine(width, 0, width, height)
    #
# Main Loop
#display.attachCamera(camera)
count = 0
while robot.step(timestep) != -1:
    #
    #odometry
    
   # print(img_test_red)
    #print(camera.getImageArray())
    #detectYellow(camera.getImageArray())
    #img = camera.getImageArray()
    #print(img[130][128])
    
    color_handler()
    #print_img(data, display)
    #print(display.height, display.width)
    #display.imageNew(len(data), len(data[0]), data, Display.RGB)
    
    distL = vL/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0
    distR = vR/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0
    pose_x += (distL+distR) / 2.0 * math.cos(pose_theta)
    pose_y += (distL+distR) / 2.0 * math.sin(pose_theta)
    pose_theta += (distR-distL)/AXLE_LENGTH
    
    #arm demonstration 
    counter = counter + 1
    for joint in range(len(arm_joint)):
        robot_parts[arm_joint[joint]].setPosition(arm_statuses[arm_status][joint])
    #print(counter)
        #print(arm_statuses[arm_status][joint])
    if(counter > 120):
        print("Change")
        arm_status = (arm_status + 1)%3
        counter = 0
            
    robot_parts["wheel_left_joint"].setVelocity(0)
    robot_parts["wheel_right_joint"].setVelocity(0)
    #print(right_gripper_enc.getValue(), left_gripper_enc.getValue(), gripper_status)
    
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
    count = (count + 1)%10
    if count==0:
        clear_display()
    
