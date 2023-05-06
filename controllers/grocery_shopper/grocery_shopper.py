"""grocery controller."""
#mapping

#

#finding
#drive to right spot, turn to 0.
#go slowly to waypoint (arm in front)
#full stop at waypoint
#if color detection says its up, reach up
#arm out, then up
#after grabbed, put in basket
#arm in front again, move. 




from controller import Robot
from controller import Display
from controller import Camera
from controller import Keyboard
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
BEAR_TOLERANCE = 0.05
SCALE = 10
OFFSET = 180
STOP_TOLERANCE = 0.0001
STOP_NUMBER = 80
# create the Robot instance.
robot = Robot()
#display = Display("camera")


# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 
TOLERANCE = 0.1
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

map = np.zeros(shape=[360,360])
robot_parts["wheel_left_joint"].getPositionSensor().enable(timestep)
robot_parts["wheel_right_joint"].getPositionSensor().enable(timestep)

arm_joint = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint","arm_5_joint","arm_6_joint", "arm_7_joint"]
middle_shelf = [1.6, -0.92, -3, 1.3, 1.32, 1.39, 0.5] 
top_shelf = [1.6, -0.18, -3.2, 1, -1.9, -1, 0]
overhead = [2.68, 1.02, 0, -0.32, 0, 0, 0]
basket = [1.6, 1.02, 0, 2.29, -2.07, 1.39, 2.07] 

#LOW: arm_grab_left = [2.68, -0.18, -1, -0.32, 2.07, 0.7, 2] 
arm_grab_left_high = [2.68, 0.3, -1, -0.32, -2.07, 1.2, 2.07]
arm_grab_left_low = [2.68, -0.4, -1, -0.32, -2.07, 1.2, 2.07] 
arm_out = [1.3, -0.2, 0, -0.2, 0, 0, 0]
arms = [overhead, arm_grab_left_high, arm_grab_left_low, basket]

get_state = "move"
state_counter = 0 

yellow_range = [[210, 210, 0], [255, 255, 30]]
MASK_WIDTH = 240
MASK_HEIGHT = 135

robots_path = []
bounding_box_x = [0,0,240,240]
bounding_box_y = [0,135,135,0]


mode = 'amanual'

#waypoints = [[0, 5.4],[8, 5.6], [16.5,5.8], [16.5, 2.5], [12, 2.2], [0, 2.0]]

up_v_down = [-1, 1, 0, 1, 0, 1, 0] #go, turn, then grabbing each. 
waypoints = [[0, 2.4], [1.1, 2.6], [2, 2.6], [7, 2.6]]
display.setColor(0x00FF00)
for w in waypoints:
    display.drawPixel(w[0]*SCALE+OFFSET, w[1]*SCALE+OFFSET)
# ------------------------------------------------------------------
# Helper Functions

#states/substates:
#1. mapping
#2. traverse
    #a. move not in shelf
    #b. right and left looking
    #c. yellow found (turn to look)
    #d. pick up 

#~~~~COLOR DETECTION~~~~~
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
    #print_img(img_mask, display)
    blobs = get_blobs(img_mask)
    centroids = get_blob_centroids(blobs)
    #if(len(blobs)>0):
    #    print(centroids) 

dir_status = "left"     
gripper_status="closed"
arm_status=0
arm_statuses = [overhead,top_shelf, basket]
counter = 0
#display = robot.getDevice('display');
width = camera.getWidth()
height = camera.getHeight()

def angle_check(theta, pose_theta):
    #I found that there was getting to be some confusion when the angle got near 0
    pose_theta2 = 0
    if(pose_theta>0): #if positive
        pose_theta2 = pose_theta - 2*np.pi
    else: #if negative
        pose_theta2 = pose_theta + 2*np.pi
    #print(np.abs(theta-pose_theta), np.abs(theta-pose_theta2))
    if(np.abs(theta-pose_theta)<np.abs(theta-pose_theta2)):
        return theta-pose_theta
    #print("what", theta-pose_theta2)
    return theta-pose_theta2
        

#print(height,width)
#display.drawLine(0, height, width, height)
#display.drawLine(width, 0, width, height)

count = 0
curr = 0
prev_positionL = 0
prev_positionR = 0
test_poseX = 0
test_poseY = 0
#~~~~~~START LOOP~~~~~~~~~~
stop_count = 0
while robot.step(timestep) != -1:
    #pose_x = gps.getValues()[0]
    #pose_y = gps.getValues()[1]
    
    robot_parts["torso_lift_joint"].setPosition(0.35)
    
    #n = compass.getValues()
    #rad = -((math.atan2(n[0], n[1]))-1.5708)
    #pose_theta = rad
    
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y
        
        mx = 360-abs(int((5+pose_x)*25))
        my = 117-abs(int(pose_y*15.66))
        
        if rho < LIDAR_SENSOR_MAX_RANGE:
            map_i = 360-abs(int((5+pose_x)*25)) #mx
            map_j = 117-abs(int(pose_y*15.66))  #my
            if (map_i >= 360):
                map_i = 359
            if (map_j >= 360):
                map_j = 359
            if (map_i < 0):
                map_i = 0
            if (map_j < 0):
                map_j = 0
            # Part 1.3: visualize map gray values.
            # Only mapping when moving
            if not(vR and vL == 0):
                map[map_j][map_i] += 0.004
                g = map[map_j][map_i]
                if (g > 1):
                    g = 1
                color = (g*256**2+g*256+g)*255
                display.setColor(int(color))
                display.drawPixel(map_j, map_i)
                
    display.setColor(int(0xFF0000))
    display.drawPixel(my,mx)
    print(map_j,map_i)
    """             
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            filter_map = map > 0.7
            ones = np.ones(shape=[360,235])
            filter_map = np.multiply(filter_map, ones)
            for i in range(360):
                for j in range(360):
                    if (filter_map[i][j] == 0):
                        display.setColor(int(0x000000))
                        display.drawPixel(i, j)
            np.save('map', filter_map)
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            load_map = np.load("map.npy")
            for i in range(360):
               for j in range(360):
                   if (load_map[i][j] == 0):
                       display.setColor(int(0x000000))
                       display.drawPixel(i, j)
                   else:
                       display.setColor(int(0xFFFFFF))
                       display.drawPixel(i, j)
                       map[i][j] = 1
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
            
        #display.setColor(0xFFFFFF)   # White lines for empty space
        #display.drawLine(int(pose_x*SCALE)+OFFSET, int(pose_y*SCALE)+OFFSET, int(wx*SCALE)+OFFSET, int(wy*SCALE)+OFFSET)
    
    #~~~~~~~LIDAR~~~~~~~~
    if robot_state=="planning":  
        for ray in lidar_readings:
            display.setColor(0xFFFFFF)   # White lines for empty space
            display.drawLine(int(pose_x*SCALE)+OFFSET, int(pose_y*SCALE)+OFFSET, int(ray[0]*SCALE)+OFFSET, int(ray[1]*SCALE)+OFFSET)
            display.setColor(0xFF0000)   # Red points for robot path
            display.drawPixel(int(pose_y*SCALE)*OFFSET, int(pose_x*SCALE)+OFFSET)
            display.setColor(0x0000FF)   # Blue points for obstacle edges
            display.drawPixel(int(ray[0]*SCALE)+OFFSET, int(ray[1]*SCALE)+OFFSET)
    """ 
    # Read Lidar           
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_readings = []
    
    
    """
    ###LIDAR READINGS
    
    
    display.setColor(0xFF0000)
    display.drawPixel(int(pose_x*SCALE)+OFFSET, int(pose_y*SCALE)+OFFSET)
    print(int(pose_y*SCALE)+OFFSET, int(pose_x*SCALE)+OFFSET)
    display.setColor(0x00FF00)   # Red points for robot path
    display.drawPixel(int(pose_y*SCALE)*OFFSET, int(pose_x*SCALE)+OFFSET)
    for i in range(83, LIDAR_ANGLE_BINS-83):
        #print(len(lidar_offsets), len(lidar_sensor_readings))
        if not(lidar_sensor_readings[i] == float('inf')):
            # Converting lidar readings to robot coordinates
            xr = np.sin(lidar_offsets[i-83]) * lidar_sensor_readings[i]
            yr = np.cos(lidar_offsets[i-83]) * lidar_sensor_readings[i]
            # Converting robot coordinates to world coordinates
            xw = np.sin(pose_theta)*xr + np.cos(pose_theta)*yr + pose_x
            yw = np.cos(pose_theta)*xr - np.sin(pose_theta)*yr + pose_y
               
            lidar_readings.append([xw, yw])
    
    ##### Part 4: Draw the obstacle and free space pixels on the map
    robots_path.append([pose_x, pose_y])   # Keeping track of all points on robot path
    """
    
    

       
    #color blob detection
    #color_handler()
    
    #******ODOMETRY*******
    
    #Left position, taken from position sensors for accuracy
    curr_positionL = robot_parts["wheel_left_joint"].getPositionSensor().getValue() 
    changeL = curr_positionL-prev_positionL
    lin_velL = ((changeL)/(timestep/1000.0))
    
    #Right positions
    curr_positionR = robot_parts["wheel_right_joint"].getPositionSensor().getValue()
    changeR = curr_positionR-prev_positionR
    lin_velR = ((changeR)/(timestep/1000.0))
    
    prev_positionL = curr_positionL
    prev_positionR = curr_positionR
    
    #Odometry 
    distL = lin_velL/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0
    distR = lin_velR/MAX_SPEED * MAX_SPEED_MS * timestep/1000.0
    pose_x += (distL+distR) / 2.0 * math.cos(pose_theta)
    pose_y += (distL+distR) / 2.0 * math.sin(pose_theta)
    pose_theta += (distR-distL)/AXLE_LENGTH
    pose_theta = pose_theta%(2*np.pi)
    
    
    #move based on state
    
    """
    if(dir_status=="stopped" and stop_count>0):
        vL = 0
        vR = 0
        stop_count-=1
        print(stop_count)
    elif(dir_status=="stopped"):
        dir_status = "drive"
    """
    #Drive, left, right
    if(dir_status=="drive"):# and stop_count==0):
        #print("drive")
        vL = MAX_SPEED/4
        vR = MAX_SPEED/4
    elif(dir_status=="left"):
        vL = -MAX_SPEED/8
        vR = MAX_SPEED/8
    elif(dir_status=="right"):
        vL = MAX_SPEED/8
        vR = -MAX_SPEED/8
    elif(dir_status=="stopped"):
        vL = 0
        vR = 0
    else:
        vL = 0
        vR = 0
        print("INVALID STATE: STOPPING")
    
    
    
    
    #Bearing calculation
    ydist = waypoints[curr][1]-pose_y
    xdist = waypoints[curr][0]-pose_x
    theta = np.arctan2(ydist,xdist)
    print("THETAS", theta, pose_theta, theta-pose_theta, "DISTS", ydist, xdist)
    
    #Errors
    bear = angle_check(theta, pose_theta) 
    dist = np.sqrt(ydist**2 + xdist**2)
    display.setColor(0xFF0000)
    
    #which way to turn based off of bearing
    if((bear>BEAR_TOLERANCE and bear<(2*np.pi - BEAR_TOLERANCE)) or (bear<-BEAR_TOLERANCE and bear>-(2*np.pi - BEAR_TOLERANCE)) and stop_count==0):
        if(bear>0):
            dir_status = "left"
        else:
            dir_status = "right"
    else:
        dir_status = "drive"
    
    #gV = robot_parts["wheel_left_joint"].getVelocity()
    #gV2 = robot_parts["wheel_right_joint"].getVelocity()
    print("B:", bear, "D:", dist, "VL,Real", lin_velL, lin_velR, "POSEX/Y:", pose_x, pose_y, vL, vR)
    
    #move to next waypoint (for now, until arm movement) 
    
       
    counter = counter + 1
    #if(dir_status=="arm"):
    #    for joint in range(len(arm_joint)):
    #        robot_parts[arm_joint[joint]].setPosition(arm_statuses[arm_status][joint])
    #print(counter)
        #print(arm_statuses[arm_status][joint])
     
     
    #*****GRAB SOMETHING STATE***** 
    #stop    
    
    if get_state=="stop":
        vL = 0
        vR = 0
        if(lin_velL<STOP_TOLERANCE and lin_velR<STOP_TOLERANCE):
            #if -1, then keep going
            get_state="move_arm_out"
            state_counter = 0
            
    #Move arm out to not hit a shelf when moving arm up or down        
    elif get_state=="move_arm_out":
        vL = 0
        vR = 0
        for i in range(len(arm_joint)):
            robot_parts[arm_joint[i]].setPosition(arm_out[i])
        state_counter = state_counter+1
        if state_counter>=STOP_NUMBER:
            get_state = "move_arm_to_shelf"
            state_counter = 0
            
    #Put arm to a given shelf        
    elif get_state=="move_arm_to_shelf":
        vL = 0
        vR = 0
        up = False
        if up_v_down[curr-1] == 1:
            up = True
        for i in range(len(arm_joint)):
            if up:
                robot_parts[arm_joint[i]].setPosition(arm_grab_left_high[i])
                robot_parts["torso_lift_joint"].setPosition(0.35)
            else:
                robot_parts[arm_joint[i]].setPosition(arm_grab_left_low[i])
                robot_parts["torso_lift_joint"].setPosition(0)
        state_counter = state_counter+1
        if state_counter>=STOP_NUMBER:
            get_state = "grab"
            state_counter = 0
    
    #Grab with hand        
    elif get_state == "grab":
        vL = 0
        vR = 0
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        state_counter = state_counter+1
        if state_counter>=STOP_NUMBER:
            get_state = "basket"
            state_counter = 0
            
    #Move arm to basket
    elif get_state == "basket":
        vL = 0
        vR = 0
        for i in range(len(arm_joint)):
            robot_parts[arm_joint[i]].setPosition(basket[i])
        state_counter = state_counter+1
        if state_counter>=STOP_NUMBER:
            get_state = "drop"
            state_counter = 0
            
    elif get_state == "drop":
        vL = 0
        vR = 0
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        state_counter = state_counter+1
        if state_counter>=STOP_NUMBER:
            get_state = "move"
            state_counter = 0
    print(get_state, state_counter)    
          
    
            
    #Error compensation
    if(vL==vR and vL!=0): #accounting for robot error, making it so that one wheel doesn't go insanse
        if(lin_velL<lin_velR - TOLERANCE):
            vR = np.abs(lin_velL) + 0.1
        elif(lin_velR<lin_velL - TOLERANCE):
           vL = np.abs(lin_velR) + 0.1
    #print(vL, vR, gV, gV2)
    
    #********ARM CONTROL HERE*****
    
    #for i in range(len(arm_joint)):
    #    robot_parts[arm_joint[i]].setPosition(arm_grab_left[i])
    #***CHECK ACCELERATION***
    if(dist<0.1):
        curr+=1
        if(up_v_down[curr-1]>=0):
            get_state = "stop"
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    #print(right_gripper_enc.getValue(), left_gripper_enc.getValue(), gripper_status)
    """
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        #if right_gripper_enc.getValue()<=0.005:
        #    gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        #if left_gripper_enc.getValue()>=0.044:
        #    gripper_status="open"
    """
    #display.drawPixel(pose_x*SCALE+130, pose_y*SCALE+130)        
    #only for blob map
    #count = (count + 1)%10
    #if count==0:
    #    clear_display()
    
