import pixellib
from pixellib.instance import instance_segmentation
import matplotlib.pyplot as plt
import cv2

class Measure:
  #Class vars
  #stores the pixel count of the desired area
  PIXEL_COUNT = 0

    #Input from user
  def __init__(self, input_im, a_height, a_leglength):
    #actual height of the person measured manually
    self.a_height = a_height
    #actual leg length of the person(below belly button) measured manually
    self.a_leglength = a_leglength
    #ratio of headtoheight : 1:7.5 approx 
    self.a_head = (1/7.5)*a_height  #actual size of head

    #Detect object(person) from image
    instance_seg = instance_segmentation()
    #mask_rcnn_coco.h5 is the pre trained model for object detection
    instance_seg.load_model("mask_rcnn_coco.h5")
    self.segmask, output = instance_seg.segmentImage(input_im, show_bboxes= True, output_image_name = "final.jpg")

   #converts the coloured image to silhouette
  def segment_to_black(self):
     #Reading the image generated after person detection 
    im = cv2.imread('final.jpg')
    #generates a rectangle around detected person in the image
    for index_x, x in enumerate(im):
        for index_y, y in enumerate(x):
          #stores the coordinates of the rectangle generated around the person in the image
          v = self.segmask['masks'][index_x][index_y] 
          #if person is detected ,it is converted into a silhouette
          if v.any() == [True]:
            im[index_x][index_y] = [0, 0, 0] 
    cv2.imwrite("final.jpg", im)   # saving and overwriting final.jpg 
    #plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) #graphical representation

  def crop_and_resize(self):
    #Cropping the final.jpg 
    final = cv2.imread("final.jpg")   
    x1, y1, x2, y2 = self.segmask['rois'][0]
    crop_img = final[x1:x2, y1:y2]
    cv2.imwrite("finalcrop.jpg", crop_img)
    print(f"Cropped image shape - {crop_img.shape}") #prints the coordinates of the cropped image
    #plt.imshow(cv2.cvtColor(cv2.imread("finalcrop.jpg"), cv2.COLOR_BGR2RGB))

    #Resizing the final image
    i_xAxis = 64.0 / crop_img.shape[1]  #setting one parameter(width) along x-axis
    dim = (64, int(crop_img.shape[0] * i_xAxis))  #stores the x and y coordinates of the cropped image
    self.resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("resized.jpg", self.resized)
    print(f"Resized image shape - {self.resized.shape}")
    #plt.imshow(cv2.cvtColor(cv2.imread("resized.jpg"), cv2.COLOR_BGR2RGB))
  
    #for front and side hip measurement
  def hip_measurement(self): 
    #calculates the ratio of upper body to height
    rubth = (self.a_height - self.a_leglength)/self.a_height
    #stores the height of upper body in the image
    i_ub = rubth * self.resized.shape[0] 
    #stores the height of lower body in the image
    i_recub = self.resized.shape[0] - i_ub
    #generates a green coloured rectangle around the calculated dimensions(hip region)
    flood_input = cv2.rectangle(self.resized, (0,int(i_ub)), (64,int(i_recub)),(0,255,0)) 
    x1, y1, x2, y2 = 0, int(i_ub), 64, int(i_recub)
    #calculates the mid point of the hip region
    mid_y = y1 + int((y2 - y1)/2) 
    mid_x = int((x2 - x1)/2)
    print(f"Hip detection rectangle mid-point -> {mid_x, mid_y}")
    return (mid_x, mid_y, flood_input)

  def shoulder_measurement(self):
    ratio_ubth = (self.a_height - self.a_leglength)/(self.a_height)
    i_head = self.resized.shape[0]*(1/6)
    i_upperbody = (6*ratio_ubth)*(i_head)
    fub = i_head + ((i_upperbody-i_head)/2)
    flood_input = cv2.rectangle(self.resized, (0,int(i_head)),(64,(int(fub))), (0,255,0))
    x1, y1, x2, y2 = 0, int(i_head), 64, int(fub)
    mid_y = y1 + int((y2 - y1)/2)
    mid_x = int((x2 - x1)/2)
    print(f"Shoulder detection rectangle mid-point -> {mid_x, mid_y}")
    return (mid_x, mid_y, flood_input)
     
    #for side chest measurement
  def side_chest_measurement(self):
    #ratio of upper body to height
    ratio_ubth = (self.a_height - self.a_leglength)/(self.a_height)
    #stores the size of head
    i_head = self.resized.shape[0]*(1/7.5)
    i_upperbody = (7.5*ratio_ubth)*(i_head)
    fub = i_head + ((i_upperbody-i_head)/2)

    x1, y1, x2, y2 = 0, int(i_head), 64, int(fub)
    #claculating the mid point of chest region
    mid_y = y1 + int((y2 - y1)/2)
    mid_x = int((x2 - x1)/2)
    a = i_head + ((int(fub)-mid_y)*2.2)
    mid_x = 32
    mid_y = int(a)
    print(f"Side-Chest detection mid-point -> {mid_x, mid_y}")
    return (mid_x, mid_y)

  #for front chest measurements
  def front_chest_measurement(self):
    #ratio of upper body to the height
    ratio_ubth = (self.a_height - self.a_leglength)/(self.a_height)
    i_head = self.resized.shape[0]*(1/7.5)
    i_upperbody = (7.5*ratio_ubth)*(i_head)
    fub = i_head + ((i_upperbody-i_head)/2)

    x1, y1, x2, y2 = 0, int(i_head), 64, int(fub)
    mid_y = y1 + int((y2 - y1)/2)
    mid_x = int((x2 - x1)/2)
    a = i_head + ((int(fub)-mid_y)*2)
    mid_x = 32
    mid_y = int(a)
    print(f"Front-Chest detection mid-point -> {mid_x, mid_y}")
    return (mid_x, mid_y)

  #area(desired body parts) filling function
  def flood_fill_right(self, x, y, im):
    OldColor = [0, 0, 0] #black
    CurrentColor = im[y][x]
    #color = [b,g,r]
    
    #Main algo -> if CurrentColor == OldColor
    if (CurrentColor[0] == OldColor[0] and CurrentColor[1] == OldColor[1] and CurrentColor[2] == OldColor[2]):
        im = cv2.circle(im, (x, y), radius=0, color=(0, 0, 255), thickness = -1) #draws a red line in the desired black region(hip/chest)
        #calculates the pixel count of the desired region
        Measure.PIXEL_COUNT += 1
        #traversing to the right side of the image
        self.flood_fill_right(x+1, y,im)

  def flood_fill_left(self, x, y, im):
      OldColor = [0, 0, 0] #black
      CurrentColor = im[y][x-1]
      # color = [b, g, r]
      
      #Main algo -> if CurrentColor == OldColor
      if (CurrentColor[0] == OldColor[0] and CurrentColor[1] == OldColor[1] and CurrentColor[2] == OldColor[2]):
          im = cv2.circle(im, (x, y), radius=0, color=(0, 0, 255), thickness = -1)
          Measure.PIXEL_COUNT += 1
          #traversing to the left side of the image
          self.flood_fill_left(x-1, y,im)
      return im


def initials(height,leglength,Gender,image1,image2,image3):
  #Gender
  #3images
  #FrontImagewitharmsclosed
  person = Measure(image1 , height, leglength) #(image_for_measurement,actual height,actual leg length)
  person.segment_to_black() #silhouette formation
  person.crop_and_resize() #cropping and resizing of image
  x, y, flood_input = person.shoulder_measurement() #for counting pixel of shoulder measurement
  Measure.PIXEL_COUNT = 0
  person.flood_fill_right(x, y, flood_input)
  im = person.flood_fill_left(x, y, flood_input)
  cv2.imwrite("shoulder.jpeg",im)
  shoulder = Measure.PIXEL_COUNT+ 6 
  print(f"Shoulder Pixel count -> {Measure.PIXEL_COUNT}")

  #FrontImagewitharmsopen
  person = Measure(image2 , height, leglength) #(image_for_measurement,actual height,actual leg length)
  person.segment_to_black() #silhouette formation
  person.crop_and_resize() #cropping and resizing of image
  x, y, flood_input = person.hip_measurement() #for counting pixel of front hip measurement
  Measure.PIXEL_COUNT = 0
  person.flood_fill_right(x, y, flood_input)
  im = person.flood_fill_left(x, y, flood_input)
  cv2.imwrite("fronthip.jpeg",im)
  fronthip = Measure.PIXEL_COUNT + 4
  print(f"Hip Pixel count -> {Measure.PIXEL_COUNT + 4}")
  person.crop_and_resize()
  x, y = person.front_chest_measurement() #for counting of front chest measurement
  Measure.PIXEL_COUNT = 0
  person.flood_fill_right(x, y, person.resized)
  im = person.flood_fill_left(x, y, person.resized)
  cv2.imwrite("frontchest.jpeg",im)
  frontchest = Measure.PIXEL_COUNT + 2
  print(f"Front Chest Pixel count -> {Measure.PIXEL_COUNT + 2}")

  #SideImage
  person = Measure(image3,height ,leglength )
  person.segment_to_black() #silhouette conversion
  person.crop_and_resize() #cropping and resizing the image
  x, y, flood_input = person.hip_measurement() #for counting pixel of side hip measurement
  Measure.PIXEL_COUNT = 0
  person.flood_fill_right(x, y, flood_input)
  im = person.flood_fill_left(x, y, flood_input)
  cv2.imwrite("sidehip.jpeg",im)
  sidehip = Measure.PIXEL_COUNT + 2
  print(f"Hip Pixel count -> {Measure.PIXEL_COUNT + 2}")
  person.crop_and_resize()
  x, y = person.side_chest_measurement()  #for counting pixel of side chest measurement
  Measure.PIXEL_COUNT = 0
  person.flood_fill_right(x, y, person.resized)
  im = person.flood_fill_left(x, y, person.resized)
  cv2.imwrite("sidechest.jpeg",im)
  sidechest = Measure.PIXEL_COUNT + 4
  print(f"Side Chest Pixel count -> {Measure.PIXEL_COUNT + 4}")

  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.linear_model import LinearRegression
  df = pd.read_csv('measure.csv') #file containing pixel count
  df.drop(df.index[[0,3,6,7,10]], inplace = True)
  df.reset_index(inplace = True,drop = True)
  f=[] # stores total pixel count
  for i,j in zip(df['front_hip'],df['side_hip']):
    # calculates total perimeter around the hip
      d = i*2 + j*2 
      f.append(round(d,2)) # rounding off to 2 decimal places

  no_of_samples = 6
  x = np.array(f) # stores input of pixel count obtained from the image
  x = x.reshape(no_of_samples, 1)
  y = np.array(df['hip']) # stores the output(actual) measurement
  y = y.reshape(no_of_samples, 1)

  # Model initialization
  regression_model = LinearRegression() #using y=mx+c
  # Fit the data(train the model)
  regression_model.fit(x, y)
  # printing values
  print('Slope:' ,regression_model.coef_) #calculating slope
  print('Intercept:', regression_model.intercept_)#calculating intercept
  # calculating hip size using pixel counts
  calculated_hip = fronthip*2 + sidehip*2
  # calculating final hip(y) using slope and intercept
  final_hip = regression_model.coef_*calculated_hip+regression_model.intercept_
  print(f"Final Hip - {final_hip[0]}")
  # list containing the calculated pixel counts
  calculated = []
  for i in x:
    # stores the actual body measurement 
      j = regression_model.coef_*i+regression_model.intercept_
      calculated.append(j)
  calculatedf = np.array(calculated)
  waist = []
  for k in calculatedf:
    if Gender =='F':
    # using waist to hip ratio
      r = k * 0.83
    else:
      r = k * 0.98
    waist.append(r)
  #print(np.array(waist))
  #print(np.array(df['waist']))
  a = np.array(waist) # stores input of pixel count obtained from the image
  a = a.reshape(no_of_samples, 1)
  b = np.array(df['waist'])#s tores the output(actual) measurement
  b = b.reshape(no_of_samples, 1)
  if Gender == 'F':
    calculated_waist = final_hip * 0.83 # for female
  else:
    calculated_waist = final_hip * 0.98 # for male

  # Model initialization
  regression_model = LinearRegression()
  # Fit the data(train the model)
  regression_model.fit(a,b)
  # printing values
  print('Slope:' ,regression_model.coef_) # calculating slope
  print('Intercept:', regression_model.intercept_) # calculating intercept
  # calculating final waist(y) using slope and intercept
  final_waist = regression_model.coef_*calculated_waist+regression_model.intercept_
  print(f"Final waist - {final_waist[0]}")

  df = pd.read_csv('measure.csv')
  df.drop(df.index[[2,3,6,7,9,10]], inplace = True)
  df.reset_index(inplace = True,drop = True)
  f=[]
  for i,j in zip(df['front_chest'],df['side_chest']):
    # calculates total perimeter around the chest
      d = i*2 + j*2 
      f.append(round(d,2)) # rounding off to 2 decimal places
  no_of_samples = 5
  x = np.array(f) # stores input of pixel count obtained from the image
  x = x.reshape(no_of_samples, 1)
  y = np.array(df['bust']) # stores the output(actual) measurement
  y = y.reshape(no_of_samples, 1)

  # Model initialization
  regression_model = LinearRegression()
  # Fit the data(train the model)
  regression_model.fit(x, y)
  # printing values
  print('Slope:' ,regression_model.coef_) # calculating slope
  print('Intercept:', regression_model.intercept_) # calculating intercept
  # calculating chest size using pixel counts
  calculated_chest = frontchest*2 + sidechest*2
  # calculating final chest(y) using slope and intercept
  final_chest = regression_model.coef_*calculated_chest+regression_model.intercept_
  print(f"Final Chest - {final_chest[0]}")

  df = pd.read_csv('FinalDatasetMyntra.csv')
  df.at[9,'Waist']= '91-95 cm (35.5"-37.5")'
  df.head(13)

  bust = float(final_chest[0])
  waist = float(final_waist[0])
  hips = float(final_hip[0])
  sample = {'Bust': bust, 'Waist': waist, 'Hips':hips}

  labels = {'XS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5, 'XXL': 6}

  for i in range(1, len(df), 2):
      # Special 'XXS' case
      if sample['Bust'] == 74 and sample['Waist'] == 58 and sample['Hips'] == 80:
          print(f"Expected size - XXS")
          break
          
      b_min =  list(map(int, df.loc[i, "Bust"].split()[0].split('-')))[0] 
      b_max =  list(map(int, df.loc[i+1, "Bust"].split()[0].split('-')))[1]
      
      w_min =  list(map(int, df.loc[i, "Waist"].split()[0].split('-')))[0] 
      w_max =  list(map(int, df.loc[i+1, "Waist"].split()[0].split('-')))[1]
      
      h_min =  list(map(int, df.loc[i, "Hips"].split()[0].split('-')))[0]
      h_max =  list(map(int, df.loc[i+1, "Hips"].split()[0].split('-')))[1]
      
      if b_min <= round(sample['Bust'], 0) <= b_max:
          b = df.loc[i, 'International']
      if w_min <= round(sample['Waist'], 0) <= w_max:
          w = df.loc[i, 'International']
      if h_min <= round(sample['Hips'], 0) <= h_max:
          h = df.loc[i, 'International']
      
      if b_min <= sample['Bust'] <= b_max and w_min <= sample['Waist'] <= w_max and h_min <= sample['Hips'] <= h_max:
          #print(f"Expected size - {df.loc[i, 'International']}")
          #break
          return df.loc[i, 'International'],100
  else:
    # Average sizecalculation
    avg = round((labels[b] + labels[w] + labels[h])/3,0)
    acc = (len([val for val in (labels[b], labels[w], labels[h]) if val == avg])/3)*100
    #print(f"Accuracy - {round(acc, 2)}")
    #print(f"Expected size - {list(labels.keys())[list(labels.values()).index(avg)]}")
    size = list(labels.keys())[list(labels.values()).index(avg)]
    accuracy = round(acc, 2)
    return size,accuracy

	
