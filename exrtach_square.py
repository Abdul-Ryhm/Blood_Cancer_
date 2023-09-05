import cv2
import numpy as np
import os 

# Load the circular image folder 
#enter the folder address 
slide_number= "low_4"
r_folder= "400"
selected_folder_path=os.path.join(slide_number,r_folder)
target_folder = os.listdir(selected_folder_path)
for i in target_folder:
    mobile_i = i.split("_")[0]
    #parts = i.split('_')
    #print(len(parts))
    #parts_0 = i.split(',')[0]
    #print(parts_0)
    #new_part_0 = parts_0.replace('ALL', '10000')

    #new_part = parts.replace('10000_', 'ALL_')
    #new_name = f"{new_part_0}_{new_part}"
    new_name=i
    #if len(parts) == 4:
    #in case  
    if mobile_i == "Mobile":
        image_path =os.path.join(selected_folder_path,i)
        print(i)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        orignal_image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        orignal_image1 = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        ret, binary_image = cv2.threshold(gray_image, 30, 255, 0)
        binary_image= np.where(binary_image > 0, 1 , binary_image)
        #binary_image[binary_image > 0] = 1

        row_threshold = np.sum(binary_image[int(binary_image.shape[0]/2)])
        col_threshold = np.sum(binary_image[:, int(binary_image.shape[1]/2)])

        # Initialize variables for bounding box
        x_start, y_start, x_end, y_end = 0, 0, binary_image.shape[1], binary_image.shape[0]
        #while True:

               # first_column_sum = np.sum(binary_image[:, 0])
               # last_column_sum = np.sum(binary_image[:, -1])
               # first_row_sum = np.sum(binary_image[0, :])
                #last_row_sum = np.sum(binary_image[-1, :])
                #print(first_row_sum2 )
                #if first_column_sum < 100 :

                   # binary_image = binary_image[:, 1:]
                   # orignal_image = orignal_image[:, 1:, :]
                   # x_start += 1
                #last_column_sum = np.sum(binary_image[:, -1])
                #if last_column_sum < 100 :
                    #print(last_column_sum)
                    #print(col_threshold )
                    #binary_image = binary_image[:, :-1]
                   #orignal_image = orignal_image[:, :-1, :]
                   # x_end -= 1
               # row_threshold = np.sum(binary_image[int(binary_image.shape[0]/2)])
               # col_threshold = np.sum(binary_image[:, int(binary_image.shape[1]/2)])
              #  ab = np.sum(binary_image[:, 0])
               # cd = np.sum(binary_image[0])

                #if binary_image.shape[0] >= binary_image.shape[1]:
                    #break
               # if  last_column_sum >= 10 and first_column_sum >= 10:
                   # break
        while True:

                
                first_row_sum = np.sum(binary_image[0, :])
                last_row_sum = np.sum(binary_image[-1, :])

                #print(first_row_sum)
                if first_row_sum  > last_row_sum:

                    binary_image = binary_image[:-1, :]
                    orignal_image = orignal_image[:-1, :, :]
                    y_end -= 1
                #last_column_sum = np.sum(binary_image[:, -1])
                #if last_row_sum > first_row_sum :
                    #print(last_column_sum)
                    #print(col_threshold )
                    # binary_image = binary_image[1:, :]
                     #orignal_image = orignal_image[1:, :, :]
                     #y_start += 1
               # row_threshold = np.sum(binary_image[int(binary_image.shape[0]/2)])
                #col_threshold = np.sum(binary_image[:, int(binary_image.shape[1]/2)])
                

                #if binary_image.shape[0] >= binary_image.shape[1]:
                    #break
                if first_row_sum  <= last_row_sum:
                    break

        while True:




            #print("i am on")
            previous_binary_image = binary_image.copy()
            row_size, column_size = binary_image.shape

            first_column_sum = np.sum(binary_image[:, 0])
            if first_column_sum < col_threshold:
                binary_image = binary_image[:, 1:]
                orignal_image = orignal_image[:, 1:, :]
                x_start += 1

            last_column_sum = np.sum(binary_image[:, -1])
            if  last_column_sum < col_threshold:
                binary_image = binary_image[:, :-1]
                orignal_image = orignal_image[:, :-1, :]
                x_end -= 1

            first_row_sum = np.sum(binary_image[0, :])
            if first_row_sum < col_threshold:
                binary_image = binary_image[1:, :]
                orignal_image = orignal_image[1:, :, :]
                y_start += 1

            last_row_sum = np.sum(binary_image[-1, :])
            if last_row_sum < col_threshold:
                binary_image = binary_image[:-1, :]
                orignal_image = orignal_image[:-1, :, :]
                y_end -= 1

            row_threshold = np.sum(binary_image[int(binary_image.shape[0]/2)])
            col_threshold = np.sum(binary_image[:, int(binary_image.shape[1]/2)])
            if np.array_equal(binary_image, previous_binary_image):
                break
        # Draw bounding box on the original image
        cv2.rectangle(orignal_image1, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        #cv2_imshow(orignal_image1)
        print("hello")
        if not os.path.exists(os.path.join(slide_number,"400_annotated_folder")):
                os.makedirs(os.path.join(slide_number,"400_annotated_folder"))
        if not os.path.exists(os.path.join(slide_number,"400_Cropped_folder")):
                os.makedirs(os.path.join(slide_number,"400_Cropped_folder"))
        cv2.imwrite(os.path.join(slide_number,"400_Cropped_folder",new_name), orignal_image)
        cv2.imwrite(os.path.join(slide_number,"400_annotated_folder",new_name), orignal_image1)
        #print(os.path.join(selected_folder_path,"10000_Cropped_folder",new_name))