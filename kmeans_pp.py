import pandas as pd
import numpy as np
import sys
import mykmeanssp

def calc_distance(vec1, vec2):
    distance = np.sum(np.power(vec1.subtract(vec2),2)[1:])
    return (float(distance))

def kmeans_pp (k,data_arr,n,d): 
    np.random.seed(0)
    index_of_first_cntrd = np.random.choice(n)
    index_of_first_cntrd = (int) (merge_data_frame.loc[merge_data_frame[0] == index_of_first_cntrd].index[0])
    init_indexes = [index_of_first_cntrd] 
    Dlist = np.full(n, np.inf)
    Plist = np.zeros((n), dtype=np.float64)  
    i = 1
    new_centroid = data_arr.iloc[index_of_first_cntrd]
    while(i < k):
        for vecIndex in range(len(data_arr)):
            vec1 = data_arr.iloc[vecIndex]
            t_index = (int)(merge_data_frame.loc[vecIndex][0])
            Dlist[t_index] = min(calc_distance(new_centroid,vec1), Dlist[t_index]) 

        Dtotal = Dlist.sum()
        for vecIndex in range(len(data_arr)):
            t_index = (int)(merge_data_frame.loc[vecIndex][0])
            Plist[t_index] = (Dlist[t_index]/Dtotal)
        i += 1 
        chosen_index = np.random.choice(n, size = None, p=Plist)
        chosen_index = (int) (merge_data_frame.loc[merge_data_frame[0] == chosen_index].index[0])
        new_centroid = data_arr.iloc[chosen_index] 
        init_indexes.append(chosen_index)
    
    return [int(data_arr.loc[i][0]) for i in init_indexes], pd.DataFrame(data_arr.iloc[[i for i in init_indexes]])
        
def recieve_input():
    arguments_size = len(sys.argv)
    k_float = float(sys.argv[1])
    k = int(k_float)   
    if k_float != k:
        print("Invalid Input")
        exit()
    if (arguments_size == 6):
        epsilon = float(sys.argv[3])
        max_iter = int(sys.argv[2])
        input_file_1 = sys.argv[4]
        input_file_2 = sys.argv[5]
    elif (arguments_size == 5):
        epsilon = float(sys.argv[2])
        max_iter = 300
        input_file_1 = sys.argv[3]
        input_file_2 = sys.argv[4]
    else:
        print("Invalid Input")
        exit()
    
    return k, max_iter,epsilon, input_file_1, input_file_2

def validate_input(k,max_iter,n,epsilon):
    if(k < 1 or k >= n or max_iter < 0):
        print("Invalid Input")
        exit()
    return



def check_len(coordinate):
            str_coor = str(coordinate)
            str_coor_array = str_coor.split(".")
            curr_len = len(str_coor_array[1])
            if curr_len < 4:
                str_coor = str_coor + '0'    
            return(str_coor)

def print_final_centroids(k_centriods):
    for centroid in k_centriods:
        print(",".join([check_len(round(coordinate,4)) for coordinate in centroid]))

if __name__ == "__main__":
    k, max_iter,epsilon, input_file_1, input_file_2 = recieve_input() 
    data_frame1 = pd.read_csv(input_file_1 , header=None)
    data_frame2 = pd.read_csv(input_file_2 , header=None)
    merge_data_frame = pd.merge(data_frame1, data_frame2 ,on=0)
    full_data_points = pd.merge(data_frame1, data_frame2, left_index=True, right_index=True)
    n = len(merge_data_frame) # n = num of rows
    d = len(merge_data_frame.columns) - 1# d=  num of columns
    validate_input(k,max_iter,n,epsilon)
    first_init, first_k = kmeans_pp(k,merge_data_frame, n,d)

    first_k.drop([0], axis=1, inplace=True)
    merge_data_frame.drop([0], axis=1, inplace=True)
    vectors = merge_data_frame.values.tolist() # DataFrame -> [[],[],[]...]
    k_init_centroids = first_k.values.tolist() 
    k_centroids = mykmeanssp.fit(n, d, k, max_iter, epsilon, vectors, k_init_centroids)

    print(",".join([str(i) for i in first_init]))
    print_final_centroids(k_centroids)
    #python3 kmeans_pp.py 15 750 0 input_3_db_1.txt input_3_db_2.txt


    

    

    
    


    
    





