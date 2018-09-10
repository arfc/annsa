import numpy as np
from numpy import genfromtxt
import random
from scipy import signal
import datetime
import os
from scipy.ndimage.interpolation import zoom

def write_time_and_date():
    os.environ['TZ'] = 'CST6CDT'
    return "Time" + datetime.datetime.now().strftime("_%H_%M_%S_")+\
    "Date" + datetime.datetime.now().strftime("_%Y_%m_%d")


def weight_variable(shape,stddev,name='none'):
    if name=='none':
        initial = tf.truncated_normal(shape,stddev=stddev, dtype=tf.float32)
        return tf.Variable(initial)
    else: 
        initial = tf.truncated_normal(shape,stddev=stddev, dtype=tf.float32)
        return tf.Variable(initial, name=name)
    

def bias_variable(shape,name='none'):
    if name=='none':
        initial = tf.truncated_normal(shape,stddev=1.0, dtype=tf.float32)
        return tf.Variable(initial)
    else:
        initial = tf.truncated_normal(shape,stddev=1.0, dtype=tf.float32)
        return tf.Variable(initial, name=name)




class ANN_structure_details(object):
    def __init__(self, 
                 layer_1_nodes = 50,
                 layer_2_nodes = 0,
                 layer_3_nodes = 0,
                 layer_4_nodes = 0,
                 learning_rate= 0.001,
                 dropout_rate= 0.5,
                 optimizer = 'Adam',
                 scale_factor = 25.0,
                 L2_batch_size = 128,
                 spectra_length = 1024,
                 num_categories = 33,
                 batch_size = 128,
                 softmax_gate = 0.0
                 ):
        self.layer_1_nodes  = layer_1_nodes
        self.layer_2_nodes  = layer_2_nodes
        self.layer_3_nodes  = layer_3_nodes
        self.layer_4_nodes  = layer_4_nodes
        self.learning_rate  = learning_rate
        self.dropout_rate   = dropout_rate
        self.optimizer      = optimizer
        self.scale_factor   = scale_factor
        self.L2_batch_size  = L2_batch_size
        self.spectra_length = spectra_length
        self.num_categories = num_categories
        self.batch_size     = batch_size
        self.softmax_gate   = softmax_gate


def make_ANN_structure(my_ANN_strucutre):
    '''
    Author: Mark Kamuda (7/1/17)
    
    This function creates the TensorFlow code necessary to create and run a neural network.
    
    Function uses the my_ANN_strucutre object 
    
    Function can only make up to a 4 hidden-layer network. 
    Addition of extra layers is easy to implemnt.
    
    '''
    
    layer_1_nodes  = my_ANN_strucutre.layer_1_nodes
    layer_2_nodes  = my_ANN_strucutre.layer_2_nodes
    layer_3_nodes  = my_ANN_strucutre.layer_3_nodes
    layer_4_nodes  = my_ANN_strucutre.layer_4_nodes
    learning_rate  = my_ANN_strucutre.learning_rate
    dropout_rate   = my_ANN_strucutre.dropout_rate
    optimizer      = my_ANN_strucutre.optimizer
    scale_factor   = my_ANN_strucutre.scale_factor
    L2_batch_size  = my_ANN_strucutre.L2_batch_size
    spectra_length = my_ANN_strucutre.spectra_length
    num_categories = my_ANN_strucutre.num_categories    
    batch_size     = my_ANN_strucutre.batch_size
    
    
    
    x_spectra = tf.placeholder(tf.float32, [None,spectra_length])
    y_ = tf.placeholder(tf.float32, [None,num_categories])
    keep_prob = tf.placeholder(tf.float32)

    if layer_1_nodes > 0:
        nodes_1 = layer_1_nodes
        W1 = weight_variable([ spectra_length, nodes_1 ], stddev = 1/np.sqrt(spectra_length), name='W1')
        b1 = bias_variable([nodes_1], name = 'b1') 
        W_out = weight_variable([ nodes_1, num_categories ], stddev = 1/np.sqrt(nodes_1))
        b_out = bias_variable([num_categories])
        N1 = tf.nn.relu( tf.matmul(x_spectra, W1) + b1 )
        N1_drop = tf.nn.dropout(N1, keep_prob)    
        
        N_out = tf.nn.softmax( tf.matmul(N1_drop , W_out) + b_out )
        L2_Reg = (scale_factor/batch_size)*\
        ( tf.nn.l2_loss(W1) + tf.nn.l2_loss(W_out)  )
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits = tf.matmul(N1_drop , W_out) + b_out  , labels = y_ ))+L2_Reg
        
        tf.add_to_collection('W1', W1)
        tf.add_to_collection('b1', b1)
        
    if layer_2_nodes > 0:
        nodes_2 = layer_2_nodes
        W2 = weight_variable([ nodes_1, nodes_2 ], stddev = 1/np.sqrt(nodes_1), name='W2')
        b2 = bias_variable([nodes_2], name='b2')
        W_out = weight_variable([ nodes_2, num_categories ], stddev = 1/np.sqrt(nodes_2), name='W_out')
        b_out = bias_variable([num_categories], name='b_out')
        N2 = tf.nn.relu( tf.matmul(N1_drop, W2) + b2 )
        N2_drop = tf.nn.dropout(N2, keep_prob)   
        
        N_out = tf.nn.softmax( tf.matmul(N2_drop , W_out) + b_out )
        L2_Reg = (scale_factor/batch_size)*\
        ( tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W_out)  )
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits = tf.matmul(N2_drop , W_out) + b_out  , labels = y_ ))+L2_Reg

        tf.add_to_collection('W2', W2)
        tf.add_to_collection('b2', b2)
        
        
    if layer_3_nodes > 0:
        nodes_3 = layer_3_nodes
        W3 = weight_variable([ nodes_2, nodes_3 ], stddev = 1/np.sqrt(nodes_2))
        b3 = bias_variable([nodes_3])
        W_out = weight_variable([ nodes_3, num_categories ], stddev = 1/np.sqrt(nodes_3), name='W_out')
        b_out = bias_variable([num_categories], name='b_out')           
        N3 = tf.nn.relu( tf.matmul(N2_drop, W3) + b3 )
        N3_drop = tf.nn.dropout(N3, keep_prob)   
        
        N_out = tf.nn.softmax( tf.matmul(N3_drop , W_out) + b_out )
        L2_Reg = (scale_factor/batch_size)*\
        ( tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) +  tf.nn.l2_loss(W_out)  )
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits = tf.matmul(N3_drop , W_out) + b_out  , labels = y_ ))+L2_Reg
       
        tf.add_to_collection('W3', W3)
        tf.add_to_collection('b3', b3)
    
    
    if layer_4_nodes > 0:
        nodes_4 = layer_4_nodes
        W4 = weight_variable([ nodes_3, nodes_4 ], stddev = 1/np.sqrt(nodes_3))
        b4 = bias_variable([nodes_4])
        W_out = weight_variable([ nodes_4, num_categories ], stddev = 1/np.sqrt(nodes_4), name='W_out')
        b_out = bias_variable([num_categories], name='b_out')     
        N4 = tf.nn.relu( tf.matmul(N3_drop, W4) + b4 )
        N4_drop = tf.nn.dropout(N4, keep_prob)                         
        
        N_out = tf.nn.softmax( tf.matmul(N4_drop , W_out) + b_out )
        L2_Reg = (scale_factor/batch_size)*\
        ( tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W_out)  )
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits = tf.matmul(N4_drop , W_out) + b_out  , labels = y_ ))+L2_Reg

        tf.add_to_collection('W4', W4)
        tf.add_to_collection('b4', b4)
        
    if optimizer == 'GradientDescent':
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    if optimizer == 'Adam':
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    tf.add_to_collection('N_out', N_out)
    tf.add_to_collection('cross_entropy', cross_entropy)
    tf.add_to_collection('L2_Reg', L2_Reg)
    
    
    
    def ANN_full(_spectra):
        return N_out
        
        
    
    return train_step, L2_Reg, cross_entropy, x_spectra, y_, keep_prob, ANN_full
    
    
    
  
    
def scale_data(data, mode='zscore'):
    '''
    function assumes data is in shape [index,channels]
    
    '''

    normalized_data = np.empty_like(data)
    
    
    if mode == 'zscore':
        data_mean = np.mean(data,axis=0)
        data_std = np.std(data,axis=0)

        normalized_data = (data-data_mean)/data_std
        
    if mode == 'minmax':
        data_max = np.max(training_data,axis=1)
        
        for i in range(0,data.shape[0]):
            normalized_data[i] = data[i]/data_max[i] 
        
        
    if mode == 'sigmoid':
        data_mean = np.mean(data,axis=0)
        data_std = np.std(data,axis=0)

        alpha = (data-data_mean)/data_std
        
        normalized_data = (1 - np.exp(-alpha))/(1 + np.exp(-alpha))
        
    if mode == 'sqrt':
        data_mean = np.sqrt(data,axis=0)
        normalized_data = (1 - np.exp(-alpha))/(1 + np.exp(-alpha))
        
        
    else:
        print 'do better at spelling'

        
        
    return normalized_data




def normalize_data(key):    
    temp_key = []
    
    for i in range(len(key)):
        temp_key.append(key[i]/float(np.sum(key[i])))
    
    return np.array(temp_key)


def log_normalize(data):
    return np.log10(data+1.0)


def minimax_data(data):    
    data_normalized = np.empty_like(data)
    
    for i in range(data.shape[0]):
        data_normalized[i] = data[i]/np.max(data[i])
    
    return data_normalized


def write_network_details_to_file(network_details_object,old_file_id,loss_train,loss_test,i,DataSubset_id):
        
    f = open('./HP_test_results/'+ old_file_id+'_' + DataSubset_id + '.txt', 'w')
    
    f.write('The following hyperparameters are measured\n')
    f.write('layer1_nodes,layer2_nodes,layer3_nodes,layer4_nodes,scale_factor,learning_rate,dropout_rate,batch_size\n')
    f.write('{},{},{},{},{},{},{},{}\n'.format(network_details_object.layer_1_nodes,network_details_object.layer_2_nodes,
                                            network_details_object.layer_3_nodes,network_details_object.layer_4_nodes,
                                            network_details_object.scale_factor,network_details_object.learning_rate,
                                            network_details_object.dropout_rate,network_details_object.batch_size))

    f.write('Final training error, final testing error, final validation error, and number of iterations are below\n')
    f.write('{},{},{}\n'.format(loss_train[i-1],loss_test[i-1],i))

    
    
    
    f.close()
    
    
DataSubset_id = 'TEST_ID'
    







def run_model(ANN_full, spectrum=np.zeros([1,1024]), print_results=True,spectrum_file_name='none',
              num_top_isotopes=5, plot_on = 0, plot_mode = 'linear', DRF_perturb = 1.0, xlim=1024):
        
    if spectrum_file_name != 'none':
        
        spectrum = np.zeros(1024)
        
        with open(spectrum_file_name) as f:

            # read each spectra into a temp file, total of 1024 channels in this spectra
            content = f.readlines() # read all of the .Spe file into contnet 
            for i in range(1024):
                spectrum[i] = int(content[12+i]) # spectra begins at index 12, int to convert string in .Spe to int

   


    model_results = sess.run(ANN_full(x_spectra), feed_dict={x_spectra: (spectrum).reshape(1,1024),keep_prob: 1.0 })[0]
    
    if print_results == True:
        #print 'ANN Results:'  
        results2(model_results,num_top_isotopes)
    
    if plot_on == 1:
#        plt.plot(spectrum)

        plt.rcParams.update({'font.size': 20})

        plt.figure(figsize=(12,6))  

        if plot_mode == 'linear':
            plt.plot(spectrum,'k')
        else:
            plt.semilogy(spectrum,'k')

        plt.xlim([0,xlim])       
        plt.xlabel('channel')
        plt.ylabel('Counts')

    return model_results

   
    
def massage_the_data(data):

    data[0:10] = 0
    return data/np.max(data)


def results2(res,number_isotopes_displayed):
    
    index = [i[0] for i in sorted(enumerate(res), key=lambda x:x[1])]
    index = list(reversed(index))
    for i in range(number_isotopes_displayed):
        print isotopes[index[i]], round(res[index[i]],3)

    
def load_template_spectra_from_folder(parent_folder,spectrum_identifier, LLD=10):
    '''
    inputs: partent_folder, spectrum_identifier
    output: dictionary containing all template spectra from a folder.
    
    Load template spectrum data into a dictionary. This allows templates from 
    different folders to be loaded into different dictionaries.
    
    '''
    
    
    #parent_folder       = "/home/ubuntu/Notebooks/GADRAS_ANN_work/GADRAS_parent_spectra/unshielded_low_dead_time/"
    #spectrum_identifier = "_1uC_spectrum.spe"
    
    temp_dict = {}
    
    def normalize_spectrum(ID):
        temp_spectrum = read_spectrum(parent_folder + ID + spectrum_identifier)
        temp_spectrum[0:LLD]     = 0
        return temp_spectrum/np.max(temp_spectrum)
    
    for i in range(len(isotopes)):
        # Fixes background spectra name issue
        if i>=len(isotopes)-3:
            spectrum_identifier = ''
            
        temp_dict[isotopes[i]] = normalize_spectrum(isotopes_GADRAS_ID[i]) # 0 
    

    return temp_dict


def RepresentsInt(s):
    '''
    Helper funtion to see if a string represents an integer    
    '''
    try: 
        int(s)
        return True
    except ValueError:
        return False

def results2(res,number_isotopes_displayed):
    
    index = [i[0] for i in sorted(enumerate(res), key=lambda x:x[1])]
    index = list(reversed(index))
    for i in range(number_isotopes_displayed):
        print isotopes[index[i]], res[index[i]]
        
def zoom_spectrum(spectrum,zoom_strength):
    spectrum= np.abs(zoom( spectrum , zoom_strength))
    if zoom_strength < 1.0:
        spectrum = np.lib.pad(spectrum, (0,1024-spectrum.shape[0]), 'constant', constant_values=0)
    if zoom_strength > 1.0:
        spectrum = spectrum[0:1024]
    
    return spectrum
    
def read_spectrum(filename):
    '''
    Reads spectrum from .spe files. 
    Works with silver detector and GADRAS formatted spectra.
    '''
    spectrum = np.empty(1024)

    with open(filename) as f:
        
        content = f.readlines()
        
        if RepresentsInt(content[8]) == True:
            for i in range(1024):
                spectrum[i] = int(content[8+i]) # spectra begins at index 8
        else: 
            for i in range(1024):
                spectrum[i] = int(content[12+i]) # spectra begins at index 12   
            
            
    return spectrum
    
def generate_single_source_key(
                                counts_from_Am241  = 0,
                                counts_from_Ba133  = 0,
                                counts_from_Co57   = 0,
                                counts_from_Co60   = 0,
                                counts_from_Cs137  = 0,
                                counts_from_Cr51   = 0,
                                counts_from_Eu152  = 0,
                                counts_from_Ga67   = 0,
                                counts_from_I123   = 0,
                                counts_from_I125   = 0,

                                counts_from_I131   = 0,
                                counts_from_In111  = 0,
                                counts_from_Ir192  = 0,
                                counts_from_U238   = 0,
                                counts_from_Lu177m = 0,
                                counts_from_Mo99   = 0,
                                counts_from_Np237  = 0,
                                counts_from_Pd103  = 0,
                                counts_from_Pu239  = 0,
                                counts_from_Pu240  = 0,
                                
                                counts_from_Ra226  = 0,
                                counts_from_Se75   = 0,
                                counts_from_Sm153  = 0,
                                counts_from_Tc99m  = 0,
                                counts_from_Xe133  = 0,
                                counts_from_Tl201  = 0,
                                counts_from_Tl204  = 0,
                                counts_from_U233   = 0,
                                counts_from_U235   = 0,
    
                                counts_from_Back_Th = 0,
                                counts_from_Back_U  = 0,
                                counts_from_Back_K  = 0 
                                ):

    temp_key = np.zeros(32)
    
    temp_key[0] = counts_from_Am241
    temp_key[1] = counts_from_Ba133
    temp_key[2] = counts_from_Co57
    temp_key[3] = counts_from_Co60
    temp_key[4] = counts_from_Cs137
    temp_key[5] = counts_from_Cr51
    temp_key[6] = counts_from_Eu152
    temp_key[7] = counts_from_Ga67
    temp_key[8] = counts_from_I123
    temp_key[9] = counts_from_I125

    temp_key[10] = counts_from_I131
    temp_key[11] = counts_from_In111
    temp_key[12] = counts_from_Ir192
    temp_key[13] = counts_from_U238
    temp_key[14] = counts_from_Lu177m
    temp_key[15] = counts_from_Mo99
    temp_key[16] = counts_from_Np237
    temp_key[17] = counts_from_Pd103
    temp_key[18] = counts_from_Pu239
    temp_key[19] = counts_from_Pu240

    temp_key[20] = counts_from_Ra226
    temp_key[21] = counts_from_Se75
    temp_key[22] = counts_from_Sm153
    temp_key[23] = counts_from_Tc99m
    temp_key[24] = counts_from_Xe133
    temp_key[25] = counts_from_Tl201
    temp_key[26] = counts_from_Tl204
    temp_key[27] = counts_from_U233
    temp_key[28] = counts_from_U235

    temp_key[29] = counts_from_Back_Th
    temp_key[30] = counts_from_Back_U
    temp_key[31] = counts_from_Back_K


    return temp_key


def generate_random_source_key():
    background_cps = 85.0
    source1_cps = 10**np.random.uniform(1,3) 
    source2_cps = 10**np.random.uniform(1,3) 
    source3_cps = 10**np.random.uniform(1,3) 
    source4_cps = 10**np.random.uniform(1,3) 
    source5_cps = 10**np.random.uniform(1,3) 


    exposure_time = 10**np.random.uniform(1,3.3) 


    # counts_from_background = np.random.normal(background_cps,np.sqrt(background_cps))*exposure_time
    counts_from_background = background_cps*exposure_time
    counts_from_source1 = int(source1_cps*exposure_time)
    counts_from_source2 = int(source2_cps*exposure_time)
    counts_from_source3 = int(source3_cps*exposure_time)
    counts_from_source4 = int(source4_cps*exposure_time)
    counts_from_source5 = int(source5_cps*exposure_time)

    '''
    Th -> 60%
    U  -> 25%
    K  -> 15%
    '''
    counts_from_background_Th = int(counts_from_background*0.6)
    counts_from_background_U  = int(counts_from_background*0.25)
    counts_from_background_K  = int(counts_from_background*0.15)

    total_counts_in_spectrum = counts_from_source1+counts_from_source2+counts_from_source3+\
    counts_from_source4+counts_from_source5+counts_from_background_Th+\
    counts_from_background_U+counts_from_background_K

    max_isotopes_in_spectrum = np.random.randint(6)
    new_list = random.sample(range(0,29), max_isotopes_in_spectrum)


    temp_key = np.zeros(32)

    temp_key.flat[new_list] = [counts_from_source1,
                          counts_from_source2,
                          counts_from_source3,
                          counts_from_source4,
                          counts_from_source5]

    temp_key[-3] = counts_from_background_Th
    temp_key[-2] = counts_from_background_U
    temp_key[-1] = counts_from_background_K
    
    
    return temp_key
    
    
def create_simplex(number_samples,number_categories):
    # make an empty array
    k = np.zeros([number_samples,number_categories+1])
    # Make a sorted array of random variables
    a = np.sort(np.random.uniform(0,1,[number_samples,number_categories-1]),axis=1)
    
    # Zero pad left side
    k[:,0] = 0
    # Put sorted array in new array
    k[:,1:number_categories] = a
    # One pad right side
    k[:,number_categories] = 1

    # Take the difference of adjacent elements
    temp_simplex = np.diff(k)
    
    return temp_simplex


def shuffle_simplex(simplex):
    
    # last term from simpex is always background, by convention 
    temp = random.sample(simplex[:-1],len(simplex)-1)
    
    # 29 here because 29 isotopes plus one background super-isotope
    shuffled_array = np.pad(temp, [0,29-len(temp)], 'constant')

    np.random.shuffle(shuffled_array)

    # Add background
    shuffled_array = np.append(shuffled_array,simplex[-1])
    
    return shuffled_array
    
    
    
    
    
    
def visualize_simplex(key,index1,index2):
    new_list = []
    
    for i in range(len(key)):
        if key[i][index1]>0 and key[i][index2]>0:
            new_list.append([key[i][index1],key[i][index2]])
    
    
    plt.scatter(np.array(new_list)[:,0],np.array(new_list)[:,1])
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    
    return new_list
    
    
    
    
def generate_spectrum_from_key(temp_key,template_dict,calibration_setting):
    
   

    positive_indicies = [i[0] for i in np.argwhere(np.array(temp_key)>0)]
    computed_spectrum = np.zeros([1024])

    for k in range(len(positive_indicies)):
        temp_template_spectrum = zoom_spectrum(template_dict[isotopes[positive_indicies[k]]],calibration_setting)
        computed_spectrum+=sample_spectrum(temp_template_spectrum,temp_key[positive_indicies[k]])


    return computed_spectrum



def fun_generate_isotope_dataset(N_samples):
    '''
    Input: 
        - N_samples: number of total isotopes to generate
    '''


    Train_spectra = np.empty([N_samples,1024])
    Train_spectra_key = np.empty([N_samples,33]) 


    for j in range(N_samples):

        temp_key = generate_random_source_key()
        computed_spectrum = generate_spectrum_from_key(temp_key)
        
        Train_spectra[j]     = computed_spectrum
        Train_spectra_key[j] = temp_key
    
        print '\1b[2k\r',    
        print('Epoch %s of %s' %(j ,N_samples)),

        # Train_spectra, Train_spectra_key, counts

    return Train_spectra, Train_spectra_key

def RepresentsInt(s):
    '''
    Helper funtion to see if a string represents an integer    
    '''
    try: 
        int(s)
        return True
    except ValueError:
        return False

isotopes = [
      'Am241'   #00
    , 'Ba133'
    , 'Co57'
    , 'Co60'
    , 'Cs137'
    , 'Cr51'
    , 'Eu152'
    , 'Ga67'
    , 'I123'
    , 'I125'   #09
    
    , 'I131'   #10
    , 'In111'
    , 'Ir192'
    , 'U238'
    , 'Lu177m' 
    , 'Mo99'
    , 'Np237'
    , 'Pd103'
    , 'Pu239'
    , 'Pu240'  #19
    
    , 'Ra226'   #20
    , 'Se75'
    , 'Sm153'
    , 'Tc99m'
    , 'Xe133'
    , 'Tl201'
    , 'Tl204'
    , 'U233'
    , 'U235'   #28

    , 'Back_Th' #29
    , 'Back_U'  #30
    , 'Back_K'  #31
                 ]


isotopes_GADRAS_ID = [
      '241AM'   #00
    , '133BA'
    , '57CO'
    , '60CO'
    , '137CS'
    , '51CR'
    , '152EU'
    , '67GA'
    , '123I'
    , '125I'   #09
    
    , '131I'   #10
    , '111IN'
    , '192IR'
    , '238U'
    , '177MLU' 
    , '99MO'
    , '237NP'
    , '103PD'
    , '239PU'
    , '240PU'  #19
    
    , '226RA'   #20
    , '75SE'
    , '153SM'
    , '99TCM'
    , '133XE'
    , '201TL'
    , '204TL'
    , '233U'
    , '235U'   #28

    , 'ThoriumInSoil.spe' #29
    , 'UraniumInSoil.spe'  #30
    , 'PotassiumInSoil.spe'  #31
                 ]


def sample_spectrum(iso_DRF,ncounts):
    '''
    Input:
	isoDRF: the 1024x1 (or whatever) vector containing the spectrum to be sampled.  Does not need to be normalized.
    Output:
	ncounts: the 1024x1 (or whatever) vector containing the sampled spectrum.

    Method:
	Normalize isoDRF, and it is effectively a probability density function (pdf)
	Calculate the cumulative distribution function (cdf)
	Generate uniform random numbers to sample the cdf 
    '''

    pdf=iso_DRF/sum(iso_DRF)
    cdf = np.cumsum(pdf)
    
    # take random samples and generate spectrum
    t_all=np.random.rand(np.int(ncounts))
    spec=pdf*0
    for t in t_all:
        pos=np.argmax(cdf>t)
        spec[pos]=spec[pos]+1
    return spec

