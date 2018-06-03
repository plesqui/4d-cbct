# The building blocks of the network: 

def identity_down(X, f, filters, block):
    """
    Implementation of the generic downward (encoding) block as defined in Figure 1
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    X_shortcut -- input tensor of shape
    f -- integer, specifying the shape of the  CONV's window for the main path (for the original U-net all kernels are 3x3)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the block (downward pass), tensor of shape (n_H, n_W, n_C)

    """
    
    # Defining the name
    conv_name_base = 'JPNet-down_CNV_' + block
    bn_name_base = 'JPNet-down_BN_' + block
 
    
    # Retrieve Filters
    F1, F2 = filters
    
    # First component
    X = Conv2D(filters = F1, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'A', kernel_initializer = he_normal())(X)
    X = LeakyReLU()(X)
    #X = Activation('relu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'A')(X)
    
    # Second component
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'B', kernel_initializer = he_normal())(X)
    X = LeakyReLU()(X)
    #X = Activation('relu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'B')(X)
    
    # Store Features to be transferred to the corresponding layer in the upward pass of the U-NET
    X_shortcut = X 
    
    # Third component to down-sample the net (we keep the same filter as in the previous conv-layer)
    X = Conv2D(filters = F2, kernel_size = (2, 2), strides = (2, 2), padding = 'valid', name = conv_name_base + 'C', kernel_initializer = he_normal())(X)
    X = LeakyReLU()(X)
    #X = Activation('leakyRelu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'C')(X)
    
    return X, X_shortcut

def bottom(X, f, filters, block):
    """
    Implementation of the bottom block of the network as defined in Figure 1
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the  CONV's window for the main path (for the original U-net all kernels are 3x3)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the block (bottom), tensor of shape (n_H, n_W, n_C)

    """
    
    # Defining the name
    conv_name_base = 'JPNet-bottom_CNV_' + block
    bn_name_base = 'JPNet-bottom_BN_' + block

    
    # Retrieve Filters
    F1, F2, F3 = filters
       
    
    # First component
    X = Conv2D(filters = F1, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'A', kernel_initializer = he_normal())(X)   
    X = LeakyReLU()(X)
    #X = Activation('leakyRelu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'A')(X)
    
    # Second component
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'B', kernel_initializer = he_normal())(X)
    X = LeakyReLU()(X)
    #X = Activation('leakyRelu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'B')(X)
    
    # Third component: Resize features by a factor of 2 using Nearest Neighbor + 2DConvolution
    X = Lambda(resize_img)(X)
    #X = K.resize_images(X, 2, 2, 'channels_last')
    X = Conv2D(filters = F3, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'C', kernel_initializer = he_normal())(X)  
    X = LeakyReLU()(X)
    #X = Activation('leakyRelu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'C')(X)
    
    return X

def identity_up(X, X_shortcut, f, filters, block):
    """
    Implementation of the generic upward block (decoding) as defined in Figure 1
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    X_shortcut -- tensor of shape ( , , ) that is transferred from the downpass to the up-pass of the network
    f -- integer, specifying the shape of the  CONV's window for the main path (for the original U-net all kernels are 3x3)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the block (upward pass), tensor of shape (n_H, n_W, n_C)

    """
    
    # Defining the name
    conv_name_base = 'JPNet-up_CNV_' + block
    bn_name_base = 'JPNet-up_BN_' + block
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Concatenate the X_shortcut to the Input from the previous block (or I can do this outside)
    X = Concatenate()([X_shortcut, X])
    
    
    # First component
    X = Conv2D(filters = F1, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'A', kernel_initializer = he_normal())(X)
    X = LeakyReLU()(X)
    #X = Activation('leakyRelu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'A')(X)
    
    # Second component
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'B', kernel_initializer = he_normal())(X)
    X = LeakyReLU()(X)
    #X = Activation('leakyRelu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'B')(X)

    
    # Third component: Resize features by a factor of 2 using Nearest Neighbor + 2D Convolution
    X = Lambda(resize_img)(X)
    #X = K.resize_images(X, 2, 2, 'channels_last')
    X = Conv2D(filters = F3, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'C', kernel_initializer = he_normal())(X) 
    X = LeakyReLU()(X)
    #X = Activation('leakyRelu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'C')(X)
    
    return X

def output_block(X, X_shortcut, f, filters, block):
    """
    Implementation of the output block as defined in Figure 1
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    X_shortcut -- tensor of shape ( , , ) that is transferred from the downpass to the up-pass of the network
    f -- integer, specifying the shape of the  CONV's window for the main path (for the original U-net all kernels are 3x3)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- final image output 

    """
    
    # Defining the name
    conv_name_base = 'JPNet-output_CNV_' + block
    bn_name_base = 'JPNet-output_BN_' + block
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Concatenate the X_shortcut to the Input from the previous block (or I can do this outside)
    X = Concatenate()([X_shortcut, X])
    
    
    # First component
    X = Conv2D(filters = F1, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'A', kernel_initializer = he_normal())(X)
    X = LeakyReLU()(X)
    #X = Activation('leakyRelu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'A')(X)
    
    # Second component
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + 'B', kernel_initializer = he_normal())(X)
    X = LeakyReLU()(X)
    #X = Activation('leakyRelu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + 'B')(X)
    
    # Third component
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'same', name = conv_name_base + 'C', kernel_initializer = he_normal())(X)
               
    return X

# Building the Network
def JPNet(input_shape = (448, 448, 1)):
    """
    Implementation of the JPNet (Joel-Pedro Net, based on the U-Net)
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() in Keras
    """
    # Size of all the filters (kernels) in original U-network is 3x3
    f = 3

    # Number of channels in each stage and convolutional layer

    # Contracting phase
    filters_0_down = [64, 64]
    filters_1_down = [128, 128]
    filters_2_down = [256, 256]
    filters_3_down = [512, 512]

    # Bottom
    filters_bottom = [1024, 1024, 512]

    # Expanding phase
    filters_3_up = [512, 512, 256]
    filters_2_up = [256, 256, 128]
    filters_1_up = [128, 128, 64]

    # Ouput block
    filters_0_up = [64, 64, 1] # Note that the last layer should be a 2D image (the output), so there should be only 1 channel.

    # Defining the block or Stages
    block = ["0","1","2","3","botton"]
        
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
      
    # Contracting phase
    X, X_shortcut_0 = identity_down(X_input, f, filters_0_down, block[0])
    X, X_shortcut_1 = identity_down(X, f, filters_1_down, block[1])
    X, X_shortcut_2 = identity_down(X, f, filters_2_down, block[2])
    X, X_shortcut_3 = identity_down(X, f, filters_3_down, block[3])
    
    # Bottom
    X = bottom(X, f, filters_bottom, block[4])
    
    # Expanding phase
    X = identity_up(X, X_shortcut_3, f, filters_3_up, block[3])
    X = identity_up(X, X_shortcut_2, f, filters_2_up, block[2])
    X = identity_up(X, X_shortcut_1, f, filters_1_up, block[1])
    
    # Output block
    X = output_block(X, X_shortcut_0, f, filters_0_up, block[0])
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name = 'JPNet')
    
    return model

# Additional function
def resize_img(input_tensor): # resizes input tensor wrt. ref_tensor
    return K.resize_images(input_tensor, 2, 2 , 'channels_last')

# We now build the model's computational graph and compile
model = JPNet(input_shape = (448, 448, 1))
Nebbie = optimizers.Adam(lr = 0.0001)
model.compile(optimizer = Nebbie, loss = combined)
