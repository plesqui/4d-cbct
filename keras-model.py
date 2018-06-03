def identity_down(X, f, filters, block):
    """
    Implementation of a generic identity block (downward) as defined in Figure 1
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    X_shortcut -- input tensor of shape
    f -- integer, specifying the shape of the  CONV's window for the main path (for the original U-net all kernels are 3x3)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block (downward pass), tensor of shape (n_H, n_W, n_C)

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
