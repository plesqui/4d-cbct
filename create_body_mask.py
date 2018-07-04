import SimpleITK as sitk

def create_body_mask(img, seed, thr = 0.012):
    """
    Create a binary mask of the body of the patient.
    INPUTS: img - 2d/3d SimpleITK object; seed - list of tuple(s) representing the initial 2d/3d indices of the region growing seed(s)
            thr - upper threshold value to differentiate tissue from background (default is 0.012). This thr is specific for the dataset 
            used in this study!
    OUTPUT: masked_img - binary mask
    """
    maskval = 10 # Value of the masked pixels.
    
    # First, smooth out the image
    imgSmooth = sitk.CurvatureFlow(img, timeStep = 0.125, numberOfIterations = 5) # Default
    
    # Mask the region using region growing. Please note we are masking the background region instead of the patient. This was chosen because
    # the background region has very uniform pixel values and connected (it is air in the patient images).
    masked_original = sitk.ConnectedThreshold(imgSmooth, seedList = seed, lower = -1, upper = thr, replaceValue = maskval)
    
    # Fill holes
    masked_img = sitk.VotingBinaryHoleFilling(masked_original, radius=[20]*10, majorityThreshold=1, backgroundValue=0, foregroundValue = maskval)
    
    return masked_img
