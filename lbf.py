# LBP 
# this script extract lbf feature 


from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
import cvutils

def lbf(infile):
    result = []
    label = []
    ix = 0

    for here, i in enumerate(open(infile).readlines()):
        ix +=1
        imgpath, l = i.split(',')
        if os.path.exists(imgpath):

            im = cv2.imread(imgpath)
            im  = cv2.resize(im, (200, 200))
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            radius = 3
            no_points = 8 * radius
            lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
            x = itemfreq(lbp.ravel())
            hist = x[:, 1]/sum(x[:, 1])
            result.append(hist)
            if "FE" in l:
                label.append(1) 
            else:
                label.append(-1) 
                
    print len(result)
    print len(label)
