import os, sys
import cv2
import numpy as np

try:
    import seaborn as sns
    import pandas as pd
except:
    raise Exception('Seaborn or Pandas packages not found. Installation: $ pip install seaborn pandas')
    
def create_histogram(img):
    assert len(img.shape) == 2 
    histogram = [0] * 256 
    for row in range(img.shape[0]): 
        for col in range(img.shape[1]): 
            histogram[img[row, col]] += 1
    return histogram

def visualize_histogram(histogram, output='histogram.png'):
    hist_data = pd.DataFrame({'intensity': list(range(256)), 'frequency': histogram})
    sns_hist = sns.barplot(x='intensity', y='frequency', data=hist_data, color='blue')
    sns_hist.set(xticks=[]) 
    
    fig = sns_hist.get_figure()
    fig.savefig(output)
    return output

def equalize_histogram(img, histogram):
    new_H = [0] * 257
    for i in range(0, len(new_H)):
        new_H[i] = sum(histogram[:i])
    new_H = new_H[1:]
    max_value = max(new_H)
    min_value = min(new_H)
    new_H = [int(((f-min_value)/(max_value-min_value))*255) for f in new_H]   
    print("H':", new_H) 
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img[row, col] = new_H[img[row, col]]
    return img

if __name__ == "__main__":
    assert len(sys.argv) == 2, '[USAGE] $ python %s img_6.jpg' % (os.path.basename(__file__), INPUT)
    INPUT = sys.argv[1]
    assert os.path.isfile(INPUT), '%s not found' % INPUT
    img = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('grey_%s' % INPUT, img)
    print('Saved grayscale image @ grey_%s' % INPUT)
    histogram = create_histogram(img)
    print('histogram:', histogram)
    
    hist_img_path = visualize_histogram(histogram)
    print('Saved histogram @ %s' % hist_img_path)
    
    equalized_img = equalize_histogram(img, histogram)
    cv2.imwrite('equalized_%s' % INPUT, equalized_img)
    print('Saved equalized image @ equalized_%s' % INPUT)
    
    new_histogram = create_histogram(equalized_img)
    print('new_histogram:', new_histogram)
    hist_img_path = visualize_histogram(new_histogram, output='histogram_eq.png')
    print('Saved new histogram @ %s' % hist_img_path)	