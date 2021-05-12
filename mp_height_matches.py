from utility_tools import heightMatchOnDirMP

base_dir = '/Users/theo/data/hough_dataset/predicted'
two_dir = '/Users/theo/data/hough_dataset/augmented_pixel_coords_2'
two_out_dir = '/Users/theo/data/hough_dataset/mp_oof_2'

if __name__ ==  '__main__':
    heightMatchOnDirMP(base_dir, two_dir, two_out_dir, 2)