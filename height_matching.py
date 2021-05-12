from utility_tools import heightMatchOnDirMP, bulkAssignIDs

base_dir = '/Users/theo/data/hough_dataset/predicted'
two_dir = '/Users/theo/data/hough_dataset/augmented_pixel_coords_2'
three_dir = '/Users/theo/data/hough_dataset/augmented_pixel_coords_3'
four_dir = '/Users/theo/data/hough_dataset/augmented_pixel_coords_4'
five_dir = '/Users/theo/data/hough_dataset/augmented_pixel_coords_5'

two_out_dir = '/Users/theo/data/hough_dataset/oof_2'
three_out_dir = '/Users/theo/data/hough_dataset/oof_3'
four_out_dir = '/Users/theo/data/hough_dataset/oof_4'
five_out_dir = '/Users/theo/data/hough_dataset/oof_5'

if __name__ ==  '__main__':
    heightMatchOnDirMP(base_dir, two_dir, two_out_dir, 2)
    heightMatchOnDirMP(two_out_dir, three_dir, three_out_dir, 3)
    heightMatchOnDirMP(three_out_dir, four_dir, four_out_dir, 4)
    heightMatchOnDirMP(four_out_dir, five_dir, five_out_dir, 5)