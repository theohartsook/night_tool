from hough_tools import houghStep1, houghStep2, houghStep3


input_root = '/Users/theo/data/vision_pipeline_mp_test/output3'
temp_root = '/Users/theo/data/vision_pipeline_mp_test/temp_3'
pixel_dir = '/Users/theo/data/vision_pipeline_mp_test/pixel_coords'
original_img_dir = '/Users/theo/data/registered_rasters/1_37'
metrics_dir = '/Users/theo/data/vision_pipeline_mp_test/metrics_coords'
shp_dir = '/Users/theo/data/vision_pipeline_mp_test/shps'



if __name__ ==  '__main__':
    print('step 1')
    #houghStep1(input_root, temp_root, pixel_dir, original_img_dir, 4)
    print('step 2')
    houghStep2(original_img_dir, pixel_dir, metrics_dir, 16)
    print('step 3')
    houghStep3(metrics_dir, shp_dir)