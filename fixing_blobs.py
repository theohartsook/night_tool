from scipy_vision_tools import segmentDirectoryMP

if __name__ ==  '__main__':
    input_dir = '/Users/theo/data/fixing_blobs/input'
    output_root = '/Users/theo/data/fixing_blobs/output'
    num_workers = 2
    segmentDirectoryMP(input_dir, output_root, num_workers, target='.tif', save_intermediate=False, overwrite=True)