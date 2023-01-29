import os
import numpy as np
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class SUADDDataset(CustomDataset):

    CLASSES = ( "WATER",
                "ASPHALT",
                "GRASS",
                "HUMAN",
                "ANIMAL",
                "HIGH_VEGETATION",
                "GROUND_VEHICLE",
                "FACADE",
                "WIRE",
                "GARDEN_FURNITURE",
                "CONCRETE",
                "ROOF",
                "GRAVEL",
                "SOIL",
                "PRIMEAIR_PATTERN",
                "SNOW")
                
    PALETTE2 = \
       ([ 148, 218, 255 ],  # light blue
        [  85,  85,  85 ],  # almost black
        [ 200, 219, 190 ],  # light green
        [ 166, 133, 226 ],  # purple    
        [ 255, 171, 225 ],  # pink
        [  40, 150, 114 ],  # green
        [ 234, 144, 133 ],  # orange
        [  89,  82,  96 ],  # dark gray
        [ 255, 255,   0 ],  # yellow
        [ 110,  87, 121 ],  # dark purple
        [ 205, 201, 195 ],  # light gray
        [ 212,  80, 121 ],  # medium red
        [ 159, 135, 114 ],  # light brown
        [ 102,  90,  72 ],  # dark brown
        [ 255, 255, 102 ],  # bright yellow
        [ 251, 247, 240 ])  # almost white

    PALETTE = \
       ([ 0, 0, 0 ],  # light blue
        [  1, 1, 1 ],  # almost black
        [ 2, 2, 2 ],  # light green
        [ 3, 3, 3 ],  # purple    
        [ 4, 4, 4 ],  # pink
        [  5, 5, 5 ],  # green
        [ 6, 6, 6 ],  # orange
        [  7, 7, 7 ],  # dark gray
        [ 8, 8, 8 ],  # yellow
        [ 9, 9, 9 ],  # dark purple
        [ 10, 10, 10 ],  # light gray
        [ 11, 11, 11 ],  # medium red
        [ 12, 12, 12 ],  # light brown
        [ 13, 13, 13 ],  # dark brown
        [ 14, 14, 14 ],  # bright yellow
        [ 255, 255, 255 ])  # almost white

    def __init__(self, split=None, **kwargs):
        super().__init__(img_suffix='.png', 
                         seg_map_suffix='.png', 
                         reduce_zero_label=False,
                         split=split, 
                         ignore_index=255, 
                         **kwargs)
        
    def format_results(self,
                       results,
                       imgfile_prefix,
                       indices=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix,indices)

        return result_files
    
    def results2img(self, results, imgfile_prefix, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        os.makedirs(imgfile_prefix, exist_ok=True)
        result_files = []
        for result, idx in zip(results, indices):
            #result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = os.path.splitext(os.path.basename(filename))[0]

            png_filename = os.path.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            palette = np.zeros((len(self.CLASSES), 3), dtype=np.uint8)
            for label_id in range(0, len(self.CLASSES)):
                palette[label_id] = self.PALETTE[label_id]

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
        return result_files