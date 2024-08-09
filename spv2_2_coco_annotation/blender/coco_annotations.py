import bpy  # type: ignore
import json
import os
import sys
from bpy_extras.object_utils import world_to_camera_view  # type: ignore
import mathutils  # type: ignore
import numpy as np  # We will use numpy for RLE encoding
import argparse

class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())



autoconfig = True
print(f"Autoconfig : {autoconfig}")
##### AUTOMATED CONFIGURATION #####
if autoconfig:
    parser = ArgumentParserForBlender()
    parser.add_argument('--root', type=str, required=True, help='Path to the root dir')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--render-mask', type=bool, default=False, help='Render segmentation mask')
    parser.add_argument('--image-height', type=int, default=200, help='Image height')
    parser.add_argument('--image-width', type=int, default=320, help='Image width')
    args = parser.parse_args()
    ROOT_PATH = args.root
    FILE = args.input_file
    OUTPUT_PATH = args.output_file
    RENDER_MASK = args.render_mask
    IMAGE_SIZE = [int(args.image_height), int(args.image_width)]
##### END OF AUTOMATED CONFIGURATION #####

else : 
    ##### MANUAL CONFIGURATION ##### 
    ### PARAMETERS ###
    RENDER_MASK = False
    ### END OF PARAMETERS ###

    ### CONFIGURATION ###
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    PROJECT_DIR = 'data/src/speedplusv2/synthetic'
    POSES_FILE_NAME = "train.json"
    OUTPUT_DIR = 'data/coco/synthetic/annotations/'
    SAVE_FILE_NAME = POSES_FILE_NAME 
    FILE = os.path.join(ROOT_PATH, PROJECT_DIR, POSES_FILE_NAME)
    OUTPUT_PATH = os.path.join(ROOT_PATH, OUTPUT_DIR, POSES_FILE_NAME)
    ### END OF CONFIGURATION ###
    ##### END OF MANUAL CONFIGURATION #####

class CocoProcessor:
    def __init__(self,json_data) :
        # Set up parameters for the satellite object
        self.coco_file = dict({ 'images': [], 'annotations': [], 'categories': []})
        self.keypoints_name = ['body_keypoint_1', 'body_keypoint_2', 'body_keypoint_3', 'body_keypoint_4', 'panel_keypoint_1', 'panel_keypoint_2', 'panel_keypoint_3', 'panel_keypoint_4', 'antenna_keypoint_1', 'antenna_keypoint_2', 'antenna_keypoint_3']
        self.image_height = IMAGE_SIZE[0]
        self.image_width = IMAGE_SIZE[1]


        self.poses = json_data

        # Reference to the satellite mesh object
        self.satellite_object = bpy.data.objects['satellite_object']
        bpy.context.view_layer.objects.active = self.satellite_object
        self.satellite_object.select_set(True)

        # Set up rendering parameters
        bpy.context.scene.world.color = (0, 0, 0)  # Set the world background color to black
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        bpy.context.scene.display.shading.light = 'FLAT'
        bpy.context.scene.display.shading.color_type = 'MATERIAL'
        bpy.context.scene.render.image_settings.file_format = 'JPEG'
        bpy.context.scene.render.resolution_x = self.image_width
        bpy.context.scene.render.resolution_y = self.image_height

        # Scene setup
        self.scene = bpy.context.scene
        self.cam = bpy.data.objects['Camera']
    
    def getDatasetInfo(self):
        """ Get dataset information. """
        dataset_info = dict({
            "description": "SII annotations of SPEED+ spacecraft dataset",
        "url": "example.com",
        "version": "1.0",
        "year": 2024,
        "contributor": "Matthieu MARCHAL",
        "date_created": "2024/05/27"})
        return dataset_info
    
    def getLicenseInfo(self):
        """ Get license information. """
        license_info = [dict({
            'id': 1,
            'name': 'All Rights Reserved',
            'url': 'example.com',
        })]
        return license_info
    

    def getImageInfo(self,pose,id):
        """ Get image information from the pose data. """
        image_info = dict({
            'file_name': pose['filename'],
            'height': self.image_height,
            'width': self.image_width,
            'id': id
        })
        return image_info
    
    def getAnnotations(self,pose,id):
        """ Process pose to extract keypoints and render segmentation masks. """
        # Update object location and rotation based on pose data
        self.satellite_object.location = pose['r_Vo2To_vbs_true']
        self.satellite_object.rotation_mode = 'QUATERNION'
        self.satellite_object.rotation_quaternion = pose['q_vbs2tango_true']
        
        # Prepare camera and scene objects
        bpy.context.view_layer.update()

        # Extract keypoint coordinates
        keypoints,nl= self.getkeypoints()
        bbox,area = self.getBoundingBox(keypoints)

        # Render segmentation mask 
        segmentation_mask = ''
        if RENDER_MASK : 
            segmentation_mask = self.getSegmentationMask() # add mask informations in dataset. WARNING : Can take long time 

        # Append results to the list
        
        annotation = dict({
            'id': id,
            'image_id': id,
            "category_id": 1,  # "satellite"
            'segmentation': segmentation_mask,
            'area': area,
            'bbox': bbox,
            'iscrowd': 0, # WARNING Default  = 1 (RLE format of mask) but 0 to work with for YOLO convert_coco function
            'keypoints': keypoints,
            'num_keypoints': nl,
            'pose_translation' : pose['r_Vo2To_vbs_true'],
            'pose_quaternion' : pose['q_vbs2tango_true']
        })
        return annotation
    
    def getkeypoints(self):
        # Initialize keypoint details
        keypoints = [bpy.data.objects[name] for name in self.keypoints_name]
        keypoint_info = []
        nl = 0
        for keypoint in keypoints:
            world_location = keypoint.matrix_world.translation
            coord = world_to_camera_view(self.scene, self.cam, world_location)
            x, y_flipped = coord.x, coord.y
            y = 1 - y_flipped  # flip y-coordinate to have correct 2D frame
            
            # Get the depsgraph
            depsgraph = bpy.context.evaluated_depsgraph_get()

            # Set up the ray casting from the camera
            ray_origin = self.cam.location
            ray_target = world_location  # keypoint's world location
            ray_direction = (ray_target - ray_origin).normalized()

            # Perform the ray cast
            result, location, normal, index, hit_object, matrix = bpy.context.scene.ray_cast(depsgraph, ray_origin, ray_direction)

            threshold = 0.01  # Minimum significant distance for an obstruction
            distance_to_hit = (location - ray_origin).length
            
            if 0<x<1 and 0<y<1:
                if distance_to_hit > threshold:
                    if hit_object != keypoint:
                        v = 1  #obstructs view of {keypoint.name}")
                        nl+=1
                    else:
                        v = 2 # visible despite initial self-hit
                        nl+=1
                else:
                    v = 2 # too close to be considered an obstruction
                    nl+=1
            else:
                x,y,v=0,0,0
            # concatenate the keypoint info
            x_pixel = int(x * self.image_width)
            y_pixel = int(y * self.image_height)
            keypoint_info.append(x_pixel)
            keypoint_info.append(y_pixel)
            keypoint_info.append(v)
            
        return keypoint_info,nl

    def getBoundingBox(self,keypoints):
        """Calculate the bounding box from the visible keypoints in COCO format.
        
        Args:
        keypoints (list): List of keypoints in the format [x1, y1, v1, x2, y2, v2, ...]
        
        Returns:
        list: Bounding box [x_min, y_min, width, height], or [0, 0, 0, 0] if no visible points.
        float: Area of the bounding box.
        """
        # Filter out keypoints where visibility v is 0 (not labeled or not visible)
        visible_points = [(keypoints[i], keypoints[i+1]) for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0]

        if not visible_points:
            return [0, 0, 0, 0]  # No visible points, return an empty box

        # Unzip the list of tuples into two lists, one for x coordinates and one for y coordinates
        x_values, y_values = zip(*visible_points)

        # Calculate the minimum and maximum x and y coordinates
        min_x, max_x = int(min(x_values)), int(max(x_values))
        min_y, max_y = int(min(y_values)), int(max(y_values))

        # Compute width and height of the bounding box
        width = int(max_x - min_x)
        height = int(max_y - min_y)

        # Compute the area of the bounding box
        area = int(width * height)

        return [min_x, min_y, width, height], area
    
    def getSegmentationMask(self):
        """Render the segmentation mask for the object."""
        
        # Render the scene
        bpy.ops.render.render(write_still=True)
        
        # Process the render result
        render_result = bpy.data.images.get('Render Result')
        if render_result is not None:
            full_path =  os.path.join(ROOT_PATH, OUTPUT_DIR, 'temporary_mask.jpg')
            render_result.save_render(filepath=full_path)
            image = bpy.data.images.load(full_path)
            pixels = np.array(image.pixels)

            mask = (pixels[::4] > 0.3).astype(int).reshape(image.size[0], image.size[1])
            mask = self.nearestNeighborResize(mask,[image.size[0],image.size[1]])
            rle_mask = self.encodeRLE(mask)

            mask_data = dict({"size": [self.image_height, self.image_width], "counts": rle_mask})
            
            # Clean up by removing the image from memory
            bpy.data.images.remove(image)
            return mask_data
 
    def nearestNeighborResize(self,img, new_shape):
        """
        Resize the given binary image using nearest neighbor interpolation.
        img: Binary image (numpy array)
        new_shape: tuple of (new_height, new_width)
        """
        old_shape = img.shape
        row_ratio, col_ratio = old_shape[0] / new_shape[0], old_shape[1] / new_shape[1]

        # Compute the mapping from output pixels to input pixels
        new_row_indices = (np.arange(new_shape[0]) * row_ratio).astype(int)
        new_col_indices = (np.arange(new_shape[1]) * col_ratio).astype(int)

        # Map the input pixels to output pixels
        resized_img = img[new_row_indices[:, None], new_col_indices]

        return resized_img
    def encodeRLE(self,mask):
        """Encode a binary mask using RLE encoding.
        
        Args:
        mask (np.array): Binary mask as a numpy array.

        Returns:
        [number of pixels, number of pixels, ...]: Encoded RLE mask as a list.
        """
        # Flip the mask vertically to match top-left origin systems
        mask = np.flipud(mask)
        mask = mask.T
        # Flatten the mask
        mask = mask.flatten()
        # Initialize variables
        rle = []
        count = 1
        color = 0
        # Iterate over the mask
        for i in range(1, len(mask)):
            if mask[i] == color:
                count += 1
            else:
                rle.append(count)
                color = mask[i]
                count = 1
        rle.append(count)
        return rle
    
    def getCategories(self):
        categories = dict({
            "id": 1, # "satellite"
            "name": "satellite",
            "supercategory": "object",
            "keypoints": ["body_1", "body_2", "body_3", "body_4", "panel_1", "panel_2", "panel_3", "panel_4", "antenna_1", "antenna_2", "antenna_3"],
            "skeleton" : [[1,2],[2,3],[3,4],[1,4],[5,6],[6,7],[7,8],[5,8],[1,5],[2,6],[3,7],[4,8],[5,9],[6,10],[7,11]]
        })
        return categories
    
    def process(self):
        self.coco_file['info'] = self.getDatasetInfo()
        self.coco_file['licenses'] = self.getLicenseInfo()
        id = 0
        
        poses = self.poses
        for pose in poses:
            id+=1
            #pose = pose['pose']
            self.coco_file['images'].append(self.getImageInfo(pose,id))
            self.coco_file['annotations'].append(self.getAnnotations(pose,id))
        
        self.coco_file['categories'].append(self.getCategories())
        return self.coco_file

if __name__ == "__main__":

    print(f"Processing {FILE}")
    with open(FILE, 'r') as file:
        data = json.load(file)
    
    
    coco_processor = CocoProcessor(data)
    coco_file = coco_processor.process()
    
    with open(OUTPUT_PATH, 'w') as file:
        json.dump(coco_file, file)
    print(f"Blender results saved to {OUTPUT_PATH}")
