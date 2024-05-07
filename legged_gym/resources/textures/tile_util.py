from PIL import Image
import random
from tqdm import tqdm

from legged_gym import LEGGED_GYM_ROOT_DIR

angles = [0, 90, 180, 270]

def create_tiled_image(image_paths, out_dir):
    for image_path in tqdm(image_paths):
        tile_count_x = random.randint(1, 5)
        tile_count_y = random.randint(1, 5)

        # Load the image
        original_image = Image.open(image_path)
        width, height = original_image.size
        #width, height = 3000, 2000

        # Calculate the size of each tile
        tile_width = width // tile_count_x
        tile_height = height // tile_count_y

        # Create a new image with the same resolution as the original
        new_image = Image.new('RGB', (width, height))

        # Iterate through the tiles and paste the resized original image
        for x in range(tile_count_x):
            for y in range(tile_count_y):

                # rotate the image
                angle = random.choice(angles)
                original_image = original_image.rotate(angle, expand=True)
                
                resized_image = original_image.resize((tile_width, tile_height))
                new_image.paste(resized_image, (tile_width * x, tile_height * y))

                rand_image = random.choice(image_paths)
                original_image = Image.open(rand_image)

        # Save the new tiled image
        new_image.save(image_path.replace('regular', out_dir))

if __name__ == '__main__':
    import os
    image_paths = os.listdir(LEGGED_GYM_ROOT_DIR+'/resources/textures/regular')
    image_paths = [LEGGED_GYM_ROOT_DIR+'/resources/textures/regular/'+image_path for image_path in image_paths]
    # image_path = 'ice_texture.jpg'  # Replace with the path to your image (PNG or JPG)

    create_tiled_image(image_paths[:250], 'train_tiled')
    create_tiled_image(image_paths[250:], 'test_tiled')