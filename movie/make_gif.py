import sys
import os
import imageio as iio
import argparse

def go(input_path, output_path):
    
    if os.path.isfile(input_path):
        reader = iio.get_reader(input_path)
        fps = reader.get_meta_data()['fps']

        writer = iio.get_writer(output_path, fps=fps)
        for img in reader:
            writer.append_data(img)
        writer.close()

    elif os.path.isdir(input_path):
        images = []
        for file_name in sorted(os.listdir(input_path)):
            if file_name.endswith('.png'):
                file_path = os.path.join(input_path, file_name)
                images.append(iio.imread(file_path))
        iio.mimsave(output_path, images)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", metavar="i", type=str,
            help="Input filepath for movie file.")
    parser.add_argument("output_path", metavar="o", type=str,
            help="Output filepath for gif file.")

    args = parser.parse_args()
    go(args.input_path, args.output_path)
