import argparse
import multiprocessing

from gatyes import Gatyes
from Neural_Patches import Neural_patch

parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_arg = parser.add_argument
add_arg('--content', default=None, type=str, help='Content image path as optimization target.')
add_arg('--style', default=None, type=str, help='Style image path to extract patches.')
add_arg('--output', default='output.png', type=str, help='Output image path to save once done.')
add_arg('--output-size', default=None, type=str, help='Size of the output image, e.g. 512x512.')
add_arg('--iterations', default=100, type=int, help='Number of iterations to run each resolution.')
add_arg('--device', default='cpu', type=str, help='Index of the GPU number to use, for theano.')
add_arg('--model', default='cpu', type=str, help='Index of the GPU number to use, for theano.')

args = parser.parse_args()

if __name__ == "__main__":
    print("Welcome to the neural style transfer!!")
    print("Slected Model", args.model, " Selected device ", args.device)
    print("Selected Content Image ", args.content, " Selected Style Image ", args.style)

    if args.device == "gpu":
        device_name = "/gpu:0"

    if args.device == "cpu":
        device_name = "/cpu:0"

    if (args.model == "Gateys"):
        mod = Gatyes(device_name)

    if (args.model == "Neural_patch"):
        mod = Neural_patch(device_name)

    if (args.model == "Semantic_Style"):
        mod = Semantic(device_name)

    p = multiprocessing.Process(target=mod.run_tensorflow(args.content, args.style))
    p.start()
    p.join()

