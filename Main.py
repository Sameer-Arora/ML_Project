import argparse
import multiprocessing

from Sematic import Semantic
from gatyes import Gatyes
from Neural_Patches import Neural_patch



parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_arg = parser.add_argument
add_arg('--content', default=None, type=str )
add_arg('--style', default=None, type=str)
add_arg('--output', default='output.png', type=str)
add_arg('--output-size', default=None, type=str)
add_arg('--iterations', default=100, type=int)
add_arg('--device', default='cpu', type=str)
add_arg('--model', default='Gateys', type=str)
add_arg('--folder', default='samples', type=str)

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
        mod = Gatyes(device_name,int(args.iterations),dataset_folder_path=args.folder)

    if (args.model == "Neural_patch"):
        mod = Neural_patch(device_name,int(args.iterations),dataset_folder_path=args.folder)

    if (args.model == "Semantic_Style"):
        mod = Semantic(device_name,int(args.iterations),dataset_folder_path=args.folder)

    p = multiprocessing.Process(target=mod.run_tensorflow(args.content, args.style))
    p.start()
    p.join()

