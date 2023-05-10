import argparse
from run_pipeline import Pipeline

# Set up the command-line argument parser
parser = argparse.ArgumentParser(description='Train or test the pipeline.')
parser.add_argument('data_path', help='path to the data file')
parser.add_argument('--inference', action='store_true', help='run the pipeline in inference mode')
parser.add_argument('--output_path', default='predictions.json', help='path to the output file')

# Parse the command-line arguments
args = parser.parse_args()

# Instantiate the Pipeline class
pipeline = Pipeline()

# Run the pipeline in training or testing mode, depending on the `inference` argument
if not args.inference:
    # Training mode
    pipeline.run(args.data_path)
else:
    # Testing mode
    pipeline.run(args.data_path, inference=True, output_path=args.output_path)