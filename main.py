import argparse
from src.summary_by_api import genearte_summary_by_api

def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--create_label', required=False, action='store_true', help='create the summary as ground truth by API.')

    return parser.parse_args()

def main():

    args = get_argument()

    if args.create_label:
        genearte_summary_by_api()



if __name__ == "__main__":
    main()