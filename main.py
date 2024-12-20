import argparse
from src.summary_by_api import genearte_summary_by_api
from fine_tuning.utils import load_config_from_yaml
from fine_tuning.ft_configs import FineTuningConfig
from fine_tuning.fine_tuning import peft_fine_tuning



def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--create_label', required=False, action='store_true', help='create the summary as ground truth by API.')
    parser.add_argument('--peft', required=False, action='store_true', help='fine-tuning.')

    return parser.parse_args()

def main():

    args = get_argument()

    if args.create_label:
        genearte_summary_by_api()

    elif args.peft:
        config_path = './fine_tuning/yamls/test.yaml'
        qlora_config = load_config_from_yaml(config_path, FineTuningConfig)
        peft_fine_tuning(config=qlora_config)


if __name__ == "__main__":
    main()