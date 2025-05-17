import os
from result_writer import ResultWriter
from sentence_reader import SentenceReader
from config import Config
from logger import logger


def main():
    config = Config('config.yaml')
    config.work_dir = os.path.join(config.work_dir, 'pretrained')
    result_writer = ResultWriter(
        config=config
    )

    fold_dirs_to_predict = result_writer.discover_fold_work_dirs(
        main_work_dir=config.work_dir)

    if fold_dirs_to_predict:
        result_writer.run_prediction_and_ensembling_pipeline(
            fold_work_dirs=fold_dirs_to_predict)
    else:
        logger.warning(
            "No fold directories found or specified. Skipping prediction and ensembling.")


if __name__ == "__main__":
    main()
